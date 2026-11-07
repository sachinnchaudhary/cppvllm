#include "kernels/fused_ops.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstdint>

namespace fused {
namespace {

constexpr int kBlockThreads = 256;   // threads per row in the norm kernels

__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T> __device__ __forceinline__ T from_float(float x);
template <> __device__ __forceinline__ float from_float<float>(float x) { return x; }
template <> __device__ __forceinline__ __half from_float<__half>(float x) { return __float2half(x); }
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

template <typename T> __device__ __forceinline__ float rnd(float x) { return to_float(from_float<T>(x)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}


__device__ __forceinline__ uint8_t to_e4m3(float x) {
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}


__device__ __forceinline__ float block_sum(float v) {
    __shared__ float part[32];
    v = warp_sum(v);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, n_warps = blockDim.x / 32;
    if (lane == 0) part[warp] = v;
    __syncthreads();
    float r = 0.f;
    for (int i = 0; i < n_warps; ++i) r += part[i];    
    __syncthreads();                                   
    return r;
}

// ---- rms_norm / add_rms_norm --------------------------------------------------------------------------------------

template <typename T, bool ADD>
__global__ void rms_norm_kernel(T* __restrict__ out, T* resid, const T* __restrict__ x, const T* __restrict__ w, int D,
                                float eps) {
    const int64_t row = blockIdx.x;
    const T* xin = x + row * D;
    T* res = ADD ? resid + row * D : nullptr;
    T* o = out + row * D;

    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v;
        if constexpr (ADD) {
            v = rnd<T>(to_float(res[i]) + to_float(xin[i]));
            res[i] = from_float<T>(v);
        } else {
            v = to_float(xin[i]);
        }
        ss += v * v;
    }
    ss = block_sum(ss);
    const float inv = rsqrtf(ss / static_cast<float>(D) + eps);

    // second pass: each thread re-reads exactly the elements it wrote/read in the first pass (served from cache)
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float v = ADD ? to_float(res[i]) : to_float(xin[i]);
        o[i] = from_float<T>(rnd<T>(v * inv) * to_float(w[i]));
    }
}

template <typename T, bool ADD>
void launch_norm(at::Tensor& out, at::Tensor* resid, const at::Tensor& x, const at::Tensor& w, double eps) {
    const int64_t N = x.size(0);
    const int D = static_cast<int>(x.size(1));
    rms_norm_kernel<T, ADD><<<static_cast<unsigned>(N), kBlockThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<T*>(out.data_ptr()), ADD ? reinterpret_cast<T*>(resid->data_ptr()) : nullptr,
        reinterpret_cast<const T*>(x.data_ptr()), reinterpret_cast<const T*>(w.data_ptr()), D, static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

constexpr int kQkvWarps = 4;

constexpr float kE4m3Max = 448.f;

template <typename T, int HD>
__global__ void qk_norm_rope_kv_fp8_kernel(T* __restrict__ q_out, const T* __restrict__ qkv, const T* __restrict__ q_norm,
                                           const T* __restrict__ k_norm, const float* __restrict__ cos_t,
                                           const float* __restrict__ sin_t, const int64_t* __restrict__ positions,
                                           const int64_t* __restrict__ slots, uint8_t* __restrict__ k_cache,
                                           uint8_t* __restrict__ v_cache, float* __restrict__ k_scale,
                                           float* __restrict__ v_scale, T* __restrict__ k_dense, T* __restrict__ v_dense,
                                           int H, int KV, float eps) {
    constexpr int EPL = HD / 32;
    constexpr int HALF = HD / 2;
    static_assert(EPL % 2 == 0, "head_dim must be a multiple of 64");

    const int lane = threadIdx.x % 32;
    const int task = blockIdx.y * kQkvWarps + threadIdx.x / 32;
    const int n_tasks = H + 2 * KV;
    if (task >= n_tasks) return;

    const int64_t t = blockIdx.x;
    const T* src = qkv + (t * n_tasks + task) * HD;
    const int64_t slot = slots[t];

    if (task >= H + KV) {                              // v: quantise
        const int kvh = task - H - KV;
        float x[EPL], amax = 0.f;
#pragma unroll
        for (int j = 0; j < EPL; ++j) { x[j] = to_float(src[lane + 32 * j]); amax = fmaxf(amax, fabsf(x[j])); }
        amax = warp_max(amax);
        const float sc = fmaxf(amax, 1e-8f) / kE4m3Max;
        const float inv = 1.f / sc;
        uint8_t* dst = v_cache + (slot * KV + kvh) * HD;
#pragma unroll
        for (int j = 0; j < EPL; ++j) dst[lane + 32 * j] = to_e4m3(x[j] * inv);
        if (lane == 0) v_scale[((slot >> 4) * KV + kvh) * 16 + (slot & 15)] = sc;   // [block, KV, 16] layout
        if (v_dense != nullptr) {
            T* d = v_dense + (t * KV + kvh) * HD;
#pragma unroll
            for (int j = 0; j < EPL; ++j) d[lane + 32 * j] = src[lane + 32 * j];
        }
        return;
    }

    const bool is_q = task < H;
    const T* wn = is_q ? q_norm : k_norm;

    float x[EPL], ss = 0.f;
#pragma unroll
    for (int j = 0; j < EPL; ++j) { x[j] = to_float(src[lane + 32 * j]); ss += x[j] * x[j]; }
    ss = warp_sum(ss);
    const float inv = rsqrtf(ss / static_cast<float>(HD) + eps);

    float n[EPL];
#pragma unroll
    for (int j = 0; j < EPL; ++j) n[j] = rnd<T>(rnd<T>(x[j] * inv) * to_float(wn[lane + 32 * j]));
    const int64_t pos = positions[t];
    float r[EPL];
#pragma unroll
    for (int j = 0; j < EPL / 2; ++j) {
        const int e = lane + 32 * j;
        const float c = cos_t[pos * HALF + e], s = sin_t[pos * HALF + e];
        const float a = n[j], b = n[j + EPL / 2];
        r[j]            = a * c - b * s;
        r[j + EPL / 2]  = b * c + a * s;
    }

    if (is_q) {
        T* dst = q_out + (t * H + task) * HD;
#pragma unroll
        for (int jj = 0; jj < EPL; ++jj) dst[lane + 32 * jj] = from_float<T>(r[jj]);
        return;
    }

    const int kvh = task - H;                          // k: quantise the rotated, T-rounded values
    float amax = 0.f;
#pragma unroll
    for (int jj = 0; jj < EPL; ++jj) { r[jj] = rnd<T>(r[jj]); amax = fmaxf(amax, fabsf(r[jj])); }
    amax = warp_max(amax);
    const float sc = fmaxf(amax, 1e-8f) / kE4m3Max;
    const float sinv = 1.f / sc;
    uint8_t* dst = k_cache + (slot * KV + kvh) * HD;
#pragma unroll
    for (int jj = 0; jj < EPL; ++jj) dst[lane + 32 * jj] = to_e4m3(r[jj] * sinv);
    if (lane == 0) k_scale[((slot >> 4) * KV + kvh) * 16 + (slot & 15)] = sc;       // [block, KV, 16] layout
    if (k_dense != nullptr) {
        T* d = k_dense + (t * KV + kvh) * HD;
#pragma unroll
        for (int jj = 0; jj < EPL; ++jj) d[lane + 32 * jj] = from_float<T>(r[jj]);
    }
}
template <typename T, int HD>
void launch_qkv_fp8(at::Tensor& q_out, const at::Tensor& qkv, const at::Tensor& qn, const at::Tensor& kn,
                    const at::Tensor& cos_t, const at::Tensor& sin_t, const at::Tensor& pos, const at::Tensor& slots,
                    at::Tensor& kc, at::Tensor& vc, at::Tensor& ks, at::Tensor& vs, at::Tensor& kd, at::Tensor& vd,
                    int H, int KV, float eps) {
    const int n_tasks = H + 2 * KV;
    dim3 grid(static_cast<unsigned>(qkv.size(0)), static_cast<unsigned>((n_tasks + kQkvWarps - 1) / kQkvWarps));
    qk_norm_rope_kv_fp8_kernel<T, HD><<<grid, kQkvWarps * 32, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<T*>(q_out.data_ptr()), reinterpret_cast<const T*>(qkv.data_ptr()),
        reinterpret_cast<const T*>(qn.data_ptr()), reinterpret_cast<const T*>(kn.data_ptr()), cos_t.data_ptr<float>(),
        sin_t.data_ptr<float>(), pos.data_ptr<int64_t>(), slots.data_ptr<int64_t>(),
        reinterpret_cast<uint8_t*>(kc.data_ptr()), reinterpret_cast<uint8_t*>(vc.data_ptr()), ks.data_ptr<float>(),
        vs.data_ptr<float>(), kd.defined() ? reinterpret_cast<T*>(kd.data_ptr()) : nullptr,
        vd.defined() ? reinterpret_cast<T*>(vd.data_ptr()) : nullptr, H, KV, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- swiglu -------------------------------------------------------------------------------------------------------
template <typename T>
__global__ void swiglu_kernel(T* __restrict__ out, const T* __restrict__ gu, int64_t n, int I) {
    const int64_t k = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const int64_t r = k / I;
    const int i = static_cast<int>(k - r * I);
    const float g = to_float(gu[r * 2 * I + i]);
    const float u = to_float(gu[r * 2 * I + I + i]);
    const float silu = rnd<T>(g / (1.f + expf(-g)));   
    out[k] = from_float<T>(silu * u);
}

}  

// ---- host entry points --------------------------------------------------------------------------------------------

void rms_norm(at::Tensor& out, const at::Tensor& x, const at::Tensor& w, double eps) {
    if (x.size(0) == 0) return;
    const c10::cuda::CUDAGuard guard(x.device());
    switch (x.scalar_type()) {
        case at::kFloat:    launch_norm<float, false>(out, nullptr, x, w, eps); break;
        case at::kHalf:     launch_norm<__half, false>(out, nullptr, x, w, eps); break;
        case at::kBFloat16: launch_norm<__nv_bfloat16, false>(out, nullptr, x, w, eps); break;
    }
}

void add_rms_norm(at::Tensor& resid, const at::Tensor& x, const at::Tensor& w, at::Tensor& out, double eps) {
    if (resid.size(0) == 0) return;
    const c10::cuda::CUDAGuard guard(resid.device());
    switch (resid.scalar_type()) {
        case at::kFloat:    launch_norm<float, true>(out, &resid, x, w, eps); break;
        case at::kHalf:     launch_norm<__half, true>(out, &resid, x, w, eps); break;
        case at::kBFloat16: launch_norm<__nv_bfloat16, true>(out, &resid, x, w, eps); break;
    }
}

void qk_norm_rope_kv_write_fp8(at::Tensor& q_out, const at::Tensor& qkv, const at::Tensor& q_norm, const at::Tensor& k_norm,
                               const at::Tensor& cos, const at::Tensor& sin, const at::Tensor& positions,
                               const at::Tensor& slots, at::Tensor& k_cache, at::Tensor& v_cache, at::Tensor& k_scale,
                               at::Tensor& v_scale, at::Tensor& k_dense, at::Tensor& v_dense, int64_t n_heads,
                               int64_t n_kv_heads, double eps) {
    const int64_t T = q_out.size(0), hd = q_out.size(2);
    if (T == 0) return;
    const c10::cuda::CUDAGuard guard(qkv.device());
    const int H = static_cast<int>(n_heads), KV = static_cast<int>(n_kv_heads);
    const float e = static_cast<float>(eps);
#define FUSED_QKV8(TYPE, HD) launch_qkv_fp8<TYPE, HD>(q_out, qkv, q_norm, k_norm, cos, sin, positions, slots, k_cache, v_cache, k_scale, v_scale, k_dense, v_dense, H, KV, e)
    if (hd == 64) FUSED_QKV8(__nv_bfloat16, 64); else FUSED_QKV8(__nv_bfloat16, 128);
#undef FUSED_QKV8
}

void swiglu(at::Tensor& out, const at::Tensor& gate_up) {
    const int64_t n = out.numel();
    if (n == 0) return;
    const c10::cuda::CUDAGuard guard(gate_up.device());
    const int I = static_cast<int>(out.size(1));
    const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
    auto stream = c10::cuda::getCurrentCUDAStream();
    switch (gate_up.scalar_type()) {
        case at::kFloat:
            swiglu_kernel<float><<<blocks, 256, 0, stream>>>(out.data_ptr<float>(), gate_up.data_ptr<float>(), n, I);
            break;
        case at::kHalf:
            swiglu_kernel<__half><<<blocks, 256, 0, stream>>>(reinterpret_cast<__half*>(out.data_ptr()),
                                                              reinterpret_cast<const __half*>(gate_up.data_ptr()), n, I);
            break;
        case at::kBFloat16:
            swiglu_kernel<__nv_bfloat16><<<blocks, 256, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
                                                                     reinterpret_cast<const __nv_bfloat16*>(gate_up.data_ptr()), n, I);
            break;
        default: TORCH_CHECK(false, "fused::swiglu: dtype must be float32, float16 or bfloat16");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {

__device__ __forceinline__ void philox4x32(uint32_t c[4], uint32_t k0, uint32_t k1) {
#pragma unroll
    for (int round = 0; round < 10; ++round) {
        const uint64_t p0 = static_cast<uint64_t>(0xD2511F53u) * c[0];
        const uint64_t p1 = static_cast<uint64_t>(0xCD9E8D57u) * c[2];
        const uint32_t n0 = static_cast<uint32_t>(p1 >> 32) ^ c[1] ^ k0;
        const uint32_t n1 = static_cast<uint32_t>(p1);
        const uint32_t n2 = static_cast<uint32_t>(p0 >> 32) ^ c[3] ^ k1;
        const uint32_t n3 = static_cast<uint32_t>(p0);
        c[0] = n0; c[1] = n1; c[2] = n2; c[3] = n3;
        k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
    }
}

__device__ __forceinline__ float u01(uint32_t x) { return (static_cast<float>(x >> 8) + 0.5f) * (1.0f / 16777216.0f); }

struct Best { float v; int i; };
__device__ __forceinline__ bool better(float v, int i, const Best& b) { return v > b.v || (v == b.v && i < b.i); }

__device__ __forceinline__ float gumbel_score(float logit, float inv_t, uint32_t rnd) {
    const float e = fmaxf(-logf(u01(rnd)), 1e-10f);          
    return logit * inv_t - logf(e);
}

__global__ void sample_kernel(int64_t* __restrict__ out, const float* __restrict__ logits, const int64_t* __restrict__ rows,
                              const float* __restrict__ temps, int V, uint64_t seed, uint64_t offset) {
    const int r = blockIdx.x;
    const float* row = logits + rows[r] * static_cast<int64_t>(V);
    const float temp = temps[r];
    const bool greedy = temp <= 0.f;
    const float inv_t = 1.0f / fmaxf(temp, 1e-5f);
    const bool vec = (V & 3) == 0;                           

    Best best{-INFINITY, 0x7fffffff};
    for (int base = threadIdx.x * 4; base < V; base += blockDim.x * 4) {
        float x[4];
        if (vec) {
            const float4 f = *reinterpret_cast<const float4*>(row + base);
            x[0] = f.x; x[1] = f.y; x[2] = f.z; x[3] = f.w;
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) x[j] = base + j < V ? row[base + j] : -INFINITY;
        }
        uint32_t rnd[4] = {0, 0, 0, 0};
        if (!greedy) {
            rnd[0] = static_cast<uint32_t>(base >> 2); rnd[1] = static_cast<uint32_t>(r);       
            rnd[2] = static_cast<uint32_t>(offset);    rnd[3] = static_cast<uint32_t>(offset >> 32);   
            philox4x32(rnd, static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32));
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (base + j >= V) break;
            const float s = greedy ? x[j] : gumbel_score(x[j], inv_t, rnd[j]);
            if (better(s, base + j, best)) { best.v = s; best.i = base + j; }
        }
    }
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, best.v, d);
        const int oi = __shfl_down_sync(0xffffffffu, best.i, d);
        if (better(ov, oi, best)) { best.v = ov; best.i = oi; }
    }
    __shared__ float sv[32];
    __shared__ int si[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = (blockDim.x + 31) >> 5;
    if (lane == 0) { sv[warp] = best.v; si[warp] = best.i; }
    __syncthreads();
    if (warp == 0) {
        best.v = lane < nwarps ? sv[lane] : -INFINITY;
        best.i = lane < nwarps ? si[lane] : 0x7fffffff;
#pragma unroll
        for (int d = 16; d > 0; d >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, best.v, d);
            const int oi = __shfl_down_sync(0xffffffffu, best.i, d);
            if (better(ov, oi, best)) { best.v = ov; best.i = oi; }
        }
        if (lane == 0) out[r] = best.i;
    }
}

}  // namespace

void sample_tokens(at::Tensor& out, const at::Tensor& logits, const at::Tensor& rows, const at::Tensor& temps,
                   uint64_t seed, uint64_t offset) {
    const int64_t R = rows.size(0);
    if (R == 0) return;
    const c10::cuda::CUDAGuard guard(logits.device());
    sample_kernel<<<static_cast<unsigned>(R), 1024, 0, c10::cuda::getCurrentCUDAStream()>>>(
        out.data_ptr<int64_t>(), logits.data_ptr<float>(), rows.data_ptr<int64_t>(), temps.data_ptr<float>(),
        static_cast<int>(logits.size(1)), seed, offset);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
// ---- token feedback -----------------------------------------------------------------------------------------------
namespace {
__global__ void feed_tokens_kernel(int64_t* __restrict__ tokens, const int64_t* __restrict__ src,
                                   const int64_t* __restrict__ prev, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t s = src[i];
    if (s >= 0) tokens[i] = prev[s];
}
}  

void feed_tokens(at::Tensor& tokens, const at::Tensor& src, const at::Tensor& prev) {
    const int64_t n = tokens.numel();
    if (n == 0) return;
    const c10::cuda::CUDAGuard guard(tokens.device());
    feed_tokens_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        tokens.data_ptr<int64_t>(), src.data_ptr<int64_t>(), prev.data_ptr<int64_t>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  