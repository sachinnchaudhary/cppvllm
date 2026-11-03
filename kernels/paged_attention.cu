#include "kernels/paged_attention.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>


namespace {

constexpr int kWarps   = 4;
constexpr int kThreads = kWarps * 32;
constexpr float kNegInf = -1e30f;   

__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T> __device__ __forceinline__ T from_float(float x);
template <> __device__ __forceinline__ float from_float<float>(float x) { return x; }
template <> __device__ __forceinline__ __half from_float<__half>(float x) { return __float2half(x); }
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

// 16 bytes from global memory -> VEC floats in registers
template <typename T, int VEC>
__device__ __forceinline__ void load_vec(const T* p, float (&f)[VEC]) {
    const uint4 u = *reinterpret_cast<const uint4*>(p);
    const T* t = reinterpret_cast<const T*>(&u);
#pragma unroll
    for (int e = 0; e < VEC; ++e) f[e] = to_float(t[e]);
}

template <typename T, int HD, int REP>
__global__ void __launch_bounds__(kThreads)
paged_attention_kernel(T* __restrict__ out, const T* __restrict__ q, const T* __restrict__ k_cache,
                       const T* __restrict__ v_cache, const int64_t* __restrict__ block_tables,
                       const int64_t* __restrict__ token_seq, const int64_t* __restrict__ token_kv_len,
                       int64_t table_width, int block_size, int n_kv_heads, float scale) {
    constexpr int VEC = 16 / sizeof(T);   // numbers per lane: 8 for 16-bit types, 4 for float
    constexpr int TPK = HD / VEC;         // lanes per key
    constexpr int KPI = 32 / TPK;         // keys a warp handles at once
    
    const int t    = blockIdx.x;                 // query token
    const int g    = blockIdx.y;                 // kv head
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int grp  = lane / TPK;                 // which of the KPI keys this lane works on
    const int sub  = lane % TPK;                 // which VEC-sized slice of the head_dim this lane owns
    const int n_heads = n_kv_heads * REP;

    const int64_t kv_len = token_kv_len[t];
    const int64_t* table = block_tables + token_seq[t] * table_width;

    // this lane's slice of the REP query vectors, pre-multiplied by the softmax scale
    float qf[REP][VEC];
#pragma unroll
    for (int r = 0; r < REP; ++r) {
        load_vec<T, VEC>(q + ((int64_t)t * n_heads + g * REP + r) * HD + sub * VEC, qf[r]);
#pragma unroll
        for (int e = 0; e < VEC; ++e) qf[r][e] *= scale;
    }

    float m[REP], l[REP], acc[REP][VEC];
#pragma unroll
    for (int r = 0; r < REP; ++r) {
        m[r] = kNegInf;
        l[r] = 0.f;
#pragma unroll
        for (int e = 0; e < VEC; ++e) acc[r][e] = 0.f;
    }

    // ---- walk the sequence's KV blocks (warp w takes blocks w, w+4, w+8, ...) ----------------------------
    const int num_kv_blocks = static_cast<int>((kv_len + block_size - 1) / block_size);
    for (int b = warp; b < num_kv_blocks; b += kWarps) {
        const int64_t slot0 = table[b] * (int64_t)block_size;      // cache slot of this block's first key
        for (int i = 0; i < block_size; i += KPI) {
            const int  key   = i + grp;                                           // key index inside the block
            const bool valid = key < block_size && (int64_t)b * block_size + key < kv_len;   // last block may be partial
            const int64_t off = ((slot0 + key) * n_kv_heads + g) * HD + sub * VEC;

            float kf[VEC];
            if (valid) load_vec<T, VEC>(k_cache + off, kf);
            else {
#pragma unroll
                for (int e = 0; e < VEC; ++e) kf[e] = 0.f;
            }

            // scores of this key for the REP query heads. Every lane must run the shuffles (even for an
            // invalid key), so validity is only checked AFTER them.
            float s[REP];
#pragma unroll
            for (int r = 0; r < REP; ++r) {
                float p = 0.f;
#pragma unroll
                for (int e = 0; e < VEC; ++e) p += qf[r][e] * kf[e];
#pragma unroll
                for (int o = TPK / 2; o > 0; o >>= 1) p += __shfl_xor_sync(0xffffffffu, p, o);   // sum over the group's lanes
                s[r] = p;
            }

            if (valid) {
                float vf[VEC];
                load_vec<T, VEC>(v_cache + off, vf);
#pragma unroll
                for (int r = 0; r < REP; ++r) {
                    const float m_new = fmaxf(m[r], s[r]);
                    const float corr  = expf(m[r] - m_new);       // shrink what we had, it was relative to the old max
                    const float p     = expf(s[r] - m_new);
                    l[r] = l[r] * corr + p;
#pragma unroll
                    for (int e = 0; e < VEC; ++e) acc[r][e] = acc[r][e] * corr + p * vf[e];
                    m[r] = m_new;
                }
            }
        }
    }

    // ---- merge the KPI groups of this warp (lanes that differ only in their group index) ------------------------
    for (int o = TPK; o < 32; o <<= 1) {
#pragma unroll
        for (int r = 0; r < REP; ++r) {
            const float m2 = __shfl_xor_sync(0xffffffffu, m[r], o);
            const float l2 = __shfl_xor_sync(0xffffffffu, l[r], o);
            const float M  = fmaxf(m[r], m2);
            const float c1 = expf(m[r] - M), c2 = expf(m2 - M);
            l[r] = l[r] * c1 + l2 * c2;
#pragma unroll
            for (int e = 0; e < VEC; ++e) acc[r][e] = acc[r][e] * c1 + __shfl_xor_sync(0xffffffffu, acc[r][e], o) * c2;
            m[r] = M;
        }
    }

    // ---- merge the warps through shared memory ---------------------------------------------------------------------
    __shared__ float s_m[kWarps][REP], s_l[kWarps][REP], s_acc[kWarps][REP][HD];
    if (lane < TPK) {                              // group 0 of each warp holds the warp's merged result
#pragma unroll
        for (int r = 0; r < REP; ++r) {
#pragma unroll
            for (int e = 0; e < VEC; ++e) s_acc[warp][r][sub * VEC + e] = acc[r][e];
            if (lane == 0) { s_m[warp][r] = m[r]; s_l[warp][r] = l[r]; }
        }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < REP * HD; idx += kThreads) {     // one output number per thread
        const int r = idx / HD, d = idx % HD;
        float M = kNegInf;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) M = fmaxf(M, s_m[w][r]);
        float L = 0.f, A = 0.f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            const float c = expf(s_m[w][r] - M);        // a warp that saw no keys has m = -1e30, l = acc = 0: contributes 0
            L += s_l[w][r] * c;
            A += s_acc[w][r][d] * c;
        }
        out[((int64_t)t * n_heads + g * REP + r) * HD + d] = from_float<T>(A / L);   // L >= 1 term: every token sees >= 1 key
    }
}

// ---- host side: pick template parameters from the runtime shapes --------------------------------------------------
template <typename T, int HD, int REP>
void launch(at::Tensor& out, const at::Tensor& q, const at::Tensor& kc, const at::Tensor& vc, const at::Tensor& bt,
            const at::Tensor& tseq, const at::Tensor& tlen, float scale) {
    const int64_t T_ = q.size(0);
    const int n_kv   = static_cast<int>(kc.size(2));
    dim3 grid(static_cast<unsigned>(T_), static_cast<unsigned>(n_kv));
    paged_attention_kernel<T, HD, REP><<<grid, kThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<T*>(out.data_ptr()), reinterpret_cast<const T*>(q.data_ptr()),
        reinterpret_cast<const T*>(kc.data_ptr()), reinterpret_cast<const T*>(vc.data_ptr()),
        bt.data_ptr<int64_t>(), tseq.data_ptr<int64_t>(), tlen.data_ptr<int64_t>(), bt.size(1),
        static_cast<int>(kc.size(1)), n_kv, scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T, int HD>
void launch_rep(int rep, at::Tensor& out, const at::Tensor& q, const at::Tensor& kc, const at::Tensor& vc,
                const at::Tensor& bt, const at::Tensor& tseq, const at::Tensor& tlen, float scale) {
    switch (rep) {
        case 1: launch<T, HD, 1>(out, q, kc, vc, bt, tseq, tlen, scale); break;
        case 2: launch<T, HD, 2>(out, q, kc, vc, bt, tseq, tlen, scale); break;
        case 4: launch<T, HD, 4>(out, q, kc, vc, bt, tseq, tlen, scale); break;
        case 8: launch<T, HD, 8>(out, q, kc, vc, bt, tseq, tlen, scale); break;
        default: TORCH_CHECK(false, "paged_attention: n_heads / n_kv_heads must be 1, 2, 4 or 8 (got ", rep, ")");
    }
}

template <typename T>
void launch_hd(int hd, int rep, at::Tensor& out, const at::Tensor& q, const at::Tensor& kc, const at::Tensor& vc,
               const at::Tensor& bt, const at::Tensor& tseq, const at::Tensor& tlen, float scale) {
    switch (hd) {
        case 64:  launch_rep<T, 64>(rep, out, q, kc, vc, bt, tseq, tlen, scale); break;
        case 128: launch_rep<T, 128>(rep, out, q, kc, vc, bt, tseq, tlen, scale); break;
        default: TORCH_CHECK(false, "paged_attention: head_dim must be 64 or 128 (got ", hd, ")");
    }
}

}  // namespace

void paged_attention(at::Tensor& out, const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
                     const at::Tensor& block_tables, const at::Tensor& token_seq, const at::Tensor& token_kv_len,
                     double scale) {

    const int64_t n_heads = q.size(1), hd = q.size(2), n_kv = k_cache.size(2);
    if (q.size(0) == 0) return;

    const c10::cuda::CUDAGuard guard(q.device());
    const int rep = static_cast<int>(n_heads / n_kv);
    const float s = static_cast<float>(scale);
    switch (q.scalar_type()) {
        case at::kFloat:    launch_hd<float>(hd, rep, out, q, k_cache, v_cache, block_tables, token_seq, token_kv_len, s); break;
        case at::kHalf:     launch_hd<__half>(hd, rep, out, q, k_cache, v_cache, block_tables, token_seq, token_kv_len, s); break;
        case at::kBFloat16: launch_hd<__nv_bfloat16>(hd, rep, out, q, k_cache, v_cache, block_tables, token_seq, token_kv_len, s); break;
        default: TORCH_CHECK(false, "paged_attention: dtype must be float32, float16 or bfloat16");
    }
}
