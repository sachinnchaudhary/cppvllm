#include "kernels/paged_attention.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>

namespace {

constexpr int kWarps   = 4;
constexpr int kThreads = kWarps * 32;
constexpr float kNegInf = -1e30f;
constexpr int kBlockSize = 16;   // KV block size (16 keys per block)
constexpr int TPK = 8;           // 8 lanes per key -> 3 shuffle levels instead of 4
constexpr int KPI = 32 / TPK;    // 4 keys per warp step (4 groups of 8 lanes)

__device__ __forceinline__ void unpack_fp8x4(uint32_t u, float* __restrict__ f) {
    const uint32_t h_even = ((u & 0x007F007Fu) << 7) | ((u & 0x00800080u) << 8);
    const uint32_t h_odd  = ((u & 0x7F007F00u) >> 1) | (u & 0x80008000u);
    const float2 e = __half22float2(*reinterpret_cast<const __half2*>(&h_even));
    const float2 o = __half22float2(*reinterpret_cast<const __half2*>(&h_odd));
    f[0] = e.x;
    f[1] = o.x;
    f[2] = e.y;
    f[3] = o.y;
}

template <int VEC> struct Fp8Vec;
template <> struct Fp8Vec<16> {
    using Type = uint4;
    static __device__ __forceinline__ Type zero() { return make_uint4(0, 0, 0, 0); }
    static __device__ __forceinline__ void unpack(Type u, float (&f)[16]) {
        unpack_fp8x4(u.x, f + 0);
        unpack_fp8x4(u.y, f + 4);
        unpack_fp8x4(u.z, f + 8);
        unpack_fp8x4(u.w, f + 12);
    }
};
template <> struct Fp8Vec<8> {
    using Type = uint2;
    static __device__ __forceinline__ Type zero() { return make_uint2(0, 0); }
    static __device__ __forceinline__ void unpack(Type u, float (&f)[8]) {
        unpack_fp8x4(u.x, f + 0);
        unpack_fp8x4(u.y, f + 4);
    }
};

template <int VEC>
__device__ __forceinline__ void load_q(const __nv_bfloat16* p, float (&f)[VEC]) {
    constexpr int kU4 = (VEC * sizeof(__nv_bfloat16)) / sizeof(uint4);
    const uint4* p4 = reinterpret_cast<const uint4*>(p);
#pragma unroll
    for (int u_idx = 0; u_idx < kU4; ++u_idx) {
        const uint4 u = p4[u_idx];
        const __nv_bfloat16* t = reinterpret_cast<const __nv_bfloat16*>(&u);
#pragma unroll
        for (int e = 0; e < 8; ++e) f[u_idx * 8 + e] = __bfloat162float(t[e]);
    }
}

__device__ __forceinline__ void reduce_4_scores_tpk8(float (&s)[4], int lane) {
    // Step 1: fold pairs across (lane ^ 1) -> 2 shuffles
    const bool odd1 = (lane & 1) != 0;
    const float a = odd1 ? s[0] : s[2];
    const float b = odd1 ? s[1] : s[3];
    const float a_shfl = __shfl_xor_sync(0xffffffffu, a, 1);
    const float b_shfl = __shfl_xor_sync(0xffffffffu, b, 1);
    const float u0 = (odd1 ? s[2] : s[0]) + a_shfl;
    const float u1 = (odd1 ? s[3] : s[1]) + b_shfl;

    const bool odd2 = (lane & 2) != 0;
    const float c = odd2 ? u0 : u1;
    const float c_shfl = __shfl_xor_sync(0xffffffffu, c, 2);
    float v = (odd2 ? u1 : u0) + c_shfl;

    v += __shfl_xor_sync(0xffffffffu, v, 4);

    const int quad_base = lane & ~3;
    s[0] = __shfl_sync(0xffffffffu, v, quad_base + 0);
    s[2] = __shfl_sync(0xffffffffu, v, quad_base + 1);
    s[1] = __shfl_sync(0xffffffffu, v, quad_base + 2);
    s[3] = __shfl_sync(0xffffffffu, v, quad_base + 3);
}

template <int HD, int REP>
__global__ void __launch_bounds__(kThreads) paged_attention_fp8_kernel(
        __nv_bfloat16* __restrict__ out,
        float* __restrict__ partial_acc,          // [T_, n_heads, num_splits, HD] when num_splits > 1
        float2* __restrict__ partial_ml,          // [T_, n_heads, num_splits]     when num_splits > 1
        const __nv_bfloat16* __restrict__ q,
        const uint8_t* __restrict__ k_cache,
        const uint8_t* __restrict__ v_cache,
        const float* __restrict__ k_scale,
        const float* __restrict__ v_scale,
        const int64_t* __restrict__ block_tables,
        const int64_t* __restrict__ token_seq,
        const int64_t* __restrict__ token_kv_len,
        int64_t table_width, int n_kv_heads, float scale) {

    constexpr int VEC = HD / TPK;   // 16 for HD=128, 8 for HD=64
    using VecOp = Fp8Vec<VEC>;
    using VecT  = typename VecOp::Type;

    const int t          = blockIdx.x;
    const int g          = blockIdx.y;
    const int split_idx  = blockIdx.z;
    const int num_splits = gridDim.z;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int grp  = lane / TPK;    // 0..3: which of the 4 keys in a step this lane handles
    const int sub  = lane % TPK;    // 0..7: which VEC-element slice of HD this lane owns
    const int n_heads = n_kv_heads * REP;

    const int64_t kv_len = token_kv_len[t];
    const int num_kv_blocks = static_cast<int>((kv_len + kBlockSize - 1) / kBlockSize);

    // Partition KV blocks across splits (minimum 4 blocks = 64 tokens per split so short seqs stay on split 0)
    const int blocks_per_split = max(4, (num_kv_blocks + num_splits - 1) / num_splits);
    const int b_start = split_idx * blocks_per_split;
    const int b_end   = min(num_kv_blocks, b_start + blocks_per_split);

    if (b_start >= b_end) {
        if (num_splits > 1 && threadIdx.x < REP) {
            const int r = threadIdx.x;
            const int64_t head_linear = ((int64_t)t * n_heads + g * REP + r) * num_splits + split_idx;
            partial_ml[head_linear] = make_float2(kNegInf, 0.f);
        }
        return;
    }

    const int64_t* table = block_tables + token_seq[t] * table_width;

    float qf[REP][VEC];
    const float q_scale = scale * 256.f;
#pragma unroll
    for (int r = 0; r < REP; ++r) {
        load_q<VEC>(q + ((int64_t)t * n_heads + g * REP + r) * HD + sub * VEC, qf[r]);
#pragma unroll
        for (int e = 0; e < VEC; ++e) qf[r][e] *= q_scale;
    }

    float m[REP], l[REP], acc[REP][VEC];
#pragma unroll
    for (int r = 0; r < REP; ++r) {
        m[r] = kNegInf;
        l[r] = 0.f;
#pragma unroll
        for (int e = 0; e < VEC; ++e) acc[r][e] = 0.f;
    }

    auto process_4_keys = [&](const VecT (&k_raw)[4], const VecT (&v_raw)[4],
                              const float (&ks)[4], const float (&vs)[4], const bool (&valid)[4]) {
        float s[REP][4];
#pragma unroll
        for (int it = 0; it < 4; ++it) {
            float kf[VEC];
            VecOp::unpack(k_raw[it], kf);
#pragma unroll
            for (int r = 0; r < REP; ++r) {
                float p = 0.f;
#pragma unroll
                for (int e = 0; e < VEC; ++e) p += qf[r][e] * kf[e];
                s[r][it] = p;
            }
        }

#pragma unroll
        for (int r = 0; r < REP; ++r) {
            reduce_4_scores_tpk8(s[r], lane);
        }

        float vf[4][VEC];
#pragma unroll
        for (int it = 0; it < 4; ++it) {
            VecOp::unpack(v_raw[it], vf[it]);
        }
#pragma unroll
        for (int r = 0; r < REP; ++r) {
            const float sr0 = valid[0] ? (s[r][0] * ks[0]) : kNegInf;
            const float sr1 = valid[1] ? (s[r][1] * ks[1]) : kNegInf;
            const float sr2 = valid[2] ? (s[r][2] * ks[2]) : kNegInf;
            const float sr3 = valid[3] ? (s[r][3] * ks[3]) : kNegInf;

            const float m_blk = fmaxf(fmaxf(sr0, sr1), fmaxf(sr2, sr3));
            if (m_blk > kNegInf * 0.5f) {
                const float m_new = fmaxf(m[r], m_blk);
                const float corr  = expf(m[r] - m_new);
                const float p0    = valid[0] ? expf(sr0 - m_new) : 0.f;
                const float p1    = valid[1] ? expf(sr1 - m_new) : 0.f;
                const float p2    = valid[2] ? expf(sr2 - m_new) : 0.f;
                const float p3    = valid[3] ? expf(sr3 - m_new) : 0.f;

                l[r] = l[r] * corr + (p0 + p1 + p2 + p3);
                m[r] = m_new;

                const float pv0 = p0 * vs[0];
                const float pv1 = p1 * vs[1];
                const float pv2 = p2 * vs[2];
                const float pv3 = p3 * vs[3];

#pragma unroll
                for (int e = 0; e < VEC; ++e) {
                    acc[r][e] = acc[r][e] * corr
                              + pv0 * vf[0][e]
                              + pv1 * vf[1][e]
                              + pv2 * vf[2][e]
                              + pv3 * vf[3][e];
                }
            }
        }
    };

    const int n_split_blocks = b_end - b_start;
    const int full_waves_end = b_start + (n_split_blocks / kWarps) * kWarps;

    for (int b = b_start + warp; b < full_waves_end; b += kWarps) {
        const int64_t phy_blk    = table[b];
        const int64_t slot0      = phy_blk * kBlockSize;
        const int64_t scale_base = (phy_blk * n_kv_heads + g) * kBlockSize;

        // 1 coalesced float4 load per group (reads all 16 consecutive scales across the 4 groups of the warp)
        const float4 ks4 = reinterpret_cast<const float4*>(k_scale + scale_base)[grp];
        const float4 vs4 = reinterpret_cast<const float4*>(v_scale + scale_base)[grp];
        const float ks[4] = {ks4.x, ks4.y, ks4.z, ks4.w};
        const float vs[4] = {vs4.x, vs4.y, vs4.z, vs4.w};

        VecT k_raw[4], v_raw[4];
        bool valid[4];

        // Fast path: all 16 keys in this block are valid -> issue all 8 16-byte global loads unconditionally!
        if (static_cast<int64_t>(b + 1) * kBlockSize <= kv_len) {
#pragma unroll
            for (int it = 0; it < 4; ++it) {
                const int key = grp * 4 + it;
                const int64_t off = ((slot0 + key) * n_kv_heads + g) * HD + sub * VEC;
                k_raw[it] = *reinterpret_cast<const VecT*>(k_cache + off);
                v_raw[it] = *reinterpret_cast<const VecT*>(v_cache + off);
                valid[it] = true;
            }
        } else {
#pragma unroll
            for (int it = 0; it < 4; ++it) {
                const int key = grp * 4 + it;
                valid[it] = (static_cast<int64_t>(b) * kBlockSize + key) < kv_len;
                const int64_t off = ((slot0 + key) * n_kv_heads + g) * HD + sub * VEC;
                if (valid[it]) {
                    k_raw[it] = *reinterpret_cast<const VecT*>(k_cache + off);
                    v_raw[it] = *reinterpret_cast<const VecT*>(v_cache + off);
                } else {
                    k_raw[it] = VecOp::zero();
                    v_raw[it] = VecOp::zero();
                }
            }
        }

        process_4_keys(k_raw, v_raw, ks, vs, valid);
    }

    if (full_waves_end < b_end) {
        const int64_t kv_rem_start = static_cast<int64_t>(full_waves_end) * kBlockSize;
        const int64_t kv_rem_end   = min(static_cast<int64_t>(b_end) * kBlockSize, kv_len);
        const int rem_keys = static_cast<int>(kv_rem_end - kv_rem_start);
        const int rem_qtrs = (rem_keys + KPI - 1) / KPI;   // 1..12 quarters of 4 keys -> at most 3 per warp!

        if (warp < rem_qtrs) {
            VecT k_raw[4], v_raw[4];
            float ks[4] = {1.f, 1.f, 1.f, 1.f};
            float vs[4] = {1.f, 1.f, 1.f, 1.f};
            bool valid[4] = {false, false, false, false};

#pragma unroll
            for (int it = 0; it < 3; ++it) {
                const int qtr = warp + it * kWarps;
                const int64_t key_global = kv_rem_start + qtr * KPI + grp;
                if (qtr < rem_qtrs && key_global < kv_rem_end) {
                    valid[it] = true;
                    const int b          = static_cast<int>(key_global >> 4);
                    const int key_in_blk = static_cast<int>(key_global & 15);
                    const int64_t phy_blk = table[b];
                    const int64_t sidx    = (phy_blk * n_kv_heads + g) * kBlockSize + key_in_blk;
                    const int64_t off     = ((phy_blk * kBlockSize + key_in_blk) * n_kv_heads + g) * HD + sub * VEC;
                    ks[it]    = k_scale[sidx];
                    vs[it]    = v_scale[sidx];
                    k_raw[it] = *reinterpret_cast<const VecT*>(k_cache + off);
                    v_raw[it] = *reinterpret_cast<const VecT*>(v_cache + off);
                } else {
                    k_raw[it] = VecOp::zero();
                    v_raw[it] = VecOp::zero();
                }
            }
            k_raw[3] = VecOp::zero();
            v_raw[3] = VecOp::zero();

            process_4_keys(k_raw, v_raw, ks, vs, valid);
        }
    }

#pragma unroll
    for (int o = TPK; o < 32; o <<= 1) {
#pragma unroll
        for (int r = 0; r < REP; ++r) {
            const float m2 = __shfl_xor_sync(0xffffffffu, m[r], o);
            const float l2 = __shfl_xor_sync(0xffffffffu, l[r], o);
            const float M  = fmaxf(m[r], m2);
            const float c1 = expf(m[r] - M);
            const float c2 = expf(m2   - M);
            l[r] = l[r] * c1 + l2 * c2;
#pragma unroll
            for (int e = 0; e < VEC; ++e) {
                const float a2 = __shfl_xor_sync(0xffffffffu, acc[r][e], o);
                acc[r][e] = acc[r][e] * c1 + a2 * c2;
            }
            m[r] = M;
        }
    }

    __shared__ float s_m[kWarps][REP], s_l[kWarps][REP], s_acc[kWarps][REP][HD];

    if (lane < TPK) {
#pragma unroll
        for (int r = 0; r < REP; ++r) {
#pragma unroll
            for (int e = 0; e < VEC; ++e) s_acc[warp][r][sub * VEC + e] = acc[r][e];
            if (lane == 0) { s_m[warp][r] = m[r]; s_l[warp][r] = l[r]; }
        }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < REP * HD; idx += kThreads) {
        const int r = idx / HD, d = idx % HD;
        float M = kNegInf;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) M = fmaxf(M, s_m[w][r]);
        float L = 0.f, A = 0.f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            const float c = expf(s_m[w][r] - M);
            L += s_l[w][r] * c;
            A += s_acc[w][r][d] * c;
        }

        // Multiply A by 256.f once at the end to compensate for the 2^-8 FP8 E4M3 V bias
        const int64_t head_idx = (int64_t)t * n_heads + g * REP + r;
        if (num_splits == 1) {
            out[head_idx * HD + d] = __float2bfloat16((A * 256.f) / L);
        } else {
            const int64_t split_linear = head_idx * num_splits + split_idx;
            partial_acc[split_linear * HD + d] = A * 256.f;
            if (d == 0) {
                partial_ml[split_linear] = make_float2(M, L);
            }
        }
    }
}

template <int HD>
__global__ void __launch_bounds__(HD) paged_attention_merge_kernel(
        __nv_bfloat16* __restrict__ out,
        const float* __restrict__ partial_acc,
        const float2* __restrict__ partial_ml,
        int n_heads, int num_splits) {

    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;

    const int64_t head_idx  = (int64_t)t * n_heads + h;
    const int64_t base_s    = head_idx * num_splits;

    float M = kNegInf;
    float L = 0.f;
    float A = 0.f;

    for (int s = 0; s < num_splits; ++s) {
        const float2 ml = partial_ml[base_s + s];
        if (ml.x > kNegInf * 0.5f && ml.y > 0.f) {
            const float acc_s = partial_acc[(base_s + s) * HD + d];
            const float m_new = fmaxf(M, ml.x);
            const float c_old = expf(M - m_new);
            const float c_new = expf(ml.x - m_new);
            L = L * c_old + ml.y * c_new;
            A = A * c_old + acc_s * c_new;
            M = m_new;
        }
    }

    out[head_idx * HD + d] = __float2bfloat16(A / L);
}

template <int HD, int REP>
void launch(at::Tensor& out, const at::Tensor& q, const at::Tensor& kc, const at::Tensor& vc, const at::Tensor& ks,
            const at::Tensor& vs, const at::Tensor& bt, const at::Tensor& tseq, const at::Tensor& tlen, float scale) {
    const int64_t T_    = q.size(0);
    const int n_heads   = static_cast<int>(q.size(1));
    const int n_kv      = static_cast<int>(kc.size(2));
    auto stream         = c10::cuda::getCurrentCUDAStream();

    // Pick split-KV factor based on total (token, kv_head) pairs to saturate SMs on tail/uneven batches
    const int64_t total_kv_tasks = T_ * n_kv;
    int num_splits = 1;
    if      (total_kv_tasks <= 32)  num_splits = 16;
    else if (total_kv_tasks <= 64)  num_splits = 8;
    else if (total_kv_tasks <= 256) num_splits = 4;
    else                            num_splits = 2;

    float* acc_ptr  = nullptr;
    float2* ml_ptr  = nullptr;
    at::Tensor tmp_acc, tmp_ml;

    if (num_splits > 1) {
        const auto f32_opts = q.options().dtype(at::kFloat);
        tmp_acc = at::empty({T_, n_heads, num_splits, HD}, f32_opts);
        tmp_ml  = at::empty({T_, n_heads, num_splits, 2}, f32_opts);
        acc_ptr = tmp_acc.data_ptr<float>();
        ml_ptr  = reinterpret_cast<float2*>(tmp_ml.data_ptr<float>());
    }

    dim3 grid(static_cast<unsigned>(T_), static_cast<unsigned>(n_kv), static_cast<unsigned>(num_splits));
    paged_attention_fp8_kernel<HD, REP><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        acc_ptr, ml_ptr,
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const uint8_t*>(kc.data_ptr()),
        reinterpret_cast<const uint8_t*>(vc.data_ptr()),
        ks.data_ptr<float>(), vs.data_ptr<float>(),
        bt.data_ptr<int64_t>(), tseq.data_ptr<int64_t>(), tlen.data_ptr<int64_t>(),
        bt.size(1), n_kv, scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (num_splits > 1) {
        dim3 merge_grid(static_cast<unsigned>(T_), static_cast<unsigned>(n_heads));
        paged_attention_merge_kernel<HD><<<merge_grid, HD, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            acc_ptr, ml_ptr, n_heads, num_splits);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

template <int HD>
void launch_rep(int rep, at::Tensor& out, const at::Tensor& q, const at::Tensor& kc, const at::Tensor& vc,
                const at::Tensor& ks, const at::Tensor& vs, const at::Tensor& bt, const at::Tensor& tseq,
                const at::Tensor& tlen, float scale) {
    switch (rep) {
        case 1: launch<HD, 1>(out, q, kc, vc, ks, vs, bt, tseq, tlen, scale); break;
        case 2: launch<HD, 2>(out, q, kc, vc, ks, vs, bt, tseq, tlen, scale); break;
        case 4: launch<HD, 4>(out, q, kc, vc, ks, vs, bt, tseq, tlen, scale); break;
        case 8: launch<HD, 8>(out, q, kc, vc, ks, vs, bt, tseq, tlen, scale); break;
        default: TORCH_CHECK(false, "paged_attention_fp8: unsupported GQA rep ", rep);
    }
}

}  // namespace

void paged_attention_fp8(at::Tensor& out, const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
                         const at::Tensor& k_scale, const at::Tensor& v_scale, const at::Tensor& block_tables,
                         const at::Tensor& token_seq, const at::Tensor& token_kv_len, double scale) {
    if (q.size(0) == 0) return;
    const int64_t n_heads = q.size(1), hd = q.size(2), n_kv = k_cache.size(2);
    TORCH_CHECK(k_cache.size(1) == kBlockSize, "paged_attention_fp8: block_size must be 16");
    const c10::cuda::CUDAGuard guard(q.device());
    const int rep = static_cast<int>(n_heads / n_kv);
    const float s = static_cast<float>(scale);
    switch (hd) {
        case 64:  launch_rep<64>(rep, out, q, k_cache, v_cache, k_scale, v_scale, block_tables, token_seq, token_kv_len, s); break;
        case 128: launch_rep<128>(rep, out, q, k_cache, v_cache, k_scale, v_scale, block_tables, token_seq, token_kv_len, s); break;
        default:  TORCH_CHECK(false, "paged_attention_fp8: head_dim must be 64 or 128");
    }
}