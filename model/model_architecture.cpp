#include "model_architecture.h"
#include <cmath>
#include <optional>
#include "kernels/fused_ops.h"
#include "kernels/paged_attention.h"
#include "namespace_config.h"

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

// vllm-flash-attn (FlashAttention-2)

namespace FLASH_NAMESPACE {
std::vector<at::Tensor> mha_varlen_fwd(
    at::Tensor& q, const at::Tensor& k, const at::Tensor& v, std::optional<at::Tensor>& out_,
    const at::Tensor& cu_seqlens_q, const at::Tensor& cu_seqlens_k, std::optional<at::Tensor>& seqused_k,
    std::optional<const at::Tensor>& leftpad_k_, std::optional<at::Tensor>& block_table_,
    std::optional<at::Tensor>& alibi_slopes_, int max_seqlen_q, const int max_seqlen_k, const float p_dropout,
    const float softmax_scale, const bool zero_tensors, bool is_causal, int window_size_left, int window_size_right,
    const float softcap, const bool return_softmax, std::optional<at::Generator> gen_);
}

namespace {

torch::Tensor rms_norm(const torch::Tensor& x, const torch::Tensor& w, double eps) {
    torch::Tensor xf = x.to(torch::kFloat);
    xf = xf * torch::rsqrt(xf.pow(2).mean(-1, /*keepdim=*/true) + eps);
    return xf.to(x.scalar_type()) * w;
}

torch::Tensor shard_rows(const torch::Tensor& weight, int rank, int world_size) {
    if (world_size == 1) return weight;
    const int64_t rows_per_rank = weight.size(0) / world_size;
    return weight.narrow(0, rank * rows_per_rank, rows_per_rank).contiguous();
}

torch::Tensor shard_cols(const torch::Tensor& weight, int rank, int world_size) {
    if (world_size == 1) return weight;
    const int64_t cols_per_rank = weight.size(1) / world_size;
    return weight.narrow(1, rank * cols_per_rank, cols_per_rank).contiguous();
}

torch::Tensor flash_attention_dense(torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
                                    const torch::Tensor& cu_q, int64_t max_q, float scale) {
    std::optional<at::Tensor> out, seqused_k, bt, alibi;
    std::optional<const at::Tensor> leftpad_k;
    auto res = FLASH_NAMESPACE::mha_varlen_fwd(
        q, k, v, out, cu_q, cu_q, seqused_k, leftpad_k, bt, alibi,
        static_cast<int>(max_q), static_cast<int>(max_q),
        /*p_dropout=*/0.f, scale, /*zero_tensors=*/false, /*is_causal=*/true,
        /*window_size_left=*/-1, /*window_size_right=*/-1, /*softcap=*/0.f, /*return_softmax=*/false,
        /*gen_=*/std::nullopt);
    return res[0];
}

} 

Qwen3::Qwen3(const ModelConfig& cfg, int tp_rank, int tp_size, ncclComm_t comm)
    : global_cfg_(cfg), cfg_(cfg), tp_rank_(tp_rank), tp_size_(tp_size), head_dim_(cfg.head_dim()), comm_(comm) {
    cfg_.n_heads /= tp_size_;
    cfg_.n_kv_heads /= tp_size_;
    cfg_.intermediate_size /= tp_size_;

    build_rope_tables();
}

void Qwen3::build_rope_tables() {
    const int64_t hd = head_dim_;
    const auto f64 = torch::TensorOptions().dtype(torch::kDouble);
    torch::Tensor idx = torch::arange(0, hd, 2, f64);
    torch::Tensor freqs = torch::pow(static_cast<double>(cfg_.rope_theta), idx / static_cast<double>(hd)).reciprocal();
    torch::Tensor angle = torch::outer(torch::arange(cfg_.max_seq_len, f64), freqs);  // [max_seq_len, hd/2]
    cos_ = angle.cos().to(torch::kFloat);
    sin_ = angle.sin().to(torch::kFloat);
}

void Qwen3::load_weights(const WeightMap& w, torch::Device target_device) {
    const torch::Tensor& full_embed = w.at("model.embed_tokens.weight");
    embed_ = full_embed.to(target_device);
    final_norm_ = w.at("model.norm.weight").to(target_device);
    if (cfg_.tie_word_embeddings) {
        const int64_t vl = global_cfg_.vocab_size / tp_size_;
        lm_head_ = embed_.narrow(0, static_cast<int64_t>(tp_rank_) * vl, vl);
    } else {
        lm_head_ = shard_rows(w.at("lm_head.weight"), tp_rank_, tp_size_).to(target_device);
    }

    layers_.clear();
    layers_.reserve(global_cfg_.n_layers);
    for (int32_t i = 0; i < global_cfg_.n_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        const torch::Tensor& q_weight = w.at(p + "self_attn.q_proj.weight");
        const torch::Tensor& k_weight = w.at(p + "self_attn.k_proj.weight");
        const torch::Tensor& v_weight = w.at(p + "self_attn.v_proj.weight");
        Layer L;
        
        L.wqkv = torch::cat({shard_rows(q_weight, tp_rank_, tp_size_),
                             shard_rows(k_weight, tp_rank_, tp_size_),
                             shard_rows(v_weight, tp_rank_, tp_size_)}, 0).to(target_device);
        const torch::Tensor& wo_weight = w.at(p + "self_attn.o_proj.weight");
        L.wo = shard_cols(wo_weight, tp_rank_, tp_size_).to(target_device);

        L.q_norm = w.at(p + "self_attn.q_norm.weight").to(target_device);
        L.k_norm = w.at(p + "self_attn.k_norm.weight").to(target_device);

        const torch::Tensor& gate_weight = w.at(p + "mlp.gate_proj.weight");
        const torch::Tensor& up_weight = w.at(p + "mlp.up_proj.weight");
        L.w_gate_up = torch::cat({shard_rows(gate_weight, tp_rank_, tp_size_),
                                  shard_rows(up_weight, tp_rank_, tp_size_)}, 0).to(target_device);

        const torch::Tensor& down_weight = w.at(p + "mlp.down_proj.weight");
        L.down = shard_cols(down_weight, tp_rank_, tp_size_).to(target_device);
        L.attn_norm = w.at(p + "input_layernorm.weight").to(target_device);
        L.ffn_norm = w.at(p + "post_attention_layernorm.weight").to(target_device);
        layers_.push_back(std::move(L));
    }

    cos_ = cos_.to(target_device);
    sin_ = sin_.to(target_device);
}

void Qwen3::bind_kv_cache(torch::Tensor k, torch::Tensor v, torch::Tensor k_scale, torch::Tensor v_scale) {
    k_cache_ = std::move(k);
    v_cache_ = std::move(v);
    k_scale_ = std::move(k_scale);
    v_scale_ = std::move(v_scale);
    block_size_ = k_cache_.size(2);
}

torch::Tensor Qwen3::forward_backbone(const FlatBatch& b) {
    torch::NoGradGuard no_grad;
    const int64_t T = b.tokens.size(0), S = b.kv_lens.size(0), BS = block_size_;
    const int64_t H = cfg_.n_heads, KV = cfg_.n_kv_heads, hd = head_dim_, I = cfg_.intermediate_size;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    const auto i64 = torch::TensorOptions().dtype(torch::kLong).device(b.tokens.device());

    torch::Tensor tpos = torch::arange(T, i64);
    torch::Tensor ends = b.cu_seqlens_q.slice(0, 1, S + 1).contiguous();
    torch::Tensor seq_idx = torch::searchsorted(ends, tpos, /*out_int32=*/false, /*right=*/true);
    torch::Tensor blk = b.block_tables.index({seq_idx, torch::div(b.positions, BS, "floor")});
    torch::Tensor slot = blk * BS + b.positions.remainder(BS);

    const bool dense_flash = b.max_q > 1 && b.total_kv == T;

    torch::Tensor cu_q32, token_kv_len;
    if (dense_flash) cu_q32 = b.cu_seqlens_q.to(torch::kInt).contiguous();
    else             token_kv_len = b.positions + 1;

    torch::Tensor h = embed_.index_select(0, b.tokens);
    torch::Tensor x = torch::empty_like(h);
    fused::rms_norm(x, h, layers_[0].attn_norm, cfg_.norm_eps);

    for (int32_t l = 0; l < cfg_.n_layers; ++l) {
        const Layer& L = layers_[l];
        torch::Tensor qkv = at::linear(x, L.wqkv);
        torch::Tensor q = torch::empty({T, H, hd}, h.options());

        torch::Tensor k8 = k_cache_.select(0, l).view({-1, KV, hd});
        torch::Tensor v8 = v_cache_.select(0, l).view({-1, KV, hd});
        torch::Tensor ks = k_scale_.select(0, l).view({-1, KV});
        torch::Tensor vs = v_scale_.select(0, l).view({-1, KV});
        torch::Tensor k_dense, v_dense;
        if (dense_flash) {
            k_dense = torch::empty({T, KV, hd}, h.options());
            v_dense = torch::empty({T, KV, hd}, h.options());
        }
        fused::qk_norm_rope_kv_write_fp8(q, qkv, L.q_norm, L.k_norm, cos_, sin_, b.positions, slot, k8, v8, ks, vs,
                                         k_dense, v_dense, H, KV, cfg_.norm_eps);

        torch::Tensor att;
        if (dense_flash) {
            att = flash_attention_dense(q, k_dense, v_dense, cu_q32, b.max_q, scale);
        } else {
            att = torch::empty_like(q);
            paged_attention_fp8(att, q, k_cache_.select(0, l), v_cache_.select(0, l), k_scale_.select(0, l),
                                v_scale_.select(0, l), b.block_tables, seq_idx, token_kv_len, scale);
        }
        torch::Tensor attn_out = at::linear(att.view({T, H * hd}), L.wo);
        all_reduce_in_place(attn_out);
        fused::add_rms_norm(h, attn_out, L.ffn_norm, x, cfg_.norm_eps);

        torch::Tensor gu = at::linear(x, L.w_gate_up);
        torch::Tensor act = torch::empty({T, I}, h.options());
        fused::swiglu(act, gu);
        torch::Tensor mlp = at::linear(act, L.down);
        all_reduce_in_place(mlp);
        if (l + 1 < cfg_.n_layers) fused::add_rms_norm(h, mlp, layers_[l + 1].attn_norm, x, cfg_.norm_eps);
        else h.add_(mlp);
    }
    return h;
}

void Qwen3::all_reduce_in_place(torch::Tensor& tensor) const {
    if (tp_size_ == 1) return;
    const auto stream = c10::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
    ncclAllReduce(tensor.data_ptr(), tensor.data_ptr(),
                            static_cast<size_t>(tensor.numel()), ncclBfloat16,
                            ncclSum, comm_, stream);
}

torch::Tensor Qwen3::all_gather_vocab(const torch::Tensor& local_logits) const {
    if (tp_size_ == 1) return local_logits;
    const int64_t rows = local_logits.size(0);
    const int64_t local_vocab = local_logits.size(1);
    torch::Tensor gathered = torch::empty({tp_size_, rows, local_vocab}, local_logits.options());
    const auto stream = c10::cuda::getCurrentCUDAStream(local_logits.get_device()).stream();
    ncclAllGather(local_logits.data_ptr(), gathered.data_ptr(),
                            static_cast<size_t>(local_logits.numel()), ncclFloat32,
                            comm_, stream);

    return gathered.permute({1, 0, 2}).contiguous().view({rows, global_cfg_.vocab_size});
}

void Qwen3::broadcast_samples(torch::Tensor& samples) const {
    if (tp_size_ == 1 || samples.numel() == 0) return;
    const auto stream = c10::cuda::getCurrentCUDAStream(samples.get_device()).stream();
    ncclBroadcast(samples.data_ptr(), samples.data_ptr(),
                            static_cast<size_t>(samples.numel()), ncclInt64,
                            /*root=*/0, comm_, stream);
}

torch::Tensor Qwen3::compute_logits(const torch::Tensor& h) {
    torch::NoGradGuard no_grad;
    torch::Tensor local = at::linear(rms_norm(h, final_norm_, cfg_.norm_eps), lm_head_).to(torch::kFloat).contiguous();
    return all_gather_vocab(local);
}

torch::Tensor Qwen3::forward(const FlatBatch& b, const torch::Tensor& sample_rows) {
    torch::NoGradGuard no_grad;
    const int64_t S = b.kv_lens.size(0);
    torch::Tensor h = forward_backbone(b);
    torch::Tensor last = b.cu_seqlens_q.slice(0, 1, S + 1) - 1;
    torch::Tensor rows = h.index_select(0, last);
    if (sample_rows.defined()) rows = rows.index_select(0, sample_rows);
    return compute_logits(rows);
}