#include "model_architecture.h"

#include <cmath>

#ifdef CPPVLLM_HAS_CUDA_KERNELS
#include "kernels/paged_attention.h"
#endif

using torch::indexing::Slice;

namespace {

torch::Tensor rms_norm(const torch::Tensor& x, const torch::Tensor& w, double eps) {
    torch::Tensor xf = x.to(torch::kFloat);
    xf = xf * torch::rsqrt(xf.pow(2).mean(-1, /*keepdim=*/true) + eps);
    return xf.to(x.scalar_type()) * w;
}

torch::Tensor rope(const torch::Tensor& t, const torch::Tensor& cos, const torch::Tensor& sin, int64_t half) {
    torch::Tensor tf = t.to(torch::kFloat);
    torch::Tensor x1 = tf.narrow(-1, 0, half);
    torch::Tensor x2 = tf.narrow(-1, half, half);
    return torch::cat({x1 * cos - x2 * sin, x2 * cos + x1 * sin}, -1).to(t.scalar_type());
}

torch::Tensor take(const WeightMap& w, const std::string& name, std::vector<int64_t> shape) {
    auto it = w.find(name);
    TORCH_CHECK(it != w.end(), "missing weight '", name, "'");
    TORCH_CHECK(it->second.sizes() == c10::IntArrayRef(shape),
                "weight '", name, "' has shape ", it->second.sizes(),
                " but the config expects ", c10::IntArrayRef(shape));
    return it->second;
}

}  // namespace

Qwen3::Qwen3(const ModelConfig& cfg) : cfg_(cfg) {
    build_rope_tables();
}

// Plain RoPE (Qwen3 has no frequency scaling): pair i rotates by position * theta^(-2i/head_dim).
void Qwen3::build_rope_tables() {
    const int64_t hd = cfg_.head_dim();
    const auto f64 = torch::TensorOptions().dtype(torch::kDouble);

    torch::Tensor idx = torch::arange(0, hd, 2, f64);
    torch::Tensor freqs = torch::pow(static_cast<double>(cfg_.rope_theta), idx / static_cast<double>(hd)).reciprocal();

    torch::Tensor angle = torch::outer(torch::arange(cfg_.max_seq_len, f64), freqs);   // [max_seq_len, hd/2]
    cos_ = angle.cos().to(torch::kFloat);
    sin_ = angle.sin().to(torch::kFloat);
}

void Qwen3::load_weights(const WeightMap& w) {
    const int64_t D = cfg_.dim, H = cfg_.n_heads, KV = cfg_.n_kv_heads, hd = cfg_.head_dim(),
                  I = cfg_.intermediate_size, V = cfg_.vocab_size;

    embed_ = take(w, "model.embed_tokens.weight", {V, D});
    final_norm_ = take(w, "model.norm.weight", {D});
    if (w.count("lm_head.weight")) lm_head_ = take(w, "lm_head.weight", {V, D});
    else if (cfg_.tie_word_embeddings) lm_head_ = embed_;
    else TORCH_CHECK(false, "no lm_head.weight in the file and tie_word_embeddings is false");

    layers_.clear();
    layers_.reserve(cfg_.n_layers);
    for (int32_t i = 0; i < cfg_.n_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        Layer L;
        L.wq        = take(w, p + "self_attn.q_proj.weight", {H * hd, D});
        L.wk        = take(w, p + "self_attn.k_proj.weight", {KV * hd, D});
        L.wv        = take(w, p + "self_attn.v_proj.weight", {KV * hd, D});
        L.wo        = take(w, p + "self_attn.o_proj.weight", {D, H * hd});
        L.q_norm    = take(w, p + "self_attn.q_norm.weight", {hd});     // Qwen3's QK-norm weights
        L.k_norm    = take(w, p + "self_attn.k_norm.weight", {hd});
        L.gate      = take(w, p + "mlp.gate_proj.weight", {I, D});
        L.up        = take(w, p + "mlp.up_proj.weight", {I, D});
        L.down      = take(w, p + "mlp.down_proj.weight", {D, I});
        L.attn_norm = take(w, p + "input_layernorm.weight", {D});
        L.ffn_norm  = take(w, p + "post_attention_layernorm.weight", {D});
        TORCH_CHECK(L.wq.device() == embed_.device() && L.wq.scalar_type() == embed_.scalar_type(),
                    "all weights must share one device and dtype (layer ", i, ")");
        layers_.push_back(std::move(L));
    }
    cos_ = cos_.to(embed_.device());
    sin_ = sin_.to(embed_.device());
    loaded_ = true;
}

void Qwen3::bind_kv_cache(torch::Tensor k, torch::Tensor v) {
    TORCH_CHECK(loaded_, "call load_weights() before bind_kv_cache()");
    TORCH_CHECK(k.dim() == 5 && k.sizes() == v.sizes(), "KV cache must be [layers, blocks, block_size, kv_heads, head_dim]");
    TORCH_CHECK(k.size(0) == cfg_.n_layers && k.size(3) == cfg_.n_kv_heads && k.size(4) == cfg_.head_dim(),
                "KV cache shape does not match the model config");
    TORCH_CHECK(k.scalar_type() == embed_.scalar_type() && v.scalar_type() == embed_.scalar_type() &&
                k.device() == embed_.device() && v.device() == embed_.device(),
                "KV cache must have the dtype and device of the weights");
    TORCH_CHECK(k.is_contiguous() && v.is_contiguous(), "KV cache must be contiguous");
    k_cache_ = std::move(k);
    v_cache_ = std::move(v);
    block_size_ = k_cache_.size(2);
}

bool Qwen3::fused_attention() const {
#ifdef CPPVLLM_HAS_CUDA_KERNELS
    const int64_t hd = cfg_.head_dim(), rep = cfg_.n_heads / cfg_.n_kv_heads;
    return loaded_ && embed_.is_cuda() && (hd == 64 || hd == 128) &&
           (rep == 1 || rep == 2 || rep == 4 || rep == 8);
#else
    return false;
#endif
}

torch::Tensor Qwen3::forward(const FlatBatch& b, const torch::Tensor& sample_rows) {
    torch::NoGradGuard no_grad;

    const int64_t T = b.tokens.size(0), S = b.kv_lens.size(0), BS = block_size_;
    const int64_t H = cfg_.n_heads, KV = cfg_.n_kv_heads, hd = cfg_.head_dim(), half = hd / 2, rep = H / KV;
    const int64_t max_q = b.max_q, max_kv = b.max_kv;
    const bool fused = fused_attention();
    const double scale = 1.0 / std::sqrt(static_cast<double>(hd));

    const auto dev = b.tokens.device();
    const auto i64 = torch::TensorOptions().dtype(torch::kLong).device(dev);

    // token -> sequence it belongs to
    torch::Tensor tpos = torch::arange(T, i64);
    torch::Tensor ends = b.cu_seqlens_q.slice(0, 1, S + 1).contiguous();
    torch::Tensor seq_idx = torch::searchsorted(ends, tpos, /*out_int32=*/false, /*right=*/true);

    // where each new K/V goes in the paged cache
    torch::Tensor blk = b.block_tables.index({seq_idx, torch::div(b.positions, BS, "floor")});
    torch::Tensor slot = blk * BS + b.positions.remainder(BS);

    torch::Tensor token_kv_len;                         // fused path
    torch::Tensor q_idx, kv_blk, kv_off, mask;          // padded fallback path
    if (fused) {
        token_kv_len = b.positions + 1;                 // each token attends to positions 0..pos: causal for free
    } else {
        torch::Tensor starts = b.cu_seqlens_q.slice(0, 0, S);
        q_idx = tpos - starts.index_select(0, seq_idx);

        torch::Tensor kvp = torch::arange(max_kv, i64);
        kv_blk = b.block_tables.index({Slice(), torch::div(kvp, BS, "floor")}).clamp_min(0);
        kv_off = kvp.remainder(BS);

        torch::Tensor qpos = torch::zeros({S, max_q}, i64);
        qpos.index_put_({seq_idx, q_idx}, b.positions);
        torch::Tensor qvalid = torch::zeros({S, max_q}, i64.dtype(torch::kBool));
        qvalid.index_put_({seq_idx, q_idx}, torch::ones({T}, i64.dtype(torch::kBool)));
        torch::Tensor kv_row = kvp.view({1, 1, -1});
        torch::Tensor allowed = (kv_row <= qpos.unsqueeze(2)) & (kv_row < b.kv_lens.view({S, 1, 1}));
        allowed.select(2, 0).logical_or_(qvalid.logical_not());
        mask = allowed.unsqueeze(1);
    }

    torch::Tensor cos = cos_.index_select(0, b.positions).unsqueeze(1);
    torch::Tensor sin = sin_.index_select(0, b.positions).unsqueeze(1);

    torch::Tensor h = embed_.index_select(0, b.tokens);
    for (int32_t l = 0; l < cfg_.n_layers; ++l) {
        const Layer& L = layers_[l];

        torch::Tensor x = rms_norm(h, L.attn_norm, cfg_.norm_eps);
        torch::Tensor q = at::linear(x, L.wq).view({T, H, hd});
        torch::Tensor k = at::linear(x, L.wk).view({T, KV, hd});
        torch::Tensor v = at::linear(x, L.wv).view({T, KV, hd});

        q = rms_norm(q, L.q_norm, cfg_.norm_eps);      // Qwen3: RMSNorm over head_dim, per head, BEFORE RoPE
        k = rms_norm(k, L.k_norm, cfg_.norm_eps);
        q = rope(q, cos, sin, half);
        k = rope(k, cos, sin, half);

        // write this step's K/V first: the kernel (and the gather) read them back from the cache
        k_cache_.select(0, l).view({-1, KV, hd}).index_copy_(0, slot, k);
        v_cache_.select(0, l).view({-1, KV, hd}).index_copy_(0, slot, v);

        torch::Tensor o;
        if (fused) {
#ifdef CPPVLLM_HAS_CUDA_KERNELS
            torch::Tensor att = torch::empty_like(q);   // [T, H, hd]
            paged_attention(att, q.contiguous(), k_cache_.select(0, l), v_cache_.select(0, l),
                            b.block_tables, seq_idx, token_kv_len, scale);
            o = att.view({T, H * hd});
#endif
        } else {
            torch::Tensor keys = k_cache_.select(0, l).index({kv_blk, kv_off}).transpose(1, 2);   // [S, KV, max_kv, hd]
            torch::Tensor vals = v_cache_.select(0, l).index({kv_blk, kv_off}).transpose(1, 2);
            keys = keys.repeat_interleave(rep, 1);                                                // [S, H, max_kv, hd]
            vals = vals.repeat_interleave(rep, 1);

            torch::Tensor q_pad = torch::zeros({S, max_q, H, hd}, q.options());
            q_pad.index_put_({seq_idx, q_idx}, q);
            torch::Tensor att = torch::scaled_dot_product_attention(
                q_pad.transpose(1, 2), keys, vals, mask, /*dropout_p=*/0.0, /*is_causal=*/false, c10::nullopt);
            o = att.transpose(1, 2).index({seq_idx, q_idx}).reshape({T, H * hd});
        }
        h = h + at::linear(o, L.wo);

        x = rms_norm(h, L.ffn_norm, cfg_.norm_eps);
        h = h + at::linear(torch::silu(at::linear(x, L.gate)) * at::linear(x, L.up), L.down);
    }

    torch::Tensor last = b.cu_seqlens_q.slice(0, 1, S + 1) - 1;   // last token of each sequence
    torch::Tensor rows = h.index_select(0, last);
    if (sample_rows.defined()) rows = rows.index_select(0, sample_rows);
    return at::linear(rms_norm(rows, final_norm_, cfg_.norm_eps), lm_head_).to(torch::kFloat);
}