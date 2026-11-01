#pragma once

#include "config.h"
#include <torch/torch.h>

#include <string>
#include <unordered_map>
#include <vector>

using WeightMap = std::unordered_map<std::string, torch::Tensor>;

// One forward pass worth of work, flattened across sequences (all int64 unless noted).
struct FlatBatch {
    torch::Tensor tokens;        // [T]      token ids
    torch::Tensor positions;     // [T]      absolute position of each token
    torch::Tensor cu_seqlens_q;  // [S + 1]  cumulative query lengths, starts with 0
    torch::Tensor kv_lens;       // [S]      total KV length per sequence (cached + new)
    torch::Tensor block_tables;  // [S, W]   physical block ids, -1 padded
    int64_t max_q = 0;
    int64_t max_kv = 0;
};

class Llama {
public:
    explicit Llama(const ModelConfig& cfg);

    void load_weights(const WeightMap& w);
    // k, v: [n_layers, num_blocks, block_size, n_kv_heads, head_dim]
    void bind_kv_cache(torch::Tensor k, torch::Tensor v);

    // Returns fp32 logits [S, V], or [R, V] if sample_rows (indices into the S sequences) is given.
    torch::Tensor forward(const FlatBatch& b, const torch::Tensor& sample_rows = torch::Tensor());

    [[nodiscard]] int64_t block_size() const { return block_size_; }
    [[nodiscard]] const ModelConfig& config() const { return cfg_; }

private:
    struct Layer {
        torch::Tensor wq, wk, wv, wo;
        torch::Tensor gate, up, down;
        torch::Tensor attn_norm, ffn_norm;
    };

    void build_rope_tables();

    ModelConfig cfg_;
    std::vector<Layer> layers_;
    torch::Tensor embed_, final_norm_, lm_head_;
    torch::Tensor cos_, sin_;
    torch::Tensor k_cache_, v_cache_;
    int64_t block_size_ = 0;
    bool loaded_ = false;
};