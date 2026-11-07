#pragma once

#include "config.h"
#include <torch/torch.h>
#include <nccl.h>
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
    int64_t total_kv = 0;        // sum of kv_lens, computed on the HOST: == T iff no sequence has cached tokens (fresh
                                 // prefill). Left 0 for decode / CUDA-graph batches, which never use it.
};

class Qwen3 {
public:

    explicit Qwen3(const ModelConfig& cfg, int tp_rank = 0, int tp_size = 1,
                    ncclComm_t comm = nullptr);

    void load_weights(const WeightMap& w, torch::Device target_device);

    void bind_kv_cache(torch::Tensor k, torch::Tensor v, torch::Tensor k_scale, torch::Tensor v_scale);

    torch::Tensor forward_backbone(const FlatBatch& b);

    torch::Tensor compute_logits(const torch::Tensor& h);

    torch::Tensor forward(const FlatBatch& b, const torch::Tensor& sample_rows = torch::Tensor());

    void broadcast_samples(torch::Tensor& samples) const;

    [[nodiscard]] int64_t block_size() const { return block_size_; }
    
    [[nodiscard]] const ModelConfig& config() const { return cfg_; }
    [[nodiscard]] const ModelConfig& global_config() const { return global_cfg_; }
    [[nodiscard]] int tp_rank() const { return tp_rank_; }
    [[nodiscard]] int tp_size() const { return tp_size_; }
    [[nodiscard]] int64_t head_dim() const { return head_dim_; }

private:
    struct Layer {
        torch::Tensor wqkv, wo;                // rank-local Q/K/V rows; rank-local input columns in wo
        torch::Tensor q_norm, k_norm;          // [head_dim] weights of the per-head RMSNorms (replicated)
        torch::Tensor w_gate_up, down;         // rank-local gate/up rows; rank-local input columns in down
        torch::Tensor attn_norm, ffn_norm;     // replicated
    };

    void build_rope_tables();
    void all_reduce_in_place(torch::Tensor& tensor) const;
    torch::Tensor all_gather_vocab(const torch::Tensor& local_logits) const;

    ModelConfig global_cfg_;  // Original, unsharded checkpoint dimensions.
    ModelConfig cfg_;         // Rank-local dimensions used by this model.
    int tp_rank_ = 0;
    int tp_size_ = 1;
    int64_t head_dim_ = 0;     // Invariant under TP; cached before local head counts are divided.
    ncclComm_t comm_ = nullptr;

    std::vector<Layer> layers_;
    torch::Tensor embed_, final_norm_, lm_head_;
    torch::Tensor cos_, sin_;
    torch::Tensor k_cache_, v_cache_;
    torch::Tensor k_scale_, v_scale_;
    int64_t block_size_ = 0;
};