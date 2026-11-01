#pragma once

#include "sequence.h"
#include "scheduler.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

inline constexpr int32_t kBlockSize = static_cast<int32_t>(Sequence::BLOCK_SIZE);

enum class DType : uint8_t { BF16, FP16, FP32 };

[[nodiscard]] constexpr size_t dtype_bytes(DType d) noexcept {
    switch (d) {
        case DType::BF16: return 2;
        case DType::FP16: return 2;
        case DType::FP32: return 4;
    }
    return 0;
}

struct ModelConfig {
    int32_t dim        = 2048;
    int32_t n_layers   = 16;
    int32_t n_heads    = 32;
    int32_t n_kv_heads = 8;
    int32_t vocab_size = 128256;

    // Read straight from config.json. multiple_of / ffn_dim_multiplier are only used by
    // ffn_hidden_dim() to derive the value for the built-in presets.
    int32_t intermediate_size  = 8192;
    int32_t multiple_of        = 256;
    double  ffn_dim_multiplier = 1.5;

    float   norm_eps = 1e-5f;

    float   rope_theta                = 500000.0f;
    float   rope_scale_factor         = 32.0f;   // 1.0 = plain RoPE, no scaling
    float   rope_low_freq_factor      = 1.0f;
    float   rope_high_freq_factor     = 4.0f;
    int32_t rope_original_max_seq_len = 8192;
    int32_t max_seq_len               = 131072;

    bool    tie_word_embeddings = false;
    int32_t bos_token_id        = 128000;
    std::vector<int32_t> eos_token_ids{128001};

    [[nodiscard]] int32_t head_dim() const noexcept { return dim / n_heads; }
    [[nodiscard]] int32_t n_reps()   const noexcept { return n_heads / n_kv_heads; }
    [[nodiscard]] int32_t q_dim()    const noexcept { return n_heads * head_dim(); }
    [[nodiscard]] int32_t k_dim()    const noexcept { return n_kv_heads * head_dim(); }

    // Defined in config.cpp
    [[nodiscard]] static int32_t     ffn_hidden_dim(int32_t dim, int32_t multiple_of, double multiplier);
    [[nodiscard]] static ModelConfig llama32_1b();
    [[nodiscard]] static ModelConfig from_hf_json(const std::string& path);

    // for testing
    [[nodiscard]] static ModelConfig tiny() {
        ModelConfig c;
        c.dim = 128; c.n_layers = 2; c.n_heads = 4; c.n_kv_heads = 2;
        c.vocab_size = 256; c.multiple_of = 32; c.max_seq_len = 64;
        c.intermediate_size = ffn_hidden_dim(c.dim, c.multiple_of, c.ffn_dim_multiplier);
        c.rope_scale_factor = 1.0f;
        return c;
    }
};

// elem_bytes overloads let ModelRunner pass c10::elementSize(torch dtype) directly.
[[nodiscard]] inline size_t kv_bytes_per_token(const ModelConfig& m, size_t elem_bytes) noexcept {
    return size_t{2} * static_cast<size_t>(m.n_layers) * static_cast<size_t>(m.n_kv_heads) *
           static_cast<size_t>(m.head_dim()) * elem_bytes;
}
[[nodiscard]] inline size_t kv_bytes_per_token(const ModelConfig& m, DType dt) noexcept {
    return kv_bytes_per_token(m, dtype_bytes(dt));
}
[[nodiscard]] inline size_t kv_bytes_per_block(const ModelConfig& m, size_t elem_bytes) noexcept {
    return kv_bytes_per_token(m, elem_bytes) * static_cast<size_t>(kBlockSize);
}
[[nodiscard]] inline size_t kv_bytes_per_block(const ModelConfig& m, DType dt) noexcept {
    return kv_bytes_per_block(m, dtype_bytes(dt));
}

[[nodiscard]] inline int32_t num_kv_blocks_for_budget(size_t budget_bytes, size_t bytes_per_block) noexcept {
    return bytes_per_block == 0 ? 0 : static_cast<int32_t>(budget_bytes / bytes_per_block);
}

struct EngineConfig {
    std::string model_path;
    DType dtype = DType::BF16;

    int32_t max_model_len = 4096;
    size_t  max_num_seqs  = 256;

    size_t  max_num_batched_tokens = 8192;
    size_t  max_chunk_tokens = 0;
    int32_t eos_token_id = 128001;     // unused by LLMEngine now; EOS ids come from config.json

    float   gpu_memory_utilization = 0.9f;
    int32_t num_kv_blocks = 0;         // 0 = size from free GPU memory (CUDA only)

    uint64_t seed = 0;

    // cuda graph
    bool enforce_eager = false;

    void validate() const {
        if (model_path.empty())            throw std::invalid_argument("EngineConfig: model_path is empty");
        if (max_model_len <= 0)            throw std::invalid_argument("EngineConfig: max_model_len must be > 0");
        if (max_num_seqs == 0)             throw std::invalid_argument("EngineConfig: max_num_seqs must be > 0");
        if (max_num_batched_tokens == 0)   throw std::invalid_argument("EngineConfig: max_num_batched_tokens must be > 0");
        if (!(gpu_memory_utilization > 0.0f && gpu_memory_utilization <= 1.0f))
            throw std::invalid_argument("EngineConfig: gpu_memory_utilization must be in (0, 1]");
        if (num_kv_blocks < 0)             throw std::invalid_argument("EngineConfig: num_kv_blocks must be >= 0");
    }

    [[nodiscard]] int32_t max_blocks_per_seq() const noexcept {
        return (max_model_len + kBlockSize - 1) / kBlockSize;
    }

    [[nodiscard]] SchedulerConfig scheduler_config() const noexcept {
        SchedulerConfig s;
        s.max_num_batched_tokens = max_num_batched_tokens;
        s.max_num_seqs = max_num_seqs;
        s.max_chunk_tokens = max_chunk_tokens;
        s.eos_token_id = eos_token_id;
        return s;
    }

    [[nodiscard]] std::vector<int32_t> graph_batch_sizes() const {
        constexpr int32_t kMaxGraphBatch = 512;
        const int32_t cap = static_cast<int32_t>(std::min<size_t>(max_num_seqs, kMaxGraphBatch));

        std::vector<int32_t> sizes;
        for (int32_t b : {1, 2, 4, 8})
            if (b <= cap) sizes.push_back(b);
        for (int32_t b = 16; b <= cap; b += 16) sizes.push_back(b);
        if (sizes.empty() || sizes.back() != cap) sizes.push_back(cap);
        return sizes;
    }
};

[[nodiscard]] inline int32_t pick_graph_batch_size(const std::vector<int32_t>& sizes, int32_t n) noexcept {
    auto it = std::lower_bound(sizes.begin(), sizes.end(), n);
    return it == sizes.end() ? -1 : *it;
}