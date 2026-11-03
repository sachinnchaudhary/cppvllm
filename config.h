#pragma once
#include "sequence.h"   // Sequence::BLOCK_SIZE

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum class DType { BF16, FP16, FP32 };

// Qwen3 dense model shape. Defaults are Qwen3-0.6B; from_hf_json() overwrites them from the model's config.json.
struct ModelConfig {
    int32_t dim               = 1024;      // hidden_size
    int32_t n_layers          = 28;
    int32_t n_heads           = 16;        // query heads
    int32_t n_kv_heads        = 8;         // key/value heads (grouped-query attention)
    int32_t head_size         = 128;       // Qwen3 sets this explicitly: it is NOT dim / n_heads (16 * 128 = 2048 != 1024)
    int32_t intermediate_size = 3072;
    int32_t vocab_size        = 151936;
    float   norm_eps          = 1e-6f;
    float   rope_theta        = 1000000.0f;   // plain RoPE, no scaling
    int32_t max_seq_len       = 40960;        // max_position_embeddings
    bool    tie_word_embeddings = true;       // the 0.6B shares embed_tokens and lm_head
    int32_t bos_token_id      = 151643;
    std::vector<int32_t> eos_token_ids = {151645};   // <|im_end|>

    [[nodiscard]] int32_t head_dim()  const noexcept { return head_size; }
    [[nodiscard]] int32_t gqa_group() const noexcept { return n_heads / n_kv_heads; }   // q heads per kv head

    void validate() const;                                  // throws std::invalid_argument
    static ModelConfig qwen3_0_6b();                        // the defaults above, no files needed
    static ModelConfig from_hf_json(const std::string& path);   // throws std::runtime_error on unsupported input
};

// Bytes of KV cache for ONE block across all layers: 2 (K and V) * layers * block_size * kv_heads * head_dim * bytes.
// Qwen3-0.6B in bf16: 2 * 28 * 16 * 8 * 128 * 2 = 1.75 MiB per block (112 KiB per token).
[[nodiscard]] inline size_t kv_bytes_per_block(const ModelConfig& m, size_t dtype_bytes) noexcept {
    return 2ull * m.n_layers * Sequence::BLOCK_SIZE * m.n_kv_heads * m.head_dim() * dtype_bytes;
}

struct EngineConfig {
    std::string model_path;                    // folder with config.json + *.safetensors

    // scheduler
    size_t max_num_batched_tokens = 4096;      // token budget per step
    size_t max_num_seqs           = 256;       // sequences running at once
    size_t max_chunk_tokens       = 0;         // per-sequence cap per step, 0 = none
    size_t max_model_len          = 4096;      // longest prompt + generation we accept (capped by the model's max_seq_len)

    // memory
    float   gpu_memory_utilization = 0.90f;    // fraction of GPU memory we may use in total
    int32_t num_kv_blocks          = -1;       // -1 = size from free memory at startup (GPU only)

    // speed / numerics
    bool     enforce_eager = false;            // true = never use CUDA graphs
    DType    dtype         = DType::BF16;      // GPU dtype; the CPU path always runs fp32
    uint64_t seed          = 0;

    void validate() const;                     // throws std::invalid_argument
};