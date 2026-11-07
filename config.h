#pragma once
#include "sequence.h"
#include <fstream>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

enum class DType { BF16, FP16, FP32 };

struct ModelConfig {
    int32_t dim = 1024;
    int32_t n_layers = 28;
    int32_t n_heads = 16; 
    int32_t n_kv_heads = 8; 
    int32_t head_size = 128; 
    int32_t intermediate_size = 3072;
    int32_t vocab_size = 151936;
    float norm_eps = 1e-6f;
    float rope_theta = 1000000.0f; 
    int32_t max_seq_len = 40960; 
    bool tie_word_embeddings = true; 
    int32_t bos_token_id = 151643;
    std::vector<int32_t> eos_token_ids = {151645}; 

    [[nodiscard]] int32_t head_dim() const noexcept { return head_size; }
    [[nodiscard]] int32_t gqa_group() const noexcept { return n_heads / n_kv_heads; } 

    static ModelConfig qwen3_0_6b() { 
        return ModelConfig{}; 
    }

    static inline ModelConfig from_hf_json(const std::string& path) {
        std::ifstream f(path);
        auto j = nlohmann::json::parse(f);
        ModelConfig c;

        c.dim               = j.at("hidden_size");
        c.n_layers          = j.at("num_hidden_layers");
        c.n_heads           = j.at("num_attention_heads");
        c.n_kv_heads        = j.value("num_key_value_heads", c.n_heads);
        c.head_size         = j.value("head_dim", c.dim / c.n_heads);
        c.vocab_size        = j.at("vocab_size");
        c.intermediate_size = j.at("intermediate_size");
        c.norm_eps          = j.value("rms_norm_eps", 1e-6f);
        c.max_seq_len       = j.value("max_position_embeddings", c.max_seq_len);
        c.tie_word_embeddings = j.value("tie_word_embeddings", false);

        if (j.contains("rope_parameters")) {
            c.rope_theta = j["rope_parameters"].value("rope_theta", c.rope_theta);
        }

        if (j.contains("eos_token_id") && j["eos_token_id"].is_array()) {
            c.eos_token_ids = j["eos_token_id"].get<std::vector<int32_t>>();
        }

        return c;
    }
};

[[nodiscard]] inline size_t kv_bytes_per_block(const ModelConfig& m, size_t dtype_bytes) noexcept {
    return 2ull * m.n_layers * Sequence::BLOCK_SIZE * m.n_kv_heads * m.head_dim() * dtype_bytes;
}

struct EngineConfig {
    std::string model_path;
    size_t max_num_batched_tokens = 4096;
    size_t max_num_seqs = 256;
    size_t max_chunk_tokens = 1024;
    size_t max_model_len = 4096;
    
    float gpu_memory_utilization = 0.90f;
    int32_t num_kv_blocks = -1;
   
    int32_t tensor_parallel_size = 1; 

    bool enforce_eager = false;
    DType dtype = DType::BF16;
    uint64_t seed = 0;
};
