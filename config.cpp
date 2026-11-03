#include "config.h"

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

using nlohmann::json;

void ModelConfig::validate() const {
    auto bad = [](const char* msg) { throw std::invalid_argument(std::string("ModelConfig: ") + msg); };
    if (dim <= 0 || n_layers <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_size <= 0 || vocab_size <= 0 ||
        intermediate_size <= 0)
        bad("all sizes must be positive");
    if (n_heads % n_kv_heads != 0) bad("n_heads must be divisible by n_kv_heads");
    if (head_size % 2 != 0)        bad("head_dim must be even (RoPE rotates pairs)");
    if (max_seq_len <= 0)          bad("max_position_embeddings must be positive");
    if (eos_token_ids.empty())     bad("need at least one eos token id");
}

ModelConfig ModelConfig::qwen3_0_6b() {
    ModelConfig c;   // the defaults are the 0.6B numbers
    c.validate();
    return c;
}

ModelConfig ModelConfig::from_hf_json(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open config file: " + path);

    json j;
    try {
        j = json::parse(f);
    } catch (const json::exception& e) {
        throw std::runtime_error("bad JSON in " + path + ": " + e.what());
    }

    ModelConfig c;
    try {
        // Refuse a model we cannot run exactly, otherwise a Llama or Gemma folder would load and produce wrong numbers.
        if (j.value("model_type", "qwen3") != "qwen3")
            throw std::runtime_error("model_type '" + j.value("model_type", std::string()) + "' is not supported (only qwen3)");
        if (j.value("attention_bias", false)) throw std::runtime_error("attention biases are not supported");
        if (j.value("hidden_act", "silu") != "silu") throw std::runtime_error("only the silu activation is supported");
        if (j.value("use_sliding_window", false)) throw std::runtime_error("sliding-window attention is not supported");

        c.dim                 = j.at("hidden_size").get<int32_t>();
        c.n_layers            = j.at("num_hidden_layers").get<int32_t>();
        c.n_heads             = j.at("num_attention_heads").get<int32_t>();
        c.n_kv_heads          = j.value("num_key_value_heads", c.n_heads);
        c.head_size           = j.value("head_dim", c.dim / c.n_heads);   // Qwen3 files always carry it
        c.vocab_size          = j.at("vocab_size").get<int32_t>();
        c.intermediate_size   = j.at("intermediate_size").get<int32_t>();
        c.norm_eps            = j.value("rms_norm_eps", 1e-6f);
        c.max_seq_len         = j.value("max_position_embeddings", c.max_seq_len);
        c.tie_word_embeddings = j.value("tie_word_embeddings", false);
        c.bos_token_id        = j.value("bos_token_id", c.bos_token_id);

        // RoPE: older files keep "rope_theta" at the top level, newer ones put it inside "rope_parameters".
        c.rope_theta = j.value("rope_theta", c.rope_theta);
        const json* rp = (j.contains("rope_parameters") && j["rope_parameters"].is_object()) ? &j["rope_parameters"]
                       : (j.contains("rope_scaling")    && j["rope_scaling"].is_object())    ? &j["rope_scaling"] : nullptr;
        if (rp) {
            c.rope_theta = rp->value("rope_theta", c.rope_theta);
            const std::string type = rp->value("rope_type", rp->value("type", std::string("default")));
            if (type != "default")
                throw std::runtime_error("rope type '" + type + "' is not supported (plain RoPE only; YaRN is not implemented)");
        }

        if (j.contains("eos_token_id") && !j["eos_token_id"].is_null()) {   // an int, or a list of ints
            const json& e = j["eos_token_id"];
            std::vector<int32_t> ids = e.is_array() ? e.get<std::vector<int32_t>>() : std::vector<int32_t>{e.get<int32_t>()};
            if (!ids.empty()) c.eos_token_ids = std::move(ids);
        }
    } catch (const json::exception& e) {
        throw std::runtime_error("config.json is missing or has a wrongly typed field: " + std::string(e.what()));
    }

    c.validate();
    return c;
}

void EngineConfig::validate() const {
    auto bad = [](const char* msg) { throw std::invalid_argument(std::string("EngineConfig: ") + msg); };
    if (max_num_seqs == 0) bad("max_num_seqs must be > 0");
    // every running sequence needs at least 1 token of budget per step
    if (max_num_batched_tokens < max_num_seqs) bad("max_num_batched_tokens must be >= max_num_seqs");
    if (max_model_len == 0) bad("max_model_len must be > 0");
    if (!(gpu_memory_utilization > 0.0f && gpu_memory_utilization <= 1.0f)) bad("gpu_memory_utilization must be in (0, 1]");
    if (num_kv_blocks == 0 || num_kv_blocks < -1) bad("num_kv_blocks must be -1 (auto) or > 0");
}