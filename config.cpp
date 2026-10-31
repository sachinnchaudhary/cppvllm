#include "config.h"
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

using nlohmann::json;

int32_t ModelConfig::ffn_hidden_dim(int32_t dim, int32_t multiple_of, double multiplier) {
    const int64_t base   = (2LL * (4LL * dim)) / 3;
    const int64_t scaled = static_cast<int64_t>(multiplier * static_cast<double>(base));
    return static_cast<int32_t>(multiple_of * ((scaled + multiple_of - 1) / multiple_of));
}

ModelConfig ModelConfig::llama32_1b() {
    ModelConfig c;
    c.intermediate_size = ffn_hidden_dim(c.dim, 256, 1.5);
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
        c.dim                 = j.at("hidden_size").get<int32_t>();
        c.n_layers            = j.at("num_hidden_layers").get<int32_t>();
        c.n_heads             = j.at("num_attention_heads").get<int32_t>();
        c.n_kv_heads          = j.value("num_key_value_heads", c.n_heads);
        c.vocab_size          = j.at("vocab_size").get<int32_t>();
        c.intermediate_size   = j.at("intermediate_size").get<int32_t>();
        c.norm_eps            = j.value("rms_norm_eps", 1e-5f);
        c.max_seq_len         = j.value("max_position_embeddings", c.max_seq_len);
        c.tie_word_embeddings = j.value("tie_word_embeddings", false);
        c.bos_token_id        = j.value("bos_token_id", c.bos_token_id);

        if (j.contains("eos_token_id")) {
            const json& e = j["eos_token_id"];
            c.eos_token_ids = e.is_array() ? e.get<std::vector<int32_t>>()
                                           : std::vector<int32_t>{e.get<int32_t>()};
        }
    } catch (const json::exception& e) {
        throw std::runtime_error("config.json is missing or has a wrongly typed field: " + std::string(e.what()));
    }

    return c;
}