#pragma once
#include "config.h"
#include "model_runner.h"
#include "scheduler.h"
#include "sequence.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

struct GenerationOutput {
    uint64_t request_id = 0;
    std::vector<int32_t> prompt_ids;
    std::vector<int32_t> token_ids;
    bool stopped_on_eos = false;
};

struct StepInfo {
    size_t num_seqs = 0, num_tokens = 0, num_sampled = 0, num_finished = 0;
};

struct EngineStats {
    size_t steps = 0, tokens_processed = 0, tokens_generated = 0;
    size_t preemptions = 0;
    double seconds = 0;
    [[nodiscard]] double generated_per_second() const { return seconds > 0 ? static_cast<double>(tokens_generated) / seconds : 0; }
};


class LLMEngine {
public:
    explicit LLMEngine(const EngineConfig& cfg, std::optional<torch::Device> device = std::nullopt);

    uint64_t add_request(std::span<const int32_t> prompt, const SamplingParams& params);
    StepInfo step();
    [[nodiscard]] bool is_finished() const noexcept { return scheduler_.is_finished(); }

    std::vector<GenerationOutput> generate(const std::vector<std::vector<int32_t>>& prompts, const SamplingParams& params);
    std::vector<GenerationOutput> take_finished();

    [[nodiscard]] const EngineStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const ModelConfig& model_config() const noexcept { return mcfg_; }
    [[nodiscard]] int32_t num_kv_blocks() const noexcept { return runner_.num_kv_blocks(); }

private:
    void harvest(Sequence& s);
    [[nodiscard]] bool is_eos(int32_t t) const;

    EngineConfig cfg_;
    ModelConfig mcfg_;
    ModelRunner runner_;
    Scheduler scheduler_;

    std::unordered_map<uint64_t, std::unique_ptr<Sequence>> live_;
    std::unordered_map<uint64_t, GenerationOutput> finished_;
    uint64_t next_id_ = 0;
    EngineStats stats_;
};