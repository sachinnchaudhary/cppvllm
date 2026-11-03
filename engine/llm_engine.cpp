#include "llm_engine.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>

namespace {

torch::Device pick_device(const std::optional<torch::Device>& d) {
    if (d) return *d;
    return torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);
}

SchedulerConfig make_scheduler_config(const EngineConfig& e, const ModelConfig& m) {
    SchedulerConfig s;
    s.max_num_batched_tokens = e.max_num_batched_tokens;
    s.max_num_seqs           = e.max_num_seqs;
    s.max_chunk_tokens       = e.max_chunk_tokens;
    if (!m.eos_token_ids.empty()) {
        s.eos_token_id = m.eos_token_ids.front();
        s.extra_eos_token_ids.assign(m.eos_token_ids.begin() + 1, m.eos_token_ids.end());
    }
    return s;
}

}  // namespace

LLMEngine::LLMEngine(const EngineConfig& cfg, std::optional<torch::Device> device)
    : cfg_(cfg),
      mcfg_(ModelConfig::from_hf_json(cfg.model_path + "/config.json")),
      runner_(cfg_, mcfg_, pick_device(device)),
      scheduler_(make_scheduler_config(cfg_, mcfg_), runner_.num_kv_blocks()) {
    cfg_.validate();
}

bool LLMEngine::is_eos(int32_t t) const {
    return std::find(mcfg_.eos_token_ids.begin(), mcfg_.eos_token_ids.end(), t) != mcfg_.eos_token_ids.end();
}

uint64_t LLMEngine::add_request(std::span<const int32_t> prompt, const SamplingParams& p) {
    const uint64_t id = next_id_++;
    auto seq = std::make_unique<Sequence>(id, prompt, p);
    scheduler_.add_request(seq.get());      // may throw; seq is simply dropped
    live_.emplace(id, std::move(seq));
    return id;
}

StepInfo LLMEngine::step() {
    StepInfo info;
    if (scheduler_.is_finished()) return info;

    const auto t0 = std::chrono::steady_clock::now();
    const ScheduledBatch& batch = scheduler_.schedule();
    if (batch.empty()) throw std::logic_error("scheduler produced an empty batch while requests are pending");

    const std::vector<int32_t>& sampled = runner_.run(batch);
    scheduler_.postprocess(batch, sampled);

    info.num_seqs = batch.seqs.size();
    info.num_tokens = batch.total_tokens;
    for (int32_t t : sampled) info.num_sampled += (t >= 0);

    std::vector<Sequence*> done;
    for (Sequence* s : batch.seqs) if (s->is_finished()) done.push_back(s);
    info.num_finished = done.size();
    for (Sequence* s : done) harvest(*s);   // destroys the Sequence; batch pointers are dead after this

    stats_.steps += 1;
    stats_.tokens_processed += info.num_tokens;
    stats_.tokens_generated += info.num_sampled;
    stats_.preemptions = scheduler_.num_preemptions();
    stats_.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return info;
}

void LLMEngine::harvest(Sequence& s) {
    GenerationOutput out;
    out.request_id = s.seq_id;
    out.prompt_ids.assign(s.token_ids.begin(), s.token_ids.begin() + s.num_prompt_tokens);
    out.token_ids.assign(s.token_ids.begin() + s.num_prompt_tokens, s.token_ids.end());
    if (!s.params.ignore_eos && !out.token_ids.empty() && is_eos(out.token_ids.back())) {
        out.stopped_on_eos = true;
        out.token_ids.pop_back();
    }
    const uint64_t id = s.seq_id;
    finished_.emplace(id, std::move(out));
    live_.erase(id);
}

std::vector<GenerationOutput> LLMEngine::take_finished() {
    std::vector<GenerationOutput> out;
    out.reserve(finished_.size());
    for (auto& [id, o] : finished_) out.push_back(std::move(o));
    finished_.clear();
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.request_id < b.request_id; });
    return out;
}

std::vector<GenerationOutput> LLMEngine::generate(const std::vector<std::vector<int32_t>>& prompts,
                                                  const SamplingParams& params) {
    std::vector<uint64_t> ids;
    ids.reserve(prompts.size());
    for (const auto& p : prompts) ids.push_back(add_request(p, params));
    while (!is_finished()) step();

    std::vector<GenerationOutput> out;
    out.reserve(ids.size());
    for (uint64_t id : ids) {
        auto it = finished_.find(id);
        if (it == finished_.end()) throw std::logic_error("generate: request " + std::to_string(id) + " never finished");
        out.push_back(std::move(it->second));
        finished_.erase(it);
    }
    return out;
}