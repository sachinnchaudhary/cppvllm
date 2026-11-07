#include "llm_engine.h"

#include <algorithm>
#include <cassert>
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

}  

LLMEngine::LLMEngine(const EngineConfig& cfg, std::optional<torch::Device> device)
    : cfg_(cfg),
      mcfg_(ModelConfig::from_hf_json(cfg.model_path + "/config.json")),
      runner_(cfg_, mcfg_, pick_device(device), cfg_.tensor_parallel_size),
      scheduler_(make_scheduler_config(cfg_, mcfg_), runner_.num_kv_blocks()) {
    for (auto& b : ring_) b.seqs.reserve(cfg_.max_num_seqs);
}

bool LLMEngine::is_eos(int32_t t) const {
    return std::find(mcfg_.eos_token_ids.begin(), mcfg_.eos_token_ids.end(), t) != mcfg_.eos_token_ids.end();
}

uint64_t LLMEngine::add_request(std::span<const int32_t> prompt, const SamplingParams& p) {
    const uint64_t id = next_id_++;
    auto seq = std::make_unique<Sequence>(id, prompt, p);
    scheduler_.add_request(seq.get());      
    live_.emplace(id, std::move(seq));
    return id;
}

void LLMEngine::retire_oldest(StepInfo& info) {
    assert(count_ > 0);
    ScheduledBatch& b = ring_[head_];
    const std::vector<int32_t>& sampled = runner_.collect();
    info.num_sampled += scheduler_.postprocess(b, sampled);
    for (Sequence* s : b.seqs) {
        if (s->is_finished() && s->num_inflight == 0) {
            ++info.num_finished;
            harvest(*s);                      
        }
    }
    head_ = (head_ + 1) % kDepth;
    --count_;
}

void LLMEngine::sync() {
    StepInfo ignored;
    while (count_ > 0) retire_oldest(ignored);
}

StepInfo LLMEngine::step() {
    StepInfo info;
    const auto t0 = std::chrono::steady_clock::now();
    const size_t sampled_before = stats_.tokens_generated;

    auto account = [&](size_t retired_sampled) {
        stats_.tokens_generated += retired_sampled;
        stats_.preemptions = scheduler_.num_preemptions();
        stats_.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };
    (void)sampled_before;

    if (scheduler_.is_finished()) {          
        sync();
        return info;
    }

    ScheduledBatch& batch = ring_[(head_ + count_) % kDepth];
    scheduler_.schedule(batch, /*allow_preempt=*/count_ == 0);

    if (batch.empty()) {
        if (count_ == 0) throw std::logic_error("scheduler produced an empty batch while requests are pending");
        while (count_ > 0) retire_oldest(info);
        account(info.num_sampled);
        return info;
    }

    runner_.launch(batch);                  
    scheduler_.on_launched(batch);         
    ++count_;

    info.num_seqs = batch.seqs.size();
    info.num_tokens = batch.total_tokens;
    stats_.steps += 1;
    stats_.tokens_processed += info.num_tokens;

    if (batch.blocked) {
        while (count_ > 0) retire_oldest(info);   // drain so the next schedule() may preempt
    } else if (count_ == kDepth) {
        retire_oldest(info);
    }
    account(info.num_sampled);
    return info;
}

void LLMEngine::harvest(Sequence& s) {
    GenerationOutput out;
    out.request_id = s.seq_id;
    out.prompt_ids.assign(s.token_ids.begin(), s.token_ids.begin() + s.num_prompt_tokens);
    out.token_ids.assign(s.token_ids.begin() + s.num_prompt_tokens, s.token_ids.end());
    assert(std::none_of(out.token_ids.begin(), out.token_ids.end(), [](int32_t t) { return t == kPendingToken; }));
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