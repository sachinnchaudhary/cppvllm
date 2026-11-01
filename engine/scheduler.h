#pragma once
#include "block_manager.h"
#include "sequence.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <span>
#include <vector>

struct SchedulerConfig {
    size_t  max_num_batched_tokens = 2048;
    size_t  max_num_seqs = 128;
    size_t  max_chunk_tokens = 0;              // 0 = no per-sequence chunk cap
    int32_t eos_token_id = -1;
    std::vector<int32_t> extra_eos_token_ids;  // models with several EOS ids
};

struct ScheduledBatch {
    std::vector<Sequence*> seqs;
    size_t total_tokens = 0;
    bool all_single_token = false;             // pure decode, for CUDA graphs later

    [[nodiscard]] bool empty() const noexcept { return seqs.empty(); }
};

// True when this step processes the last uncached token, so logits get sampled.
[[nodiscard]] inline bool samples_this_step(const Sequence& s) noexcept {
    return s.num_cached_tokens + s.num_scheduled_tokens == s.size();
}

class Scheduler {
public:
    Scheduler(const SchedulerConfig& cfg, int32_t num_kv_blocks);

    void add_request(Sequence* seq);
    [[nodiscard]] const ScheduledBatch& schedule();
    // sampled[i] belongs to batch.seqs[i]; -1 means "no token sampled" (mid-prefill chunk)
    void postprocess(const ScheduledBatch& batch, std::span<const int32_t> sampled);

    [[nodiscard]] bool is_finished() const noexcept { return waiting_.empty() && running_.empty(); }
    [[nodiscard]] size_t num_waiting() const noexcept { return waiting_.size(); }
    [[nodiscard]] size_t num_running() const noexcept { return running_.size(); }
    [[nodiscard]] size_t num_preemptions() const noexcept { return num_preemptions_; }
    [[nodiscard]] const BlockManager& block_manager() const noexcept { return bm_; }

private:
    void preempt(Sequence* seq);
    [[nodiscard]] bool is_eos(int32_t t) const;

    SchedulerConfig cfg_;
    BlockManager bm_;
    std::deque<Sequence*> waiting_;
    std::vector<Sequence*> running_;   // admission order = priority order (oldest first)

    ScheduledBatch batch_;
    size_t num_preemptions_ = 0;
};