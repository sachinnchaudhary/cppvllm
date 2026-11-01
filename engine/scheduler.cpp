#include "scheduler.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

Scheduler::Scheduler(const SchedulerConfig& cfg, int32_t num_kv_blocks)
    : cfg_(cfg), bm_(num_kv_blocks) {
    if (cfg_.max_num_batched_tokens == 0 || cfg_.max_num_seqs == 0)
        throw std::invalid_argument("Scheduler: max_num_batched_tokens and max_num_seqs must be > 0");
    batch_.seqs.reserve(cfg_.max_num_seqs);
}

bool Scheduler::is_eos(int32_t t) const {
    return t == cfg_.eos_token_id ||
           std::find(cfg_.extra_eos_token_ids.begin(), cfg_.extra_eos_token_ids.end(), t) !=
               cfg_.extra_eos_token_ids.end();
}

void Scheduler::add_request(Sequence* seq) {
    // A sequence that cannot fit in the whole cache even when running alone would be preempted
    // forever, so reject it up front.
    const size_t max_len = seq->num_prompt_tokens + static_cast<size_t>(std::max(seq->params.max_tokens, 0));
    const size_t blocks = (max_len + Sequence::BLOCK_SIZE - 1) / Sequence::BLOCK_SIZE;
    if (blocks > static_cast<size_t>(bm_.num_blocks()))
        throw std::invalid_argument("request needs " + std::to_string(blocks) + " KV blocks but the cache only has " +
                                    std::to_string(bm_.num_blocks()));
    seq->status = Status::WAITING;
    waiting_.push_back(seq);
}

void Scheduler::preempt(Sequence* seq) {
    bm_.deallocate(*seq);          // needs the block table, so before reset
    seq->reset_for_preemption();   // keeps token_ids: the sequence is recomputed from scratch
    waiting_.push_front(seq);      // victims are picked newest-first, so oldest ends up in front
    ++num_preemptions_;
}

const ScheduledBatch& Scheduler::schedule() {
    batch_.seqs.clear();
    batch_.total_tokens = 0;
    batch_.all_single_token = false;

    size_t budget = cfg_.max_num_batched_tokens;
    const size_t chunk_cap = cfg_.max_chunk_tokens > 0 ? cfg_.max_chunk_tokens : std::numeric_limits<size_t>::max();
    bool preempted_now = false;

    auto push = [&](Sequence* s, size_t n) {
        s->num_scheduled_tokens = static_cast<uint32_t>(n);
        batch_.seqs.push_back(s);
        batch_.total_tokens += n;
        budget -= n;
    };

    // 1) Running sequences first (decodes and unfinished prefill chunks), oldest first.
    std::deque<Sequence*> pending(running_.begin(), running_.end());
    std::vector<Sequence*> kept;
    kept.reserve(running_.size());

    while (!pending.empty()) {
        Sequence* seq = pending.front();
        pending.pop_front();

        const size_t want = std::min({seq->num_uncached_tokens(), budget, chunk_cap});
        if (want == 0 || batch_.seqs.size() >= cfg_.max_num_seqs) {   // no room this step, stays running
            kept.push_back(seq);
            continue;
        }

        // Make sure its block table covers the tokens it will hold. If the cache is full, evict the
        // newest running sequences; if none are left, this sequence is itself the newest, so it yields.
        bool yielded = false;
        while (!bm_.can_append(*seq)) {
            Sequence* victim;
            if (!pending.empty()) { victim = pending.back(); pending.pop_back(); }
            else                  { victim = seq; yielded = true; }
            preempt(victim);
            preempted_now = true;
            if (yielded) break;
        }
        if (yielded) continue;

        bm_.may_append(*seq);
        push(seq, want);
        kept.push_back(seq);
    }
    running_ = std::move(kept);

    // 2) Admit waiting sequences FCFS. Skipped after a preemption, otherwise we would just
    //    re-admit the sequence we evicted.
    while (!preempted_now && !waiting_.empty() && budget > 0 && running_.size() < cfg_.max_num_seqs) {
        Sequence* seq = waiting_.front();
        if (!bm_.can_allocate(*seq)) break;
        waiting_.pop_front();
        bm_.allocate(*seq);                  // sets num_cached_tokens from the prefix cache
        seq->status = Status::RUNNING;
        running_.push_back(seq);
        push(seq, std::min({seq->num_uncached_tokens(), budget, chunk_cap}));
    }

    batch_.all_single_token =
        !batch_.seqs.empty() &&
        std::all_of(batch_.seqs.begin(), batch_.seqs.end(), [](const Sequence* s) { return s->num_scheduled_tokens == 1; });
    return batch_;
}

void Scheduler::postprocess(const ScheduledBatch& batch, std::span<const int32_t> sampled) {
    if (sampled.size() != batch.seqs.size())
        throw std::invalid_argument("Scheduler::postprocess: sampled.size() != batch size");

    for (size_t i = 0; i < batch.seqs.size(); ++i) {
        Sequence* seq = batch.seqs[i];
        seq->num_cached_tokens += seq->num_scheduled_tokens;
        seq->num_scheduled_tokens = 0;
        bm_.commit(*seq);                      // register newly completed blocks in the prefix cache

        const int32_t tok = sampled[i];
        if (tok < 0) continue;                 // mid-prefill chunk, nothing sampled
        seq->append_token(tok);

        const bool hit_eos = !seq->params.ignore_eos && is_eos(tok);
        const bool hit_len = static_cast<int64_t>(seq->num_completion_tokens()) >= seq->params.max_tokens;
        if (hit_eos || hit_len) {
            seq->status = Status::FINISHED;
            bm_.deallocate(*seq);
        }
    }
    std::erase_if(running_, [](const Sequence* s) { return s->is_finished(); });
}