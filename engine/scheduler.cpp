#include "scheduler.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <string>

Scheduler::Scheduler(const SchedulerConfig& cfg, int32_t num_kv_blocks)
    : cfg_(cfg), bm_(num_kv_blocks) {}

bool Scheduler::is_eos(int32_t t) const {
    return t == cfg_.eos_token_id ||
           std::find(cfg_.extra_eos_token_ids.begin(), cfg_.extra_eos_token_ids.end(), t) !=
               cfg_.extra_eos_token_ids.end();
}

void Scheduler::add_request(Sequence* seq) {
    const size_t max_len = seq->num_prompt_tokens + static_cast<size_t>(std::max(seq->params.max_tokens, 0));
    const size_t blocks = (max_len + Sequence::BLOCK_SIZE - 1) / Sequence::BLOCK_SIZE;
    if (blocks > static_cast<size_t>(bm_.num_blocks()))
        throw std::invalid_argument("request needs " + std::to_string(blocks) + " KV blocks but the cache only has " +
                                    std::to_string(bm_.num_blocks()));
    seq->status = Status::WAITING;
    waiting_.push_back(seq);
}

void Scheduler::preempt(Sequence* seq) {
    assert(seq->num_inflight == 0 && "never preempt a sequence that has a step in flight");
    bm_.deallocate(*seq);         
    seq->reset_for_preemption();  
    waiting_.push_front(seq);      
    ++num_preemptions_;
}

void Scheduler::schedule(ScheduledBatch& out, bool allow_preempt) {
    out.seqs.clear();
    out.sample_pos.clear();
    out.total_tokens = 0;
    out.all_single_token = false;
    out.blocked = false;

    size_t budget = cfg_.max_num_batched_tokens;
    const size_t chunk_cap = cfg_.max_chunk_tokens > 0 ? cfg_.max_chunk_tokens : std::numeric_limits<size_t>::max();
    bool preempted_now = false;
    bool blocked = false;

    auto push = [&](Sequence* s, size_t n) {
        s->num_scheduled_tokens = static_cast<uint32_t>(n);
        out.seqs.push_back(s);
        out.total_tokens += n;
        budget -= n;
    };

    std::deque<Sequence*> pending(running_.begin(), running_.end());
    std::vector<Sequence*> kept;
    kept.reserve(running_.size());

    while (!pending.empty()) {
        Sequence* seq = pending.front();
        pending.pop_front();
        if (seq->num_completion_tokens() > 0 &&
            static_cast<int64_t>(seq->num_completion_tokens()) >= seq->params.max_tokens) {
            kept.push_back(seq);
            continue;
        }

        const size_t want = std::min({seq->num_uncached_tokens(), budget, chunk_cap});
        if (want == 0 || out.seqs.size() >= cfg_.max_num_seqs) {  
            kept.push_back(seq);
            continue;
        }

        bool yielded = false, stuck = false;
        while (!bm_.can_append(*seq)) {
            if (!allow_preempt) { stuck = true; break; }
            Sequence* victim;
            if (!pending.empty()) { victim = pending.back(); pending.pop_back(); }
            else                  { victim = seq; yielded = true; }
            preempt(victim);
            preempted_now = true;
            if (yielded) break;
        }
        if (stuck) { blocked = true; kept.push_back(seq); continue; }
        if (yielded) continue;

        bm_.may_append(*seq);
        push(seq, want);
        kept.push_back(seq);
    }
    running_ = std::move(kept);
    while (!preempted_now && !blocked && !waiting_.empty() && budget > 0 && running_.size() < cfg_.max_num_seqs) {
        Sequence* seq = waiting_.front();
        if (!bm_.can_allocate(*seq)) break;
        waiting_.pop_front();
        bm_.allocate(*seq);                  
        seq->status = Status::RUNNING;
        running_.push_back(seq);
        push(seq, std::min({seq->num_uncached_tokens(), budget, chunk_cap}));
    }

    out.blocked = blocked;
    out.all_single_token =
        !out.seqs.empty() &&
        std::all_of(out.seqs.begin(), out.seqs.end(), [](const Sequence* s) { return s->num_scheduled_tokens == 1; });
}

void Scheduler::on_launched(ScheduledBatch& batch) {
    batch.sample_pos.assign(batch.seqs.size(), -1);
    for (size_t i = 0; i < batch.seqs.size(); ++i) {
        Sequence* seq = batch.seqs[i];
        const bool samples = samples_this_step(*seq);
        seq->num_cached_tokens += seq->num_scheduled_tokens;
        seq->num_scheduled_tokens = 0;
        ++seq->num_inflight;
        if (samples) {
            batch.sample_pos[i] = static_cast<int64_t>(seq->token_ids.size());
            seq->append_token(kPendingToken);  
        }
    }
}

size_t Scheduler::postprocess(const ScheduledBatch& batch, std::span<const int32_t> sampled) {
    size_t accepted = 0;
    for (size_t i = 0; i < batch.seqs.size(); ++i) {
        Sequence* seq = batch.seqs[i];
        assert(seq->num_inflight > 0);
        --seq->num_inflight;
        if (seq->is_finished()) continue;    

        const int64_t pos = batch.sample_pos[i];
        if (pos < 0) {                       
            bm_.commit(*seq);                 
            continue;
        }

        const int32_t tok = sampled[i];
        seq->token_ids[static_cast<size_t>(pos)] = tok;
        ++accepted;

        const bool hit_eos = !seq->params.ignore_eos && is_eos(tok);
        const bool hit_len = static_cast<int64_t>(pos + 1 - seq->num_prompt_tokens) >= seq->params.max_tokens;
        if (hit_eos || hit_len) {
            seq->token_ids.resize(static_cast<size_t>(pos) + 1);
            seq->num_cached_tokens = static_cast<uint32_t>(std::min<size_t>(seq->num_cached_tokens, seq->token_ids.size()));
            bm_.commit(*seq);
            seq->status = Status::FINISHED;
            bm_.deallocate(*seq);
        } else {
            bm_.commit(*seq);
        }
    }
    std::erase_if(running_, [](const Sequence* s) { return s->is_finished(); });
    return accepted;
}