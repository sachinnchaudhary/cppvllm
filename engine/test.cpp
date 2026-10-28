#include "scheduler.h"
#include <cstdio>
#include <numeric>
#include <stdexcept>

static std::vector<int32_t> make(int n, int base = 0) {
    std::vector<int32_t> v(n);
    std::iota(v.begin(), v.end(), base);
    return v;
}

// ---- a fake "model": no GPU, it just returns a token wherever one is required ----
template <class F>
static void complete(Scheduler& s, const ScheduledBatch& b, F next_token) {
    std::vector<int32_t> sampled(b.seqs.size(), -1);
    for (size_t i = 0; i < b.seqs.size(); ++i)
        if (samples_this_step(*b.seqs[i])) sampled[i] = next_token(*b.seqs[i]);
    s.postprocess(b, sampled);
}

static int32_t any_token(const Sequence&) { return 100; }

// Run until everything finishes. Fails loudly instead of hanging.
static size_t run_all(Scheduler& s) {
    size_t steps = 0;
    while (!s.is_finished()) {
        const ScheduledBatch& b = s.schedule();
        assert(!b.empty() && "scheduler is stuck: nothing runnable but work remains");
        complete(s, b, any_token);
        assert(++steps < 100000);
    }
    return steps;
}

int main() {
    // ---- 1. one sequence, start to finish -------------------------------------
    {
        Scheduler sc({}, 16);
        Sequence a(0, make(40), SamplingParams{1.0f, 5, false});
        sc.add_request(&a);

        const ScheduledBatch& b1 = sc.schedule();
        assert(b1.seqs.size() == 1 && b1.total_tokens == 40 && !b1.all_single_token);
        complete(sc, b1, any_token);
        assert(a.num_completion_tokens() == 1 && a.num_cached_tokens == 40);

        for (int i = 0; i < 4; ++i) {
            const ScheduledBatch& b = sc.schedule();
            assert(b.total_tokens == 1 && b.all_single_token);   // decode-shaped -> graph-able
            complete(sc, b, any_token);
        }
        assert(a.is_finished() && a.num_completion_tokens() == 5);
        assert(sc.is_finished());
        assert(sc.block_manager().num_free_blocks() == 16);      // all blocks returned
    }

    // ---- 2. chunked prefill: 100-token prompt, budget 40 -----------------------
    {
        SchedulerConfig cfg; cfg.max_num_batched_tokens = 40;
        Scheduler sc(cfg, 16);
        Sequence a(0, make(100), SamplingParams{1.0f, 3, false});
        sc.add_request(&a);

        const ScheduledBatch& b1 = sc.schedule();
        assert(b1.total_tokens == 40 && !samples_this_step(a));
        complete(sc, b1, any_token);
        assert(a.num_cached_tokens == 40 && a.num_completion_tokens() == 0);

        const ScheduledBatch& b2 = sc.schedule();
        assert(b2.total_tokens == 40);
        complete(sc, b2, any_token);
        assert(a.num_cached_tokens == 80 && a.num_completion_tokens() == 0);

        const ScheduledBatch& b3 = sc.schedule();
        assert(b3.total_tokens == 20 && samples_this_step(a));   // last chunk -> first token
        complete(sc, b3, any_token);
        assert(a.num_cached_tokens == 100 && a.num_completion_tokens() == 1);
    }

    // ---- 3. mixed step: decode token + prefill chunk in the same batch ---------
    {
        SchedulerConfig cfg; cfg.max_num_batched_tokens = 40;
        Scheduler sc(cfg, 32);
        Sequence a(0, make(16), SamplingParams{1.0f, 50, false});
        sc.add_request(&a);
        complete(sc, sc.schedule(), any_token);                  // A prefilled, now decoding

        Sequence b(1, make(100, 500), SamplingParams{1.0f, 5, false});
        sc.add_request(&b);
        const ScheduledBatch& m = sc.schedule();
        assert(m.seqs.size() == 2);
        assert(m.seqs[0] == &a && a.num_scheduled_tokens == 1);   // decode first
        assert(m.seqs[1] == &b && b.num_scheduled_tokens == 39);  // gets the leftover budget
        assert(m.total_tokens == 40 && !m.all_single_token);
    }

    // ---- 4. prefix cache across requests ----------------------------------------
    {
        Scheduler sc({}, 16);
        Sequence a(0, make(40), SamplingParams{1.0f, 10, false});
        sc.add_request(&a);
        complete(sc, sc.schedule(), any_token);                  // A's blocks 0,1 now published

        Sequence b(1, make(40), SamplingParams{1.0f, 10, false});
        sc.add_request(&b);
        const ScheduledBatch& m = sc.schedule();
        assert(b.num_cached_tokens == 32);                       // 2 blocks reused
        assert(b.num_scheduled_tokens == 8);                     // only the tail is computed
        assert(m.total_tokens == 1 + 8);
    }

    // ---- 5. preemption: pool too small for both sequences to finish together ---
    {
        Scheduler sc({}, 4);                                     // 4 blocks = 64 tokens total
        Sequence a(0, make(16),      SamplingParams{1.0f, 40, false});
        Sequence b(1, make(16, 500), SamplingParams{1.0f, 40, false});
        sc.add_request(&a);
        sc.add_request(&b);
        run_all(sc);
        assert(sc.num_preemptions() >= 1);                       // somebody had to be evicted
        assert(a.is_finished() && b.is_finished());
        assert(a.num_completion_tokens() == 40 && b.num_completion_tokens() == 40);
        assert(sc.block_manager().num_free_blocks() == 4);       // nothing leaked
    }

    // ---- 6. EOS stops early, ignore_eos does not ---------------------------------
    {
        SchedulerConfig cfg; cfg.eos_token_id = 2;
        Scheduler sc(cfg, 16);
        auto eos_on_third = [](const Sequence& s) { return s.num_completion_tokens() == 2 ? 2 : 100; };

        Sequence a(0, make(20), SamplingParams{1.0f, 50, false});
        Sequence b(1, make(20, 900), SamplingParams{1.0f, 6, true});   // ignore_eos
        sc.add_request(&a);
        sc.add_request(&b);
        while (!sc.is_finished()) {
            const ScheduledBatch& m = sc.schedule();
            assert(!m.empty());
            complete(sc, m, eos_on_third);
        }
        assert(a.num_completion_tokens() == 3);                  // stopped at EOS
        assert(b.num_completion_tokens() == 6);                  // ran to max_tokens
    }

    // ---- 7. max_num_seqs caps the running set; FCFS preserved --------------------
    {
        SchedulerConfig cfg; cfg.max_num_seqs = 2;
        Scheduler sc(cfg, 32);
        Sequence a(0, make(16), SamplingParams{1.0f, 4, false});
        Sequence b(1, make(16, 100), SamplingParams{1.0f, 4, false});
        Sequence c(2, make(16, 200), SamplingParams{1.0f, 4, false});
        sc.add_request(&a); sc.add_request(&b); sc.add_request(&c);
        (void)sc.schedule();
        assert(sc.num_running() == 2 && sc.num_waiting() == 1);
        run_all(sc);
        assert(a.is_finished() && b.is_finished() && c.is_finished());
    }

    

    std::puts("all scheduler tests passed");
}