#include "config.h"
#include <cstdio>

template <class F>
static bool throws(F f) {
    try { f(); } catch (const std::invalid_argument&) { return true; }
    return false;
}

int main() {
    // ---- Llama 3.2 1B numbers ----------------------------------------------------
    ModelConfig m = ModelConfig::llama32_1b();
              // int(1.5*5461) = 8191 -> rounded up to 8192
    // ---- the tiny config from your Python __main__ ---------------------------------
    ModelConfig t = ModelConfig::tiny();
    assert(t.head_dim() == 32);
    assert(t.intermediate_size() == 512);           // int(1.5*341) = 511 -> rounded up to 512

    // ---- KV cache sizing ------------------------------------------------------------
    assert(kv_bytes_per_token(m, DType::BF16) == 32 * 1024);
    assert(kv_bytes_per_block(m, DType::BF16) == 512 * 1024);   // 16 tokens
    assert(kv_bytes_per_block(m, DType::FP32) == 1024 * 1024);
    assert(num_kv_blocks_for_budget(size_t{1} << 30, kv_bytes_per_block(m, DType::BF16)) == 2048); // 1 GiB

    // ---- invalid model configs are rejected -------------------------------------------
   

    // ---- EngineConfig ---------------------------------------------------------------------
    EngineConfig e;
   
    assert(kBlockSize == 16);
    assert(e.max_blocks_per_seq() == 256);                       // 4096 / 16

                      // exactly enough

    // ---- scheduler slice -------------------------------------------------------------------
    {
        EngineConfig b = e; b.max_chunk_tokens = 512;
        SchedulerConfig s = b.scheduler_config();
        assert(s.max_num_batched_tokens == 8192 && s.max_num_seqs == 256);
        assert(s.max_chunk_tokens == 512 && s.eos_token_id == 128001);
    }

    // ---- CUDA graph batch sizes ----------------------------------------------------------------
    {
        auto g = e.graph_batch_sizes();                          // max_num_seqs = 256
        assert(g.size() == 20 && g.front() == 1 && g.back() == 256);
        assert((g[0] == 1 && g[1] == 2 && g[2] == 4 && g[3] == 8 && g[4] == 16 && g[5] == 32));

        EngineConfig b = e; b.max_num_seqs = 100;
        auto g100 = b.graph_batch_sizes();
        assert(g100.back() == 100 && g100[g100.size() - 2] == 96);

        b.max_num_seqs = 3;
        auto g3 = b.graph_batch_sizes();
        assert((g3 == std::vector<int32_t>{1, 2, 3}));

        b.max_num_seqs = 1000;                                   // capped at 512
        assert(b.graph_batch_sizes().back() == 512);

        assert(pick_graph_batch_size(g, 1) == 1);
        assert(pick_graph_batch_size(g, 5) == 8);                // padded up to the next captured size
        assert(pick_graph_batch_size(g, 9) == 16);
        assert(pick_graph_batch_size(g, 256) == 256);
        assert(pick_graph_batch_size(g, 257) == -1);             // too big -> eager
    }

    std::puts("all config tests passed");
}