#include "config.h"
#include "engine/llm_engine.h"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: bench <model_dir>\n"; return 2; }

    const int tp_size = 1;   

    std::mt19937 gen(0);
    auto randint = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(gen); };
    const int num_seqs = 256;
    const int max_input_len = 1024;
    const int max_output_len = 1024;

    EngineConfig cfg;
    cfg.model_path = argv[1];
    cfg.tensor_parallel_size = tp_size;
    cfg.enforce_eager = false;
    cfg.max_num_batched_tokens = 4096;
    cfg.gpu_memory_utilization = 0.80f;     
    LLMEngine llm(cfg);

    std::vector<std::vector<int32_t>> prompt_token_ids;
    std::vector<SamplingParams> sampling_params;
    for (int i = 0; i < num_seqs; ++i) {
        std::vector<int32_t> p(randint(100, max_input_len));
        for (auto& t : p) t = randint(0, 10000);
        prompt_token_ids.push_back(std::move(p));
    }
    for (int i = 0; i < num_seqs; ++i) {
        SamplingParams sp;
        sp.temperature = 0.6f;
        sp.ignore_eos = true;
        sp.max_tokens = static_cast<int32_t>(randint(100, max_output_len));
        sampling_params.push_back(sp);
    }

    SamplingParams warm;
    warm.max_tokens = 16;
    (void)llm.generate({{1, 2, 3, 4, 5, 6, 7, 8}}, warm);
    const EngineStats before = llm.stats();

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < num_seqs; ++i) llm.add_request(prompt_token_ids[i], sampling_params[i]);
    while (!llm.is_finished()) llm.step();
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    (void)llm.take_finished();

    unsigned long total_tokens = 0;
    for (const auto& sp : sampling_params) total_tokens += static_cast<unsigned long>(sp.max_tokens);
    const double throughput = static_cast<double>(total_tokens) / t;
    std::printf("Total: %lutok, Time: %.2fs, Throughput: %.2ftok/s\n", total_tokens, t, throughput);

    const EngineStats& after = llm.stats();
    std::printf("steps: %zu, preemptions: %zu, KV blocks: %d\n", after.steps - before.steps,
                after.preemptions - before.preemptions, llm.num_kv_blocks());
    return 0;
}