// Scratch test for model_runner + llm_engine on a REAL model.
//     ./build/scratch <model_dir> [ref_file]
// ref_file lines: "p1 p2 ... | o1 o2 ..."  (prompt ids | HF greedy output ids)
//
// Checks:
//   1. run A (generous budget) == HuggingFace greedy output        (model math is right)
//   2. run B (8-token budget, 4-token chunks, 4 KV blocks) == run A (scheduling doesn't change results)
//   3. duplicate prompts give identical outputs                    (prefix cache is safe)
#include "config.h"
#include "engine/llm_engine.h"

#include <fstream>
#include <iostream>
#include <sstream>

struct Case { std::vector<int32_t> prompt, expect; };

static std::vector<Case> load_ref(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open ref file: " + path);
    std::vector<Case> cases;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        const size_t bar = line.find('|');
        if (bar == std::string::npos) continue;
        Case c;
        std::istringstream a(line.substr(0, bar)), b(line.substr(bar + 1));
        for (int32_t t; a >> t;) c.prompt.push_back(t);
        for (int32_t t; b >> t;) c.expect.push_back(t);
        cases.push_back(std::move(c));
    }
    return cases;
}

static std::vector<GenerationOutput> run(EngineConfig cfg, const std::vector<std::vector<int32_t>>& prompts,
                                         int max_tokens, EngineStats* st) {
    const auto dev = torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);
    LLMEngine eng(cfg, dev);
    SamplingParams sp;
    sp.temperature = 0;  sp.max_tokens = max_tokens;  sp.ignore_eos = true;
    auto out = eng.generate(prompts, sp);
    *st = eng.stats();
    return out;
}

static void report_diff(const char* what, size_t i, const std::vector<int32_t>& got, const std::vector<int32_t>& want) {
    size_t k = 0;
    while (k < got.size() && k < want.size() && got[k] == want[k]) ++k;
    std::cout << "  " << what << ": prompt " << i << " diverges at token " << k << " (got "
              << (k < got.size() ? std::to_string(got[k]) : "-") << ", want "
              << (k < want.size() ? std::to_string(want[k]) : "-") << ")\n";
}

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: scratch <model_dir> <ref_file>\n"; return 2; }

    const auto cases = load_ref(argv[2]);
    if (cases.empty()) { std::cerr << "ref file has no cases\n"; return 2; }
    std::vector<std::vector<int32_t>> prompts;
    for (const auto& c : cases) prompts.push_back(c.prompt);
    const int max_tokens = static_cast<int>(cases[0].expect.size());

    EngineConfig a;
    a.model_path = argv[1];
    a.dtype = DType::FP32;            // fp32 so schedule changes can't flip near-tied greedy choices
    a.num_kv_blocks = 256;
    a.max_model_len = 256;
    EngineConfig b = a;
    b.max_num_batched_tokens = 8;  b.max_num_seqs = 4;  b.max_chunk_tokens = 4;  b.num_kv_blocks = 4;

    EngineStats sa, sb;
    auto oa = run(a, prompts, max_tokens, &sa);
    auto ob = run(b, prompts, max_tokens, &sb);

    bool ok = oa.size() == prompts.size() && ob.size() == prompts.size();
    bool ref_ok = ok, sched_ok = ok, dup_ok = ok;
    for (size_t i = 0; ok && i < prompts.size(); ++i) {
        if (oa[i].token_ids != cases[i].expect) { ref_ok = false; report_diff("A vs HF", i, oa[i].token_ids, cases[i].expect); }
        if (ob[i].token_ids != oa[i].token_ids) { sched_ok = false; report_diff("B vs A", i, ob[i].token_ids, oa[i].token_ids); }
    }
    if (ok) {
        for (size_t i = 0; i < prompts.size(); ++i)
            for (size_t j = i + 1; j < prompts.size(); ++j)
                if (prompts[i] == prompts[j] && oa[i].token_ids != oa[j].token_ids) dup_ok = false;
    }

    std::cout << "run A: " << sa.steps << " steps, " << sa.preemptions << " preemptions\n"
              << "run B: " << sb.steps << " steps, " << sb.preemptions << " preemptions\n"
              << (ref_ok   ? "PASS" : "FAIL") << ": matches HuggingFace greedy output\n"
              << (sched_ok ? "PASS" : "FAIL") << ": same tokens under both schedules\n"
              << (dup_ok   ? "PASS" : "FAIL") << ": duplicate prompts give identical outputs\n";
    if (sb.preemptions == 0) std::cout << "(note: run B never preempted; shrink num_kv_blocks to exercise it)\n";
    return (ref_ok && sched_ok && dup_ok && ok) ? 0 : 1;
}