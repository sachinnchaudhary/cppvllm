#include "config.h"
#include "engine/llm_engine.h"
#include "utils/tokenizer.h"

#include <iostream>
#include <string>
#include <vector>

static std::string apply_chat_template(const std::string& user) {
    return "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: example <model_dir>\n"; return 2; }

    const int tp_size = 1;   

    const std::string path = argv[1];

    Tokenizer tokenizer = Tokenizer::load(path);

    EngineConfig cfg;
    cfg.model_path    = path;
    cfg.tensor_parallel_size = tp_size;
    cfg.enforce_eager = false;
    LLMEngine llm(cfg);

    SamplingParams sampling_params;
    sampling_params.temperature = 0.6f;
    sampling_params.max_tokens  = 512;   

    std::vector<std::string> prompts = {
        "introduce yourself",
        "explain deep learning",
    };
    for (auto& p : prompts) p = apply_chat_template(p);

    std::vector<std::vector<int32_t>> prompt_ids;
    for (const auto& p : prompts) prompt_ids.push_back(tokenizer.encode(p, /*allow_special=*/true));

    auto outputs = llm.generate(prompt_ids, sampling_params);

    for (size_t i = 0; i < prompts.size(); ++i) {
        std::cout << "\n\n";
        std::cout << "Prompt: \"" << prompts[i] << "\"\n";
        std::cout << "Completion: \"" << tokenizer.decode(outputs[i].token_ids, /*skip_special=*/true) << "\"\n";
    }
    return 0;
}