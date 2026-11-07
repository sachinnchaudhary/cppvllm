# cppvllm

cppvllm is a light implementation of vLLM (and its ideas) written in ~3000 lines of C++/CUDA. Inspired by Nano-vLLM, done in Python.

It utilizes major techniques which vLLM uses, like paged attention, flattened padding, prefix caching and chunked prefill.
It has optimizations done on top of it, such as CUDA graph, overlap scheduling, custom FP8 paged attention kernel and fused operation kernel, integration of vllm-flash-attention for prefill, and multi-GPU support with TP.

cppvllm beats all mainstream inference engines for the Qwen3-0.6B model with an achieved throughput of 5156.19 tok/s. It beats other engines even with BF16 (currently it supports only FP8) with a throughput of 3342.82 tok/s.

cppvllm(FP8) is 1.69x faster than Nano-Vllm and 1.61x faster than Vllm. 

## Benchmark

- Model: Qwen3-0.6B
- GPU: single A10G GPU
- Requests: 256 sequences
- Input length: random_sample(100–1024 tokens)
- Output length: random_sample(100–1024 tokens)
- File: `bench.cpp`

**BENCHMARK COMPARISON (Qwen/Qwen3-0.6B | 256 Seqs | GPU: A10G)**

| Engine                 | Output Tokens | Time (s) | Throughput (tok/s) |
| ---------------------- | ------------: | -------: | -----------------: |
| cppvllm (C++, FP8 KV)  |       141,979 |    27.54 |            5156.19 |
| cppvllm (C++, BF16 KV) |       141,979 |    42.47 |            3342.82 |
| minisglang             |       133,966 |    40.63 |            3297.38 |
| vllm                   |       133,966 |    41.73 |            3210.26 |
| nanovllm               |       133,966 |    43.97 |            3046.85 |
| sglang                 |       133,966 |    51.49 |            2601.88 |

## Usage

Usage is in `example.cpp` with a clean interface:

```cpp
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
```



## Commands to run

### Installation

Requirements: a Linux machine with an NVIDIA GPU (compute capability 8.0, 8.6 or 8.9: A100, A10G, RTX 30/40 series), the CUDA toolkit (`nvcc`), and a C++20 compiler.

```bash
# system packages
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build git pkg-config \
    nlohmann-json3-dev libpcre2-dev

# libtorch 2.5.1 and NCCL come from the PyTorch pip wheel
pip install torch==2.5.1 "huggingface_hub[cli]"

# get the code and the model
git clone <repo-url> cppvllm
cd cppvllm
huggingface-cli download Qwen/Qwen3-0.6B --local-dir models/qwen3-0.6b

# build (the first build downloads vllm-flash-attention and compiles it, which takes a few minutes)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build -j
```

### For bench.cpp

```bash
./build/bench models/qwen3-0.6b
```

### For example.cpp

```bash
./build/example models/qwen3-0.6b
```

