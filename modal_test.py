"""Build cppvllm on a Modal GPU box and test it against the real Llama 3.2 1B.

Put this file in the repo ROOT (next to CMakeLists.txt), then:

    pip install modal && modal setup        # once
    modal run modal_test.py

Steps (one container: CUDA 12.4 + PyTorch 2.5.1 cu124 for libtorch):
  1. download Llama 3.2 1B (cached in a Modal Volume after the first run)
  2. HuggingFace greedy reference: 4 fixed prompts -> expected tokens
  3. cmake + build (target `scratch` = your test.cpp)
  4. test.cpp (fp32, GPU): engine == HF, tiny-budget schedule == generous schedule, prefix cache safe
  5. CUDA smoke: bf16, KV cache auto-sized from free GPU memory (warmup, cudaMemGetInfo, pinned buffers)
Any failing step makes `modal run` exit non-zero.
"""
import subprocess
import time
from pathlib import Path

import modal

GPU = "T4"            # if you change it, update ARCH (T4 = 7.5, A10G = 8.6, L4 = 8.9, A100 = 8.0, H100 = 9.0)
ARCH = "7.5"
MODEL_REPO = "unsloth/Llama-3.2-1B"          # ungated mirror of meta-llama/Llama-3.2-1B
MODEL_DIR = "/models/llama-3.2-1b"
ROOT = Path(__file__).parent.resolve()

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential", "cmake", "ninja-build", "nlohmann-json3-dev", "git")
    .pip_install("numpy")
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("transformers==4.46.3", "huggingface_hub", "accelerate")
    .env({"TORCH_CUDA_ARCH_LIST": ARCH, "CUDA_HOME": "/usr/local/cuda"})
    .add_local_dir(                                   # must stay LAST in the chain
        ROOT, remote_path="/src",
        ignore=["build", "**/build", ".git", "**/.git", "**/__pycache__", "**/*.safetensors", "**/*.pth", "**/*.pt", "**/.venv"],
    )
)
models = modal.Volume.from_name("cppvllm-models", create_if_missing=True)
app = modal.App("cppvllm-test", image=image)

# HuggingFace greedy reference. Same LCG as the old test, but token ids kept in [1000, 30000)
# so the prompts are ordinary text tokens, not Llama's reserved special tokens.
REF_SCRIPT = r'''
import sys, torch
from transformers import AutoModelForCausalLM
model_dir, out_path = sys.argv[1], sys.argv[2]
NEW = 12
x = 12345
def nxt():
    global x
    x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
    return 1000 + (x >> 8) % 29000
p = [[nxt() for _ in range(n)] for n in (20, 17, 33)]
p.append(list(p[0]))                                  # duplicate -> prefix-cache hit
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).cuda().eval()
with open(out_path, "w") as f:
    for ids in p:
        inp = torch.tensor([ids]).cuda()
        with torch.no_grad():
            out = model.generate(inp, do_sample=False, max_new_tokens=NEW, min_new_tokens=NEW,
                                 attention_mask=torch.ones_like(inp), pad_token_id=0)
        gen = out[0, len(ids):].tolist()
        f.write(" ".join(map(str, ids)) + " | " + " ".join(map(str, gen)) + "\n")
print("wrote", out_path)
'''

CUDA_SMOKE = r'''
#include "config.h"
#include "engine/llm_engine.h"
#include <iostream>
int main(int argc, char** argv) {
    if (argc < 2 || !torch::cuda::is_available()) { std::cerr << "need a model dir and a CUDA device\n"; return 2; }
    EngineConfig c;  c.model_path = argv[1];
    c.dtype = DType::BF16;
    c.max_model_len = 1024;
    c.max_num_batched_tokens = 2048;                 // keeps the worst-case warmup forward small
    c.gpu_memory_utilization = 0.5f;                 // num_kv_blocks stays 0: sized from free GPU memory
    LLMEngine eng(c);                                // default device = cuda:0
    std::vector<std::vector<int32_t>> prompts{{5000,5001,5002,5003,5004,5005,5006,5007}, {9000,9001,9002}};
    SamplingParams sp;  sp.temperature = 0.8f;  sp.max_tokens = 10;  sp.ignore_eos = true;
    auto out = eng.generate(prompts, sp);
    const int32_t V = eng.model_config().vocab_size;
    bool ok = out.size() == 2 && eng.num_kv_blocks() > 100;
    for (auto& o : out) {
        ok = ok && o.token_ids.size() == 10;
        std::cout << "request " << o.request_id << ":";
        for (int32_t t : o.token_ids) { ok = ok && t >= 0 && t < V; std::cout << ' ' << t; }
        std::cout << "\n";
    }
    std::cout << "kv blocks: " << eng.num_kv_blocks() << ", steps: " << eng.stats().steps << "\n"
              << (ok ? "PASS: CUDA smoke\n" : "FAIL: CUDA smoke\n");
    return ok ? 0 : 1;
}
'''


def sh(cmd: str, check: bool = True) -> int:
    print(f"\n$ {cmd}", flush=True)
    rc = subprocess.run(cmd, shell=True, executable="/bin/bash").returncode
    if check and rc != 0:
        raise RuntimeError(f"command failed (exit {rc}): {cmd}")
    return rc


@app.function(gpu=GPU, cpu=8, memory=32768, timeout=3600, volumes={"/models": models})
def run_all() -> dict:
    results: dict[str, bool] = {}
    t0 = time.time()

    sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    sh("python -c \"import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu ok:', torch.cuda.is_available())\"")
    if not Path("/src/CMakeLists.txt").exists():
        raise RuntimeError("no CMakeLists.txt in the uploaded folder: put modal_test.py in the repo root")

    # ---- 1. model ---------------------------------------------------------------------------------
    if not Path(MODEL_DIR, "config.json").exists():
        sh(f"python -c \"from huggingface_hub import snapshot_download; "
           f"snapshot_download('{MODEL_REPO}', local_dir='{MODEL_DIR}', "
           f"allow_patterns=['config.json', '*.safetensors'])\"")
        models.commit()
    sh(f"ls -la {MODEL_DIR}")

    # ---- 2. HF reference --------------------------------------------------------------------------
    Path("/tmp/ref.py").write_text(REF_SCRIPT)
    sh(f"python /tmp/ref.py {MODEL_DIR} /tmp/ref.txt")

    # ---- 3. build ---------------------------------------------------------------------------------
    sh("cp /src/test.cpp /tmp/user_test.cpp.keep")            # your test; we swap in the smoke test later
    sh('PREFIX=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") && '
       'cmake -S /src -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PREFIX" '
       '-DCMAKE_CXX_FLAGS=-I/usr/local/cuda/include')
    sh("cmake --build /build --target scratch -j8")
    print(f"\n[build done in {time.time() - t0:.0f}s]", flush=True)

    # ---- 4. your test on the real model ---------------------------------------------------------
    print("\n===== test.cpp: engine vs HuggingFace, schedule A vs B (GPU, fp32) =====", flush=True)
    results["test.cpp (real model)"] = sh(f"/build/scratch {MODEL_DIR} /tmp/ref.txt", check=False) == 0

    # ---- 5. CUDA smoke ----------------------------------------------------------------------------
    Path("/src/test.cpp").write_text(CUDA_SMOKE)
    sh("cmake --build /build --target scratch -j8")
    print("\n===== CUDA smoke test (GPU, bf16, auto-sized KV cache) =====", flush=True)
    results["cuda smoke"] = sh(f"/build/scratch {MODEL_DIR}", check=False) == 0
    sh("cp /tmp/user_test.cpp.keep /src/test.cpp")

    print("\n================ SUMMARY ================")
    for k, v in results.items():
        print(("PASS  " if v else "FAIL  ") + k)
    print(f"total {time.time() - t0:.0f}s")
    if not all(results.values()):
        raise RuntimeError("some tests failed (see the log above)")
    return results


@app.local_entrypoint()
def main():
    run_all.remote()