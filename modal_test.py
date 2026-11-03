"""Build cppvllm on a Modal GPU box and run example.cpp and bench.cpp on Qwen3-0.6B.

Put this file in the repo ROOT (next to CMakeLists.txt), then:

    pip install modal && modal setup        # once
    modal run modal_test.py                 # example, then bench
    modal run modal_test.py --only example  # just the chat demo
    modal run modal_test.py --only bench    # just the throughput benchmark

Steps (one container: CUDA 12.4 + PyTorch 2.5.1 cu124 for libtorch):
  1. download Qwen3-0.6B + tokenizer.json (cached in a Modal Volume after the first run)
  2. cmake + build targets `example` and `bench`
  3. run them; any non-zero exit makes `modal run` fail
"""
import subprocess
import time
from pathlib import Path

import modal

GPU = "A10G"            # if you change it, update ARCH (T4 = 7.5, A10G = 8.6, L4 = 8.9, A100 = 8.0, H100 = 9.0)
ARCH = "8.6"
MODEL_REPO = "Qwen/Qwen3-0.6B"                   # the chat (post-trained) model, ungated; example.cpp uses its ChatML template
MODEL_DIR = "/models/qwen3-0.6b"
ROOT = Path(__file__).parent.resolve()

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential", "cmake", "ninja-build", "nlohmann-json3-dev", "git",
                 "pkg-config", "libpcre2-dev")             # PCRE2 for the tokenizer
    .pip_install("numpy")
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("huggingface_hub")
    .env({"TORCH_CUDA_ARCH_LIST": ARCH, "CUDA_HOME": "/usr/local/cuda"})
    .add_local_dir(                                   # must stay LAST in the chain
        ROOT, remote_path="/src",
        ignore=["build", "**/build", ".git", "**/.git", "**/__pycache__", "**/*.safetensors", "**/*.pth", "**/*.pt", "**/.venv"],
    )
)
models = modal.Volume.from_name("cppvllm-models", create_if_missing=True)
app = modal.App("cppvllm-run", image=image)


def sh(cmd: str, check: bool = True) -> int:
    print(f"\n$ {cmd}", flush=True)
    rc = subprocess.run(cmd, shell=True, executable="/bin/bash").returncode
    if check and rc != 0:
        raise RuntimeError(f"command failed (exit {rc}): {cmd}")
    return rc


@app.function(gpu=GPU, cpu=8, memory=32768, timeout=3600, volumes={"/models": models})
def run_all(only: str = "all") -> dict:
    results: dict[str, bool] = {}
    t0 = time.time()
    want_example = only in ("all", "example")
    want_bench = only in ("all", "bench")

    sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    if not Path("/src/CMakeLists.txt").exists():
        raise RuntimeError("no CMakeLists.txt in the uploaded folder: put modal_test.py in the repo root")

    # ---- 1. model + tokenizer ---------------------------------------------------------------------
    if not (Path(MODEL_DIR, "config.json").exists() and Path(MODEL_DIR, "tokenizer.json").exists()):
        sh(f"python -c \"from huggingface_hub import snapshot_download; "
           f"snapshot_download('{MODEL_REPO}', local_dir='{MODEL_DIR}', "
           f"allow_patterns=['config.json', '*.safetensors', 'tokenizer.json', 'tokenizer_config.json'])\"")
        models.commit()
    sh(f"ls -la {MODEL_DIR}")

    # ---- 2. build ---------------------------------------------------------------------------------
    targets = " ".join(t for t, w in (("example", want_example), ("bench", want_bench)) if w)
    sh('PREFIX=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") && '
       'cmake -S /src -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PREFIX" '
       '-DCMAKE_CXX_FLAGS=-I/usr/local/cuda/include')
    sh(f"cmake --build /build --target {targets} -j8")
    print(f"\n[build done in {time.time() - t0:.0f}s]", flush=True)

    # ---- 3. run -----------------------------------------------------------------------------------
    if want_example:
        print("\n===== example: chat demo =====", flush=True)
        results["example"] = sh(f"/build/example {MODEL_DIR}", check=False) == 0
    if want_bench:
        print("\n===== bench: 256 random requests =====", flush=True)
        t1 = time.time()
        results["bench"] = sh(f"/build/bench {MODEL_DIR}", check=False) == 0
        print(f"[bench wall time incl. load: {time.time() - t1:.0f}s]", flush=True)

    print("\n================ SUMMARY ================")
    for k, v in results.items():
        print(("OK    " if v else "FAIL  ") + k)
    print(f"total {time.time() - t0:.0f}s")
    if not all(results.values()):
        raise RuntimeError("some programs failed (see the log above)")
    return results


@app.local_entrypoint()
def main(only: str = "all"):
    run_all.remote(only)