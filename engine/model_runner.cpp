#include "model_runner.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#if __has_include(<cuda_runtime_api.h>)
#include <cuda_runtime_api.h>
#define CPPVLLM_HAS_CUDART 1
#endif

namespace {

int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

torch::ScalarType pick_dtype(const EngineConfig& cfg, torch::Device device) {
    if (!device.is_cuda()) return torch::kFloat;      // CPU path stays fp32
    switch (cfg.dtype) {
        case DType::BF16: return torch::kBFloat16;
        case DType::FP16: return torch::kHalf;
        case DType::FP32: return torch::kFloat;
    }
    return torch::kBFloat16;
}

// logits: [R, V] fp32, temps: [R]. temperature <= 0 means greedy.
torch::Tensor sample(const torch::Tensor& logits, const torch::Tensor& temps) {
    torch::Tensor greedy = logits.argmax(-1);
    torch::Tensor probs = torch::softmax(logits / temps.clamp_min(1e-5).unsqueeze(1), -1);
    // Gumbel-free exponential-race sampling: argmax(p / Exp(1)) draws from p
    torch::Tensor clocks = torch::empty_like(probs).exponential_(1.0).clamp_min_(1e-10);
    torch::Tensor sampled = (probs / clocks).argmax(-1);
    return torch::where(temps <= 0, greedy, sampled);
}

}  // namespace

ModelRunner::ModelRunner(const EngineConfig& cfg, const ModelConfig& mcfg, torch::Device device)
    : cfg_(cfg),
      mcfg_(mcfg),
      device_(device),
      dtype_(pick_dtype(cfg, device)),
      model_(mcfg),
      max_blocks_per_seq_(ceil_div(std::min<int64_t>(cfg.max_model_len, mcfg.max_seq_len), kBlockSize)),
      tokens_(static_cast<int64_t>(cfg.max_num_batched_tokens), device),
      positions_(static_cast<int64_t>(cfg.max_num_batched_tokens), device),
      cu_seqlens_(static_cast<int64_t>(cfg.max_num_seqs) + 1, device),
      kv_lens_(static_cast<int64_t>(cfg.max_num_seqs), device),
      tables_(static_cast<int64_t>(cfg.max_num_seqs) * max_blocks_per_seq_, device, int64_t{-1}),
      rows_(static_cast<int64_t>(cfg.max_num_seqs), device),
      temps_(static_cast<int64_t>(cfg.max_num_seqs), device, 1.0f) {
    cfg_.validate();
    torch::manual_seed(cfg_.seed);

    load_weights();
    num_kv_blocks_ = size_kv_cache();
    bind_cache(num_kv_blocks_);

    out_.reserve(cfg_.max_num_seqs);
    row_owner_.reserve(cfg_.max_num_seqs);
}

void ModelRunner::load_weights() {
    WeightMap weights = ::load_weights(cfg_.model_path, device_, dtype_);   // free function from utils/tensor.h
    model_.load_weights(weights);
}

void ModelRunner::bind_cache(int64_t num_blocks) {
    const std::vector<int64_t> shape{mcfg_.n_layers, num_blocks, kBlockSize, mcfg_.n_kv_heads, mcfg_.head_dim()};
    const auto opts = torch::TensorOptions().dtype(dtype_).device(device_);
    model_.bind_kv_cache(torch::zeros(shape, opts), torch::zeros(shape, opts));
}

// One worst-case forward so activation memory is reserved before we measure what is left for the KV cache.
void ModelRunner::warmup() {
    const int64_t n = std::min<int64_t>(static_cast<int64_t>(cfg_.max_num_batched_tokens), max_blocks_per_seq_ * kBlockSize);
    const int64_t nb = ceil_div(n, kBlockSize);
    bind_cache(nb);

    const auto i64 = torch::TensorOptions().dtype(torch::kLong).device(device_);
    FlatBatch fb;
    fb.tokens = torch::zeros({n}, i64);
    fb.positions = torch::arange(n, i64);
    fb.cu_seqlens_q = torch::tensor(std::vector<int64_t>{0, n}, i64);
    fb.kv_lens = torch::tensor(std::vector<int64_t>{n}, i64);
    fb.block_tables = torch::arange(nb, i64).view({1, nb});
    fb.max_q = fb.max_kv = n;
    (void)model_.forward(fb);
    if (device_.is_cuda()) torch::cuda::synchronize();
    bind_cache(1);
}

int32_t ModelRunner::size_kv_cache() {
    if (cfg_.num_kv_blocks > 0) return cfg_.num_kv_blocks;
    if (!device_.is_cuda())
        throw std::runtime_error("EngineConfig::num_kv_blocks must be set > 0 for CPU execution");

#ifdef CPPVLLM_HAS_CUDART
    warmup();
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) throw std::runtime_error("cudaMemGetInfo failed");
    const double used = static_cast<double>(total_b - free_b);
    const double budget = static_cast<double>(total_b) * cfg_.gpu_memory_utilization - used;
    const double per_block = static_cast<double>(kv_bytes_per_block(mcfg_, c10::elementSize(dtype_)));
    const int64_t n = static_cast<int64_t>(budget / per_block);
    if (n <= 0)
        throw std::runtime_error("no GPU memory left for the KV cache: lower max_num_batched_tokens or raise gpu_memory_utilization");
    std::cerr << "[cppvllm] KV cache: " << n << " blocks (" << n * kBlockSize << " tokens, "
              << static_cast<double>(n) * per_block / (1ull << 30) << " GiB)\n";
    return static_cast<int32_t>(std::min<int64_t>(n, INT32_MAX));
#else
    throw std::runtime_error("built without cuda_runtime_api.h: set EngineConfig::num_kv_blocks explicitly");
#endif
}

const std::vector<int32_t>& ModelRunner::run(const ScheduledBatch& batch) {
    const size_t n = batch.seqs.size();
    out_.assign(n, -1);

    // The naive attention pads every sequence's queries to max_q. If padding would waste more than 4x,
    // run decode tokens and prompt chunks as two forwards. (A fused paged-attention kernel removes this later.)
    size_t max_q = 0;
    for (const Sequence* s : batch.seqs) max_q = std::max<size_t>(max_q, s->num_scheduled_tokens);
    if (n * max_q <= 4 * batch.total_tokens) {
        std::vector<size_t> all(n);
        for (size_t i = 0; i < n; ++i) all[i] = i;
        run_group(batch, all);
    } else {
        std::vector<size_t> decode, prefill;
        for (size_t i = 0; i < n; ++i) (batch.seqs[i]->num_scheduled_tokens == 1 ? decode : prefill).push_back(i);
        if (!decode.empty()) run_group(batch, decode);
        if (!prefill.empty()) run_group(batch, prefill);
    }
    return out_;
}

void ModelRunner::run_group(const ScheduledBatch& batch, const std::vector<size_t>& idx) {
    const int64_t S = static_cast<int64_t>(idx.size());
    int64_t T = 0, W = 0, max_q = 0, max_kv = 0;
    for (size_t i : idx) {
        const Sequence& s = *batch.seqs[i];
        const int64_t q = s.num_scheduled_tokens, kv = s.num_cached_tokens + q;
        T += q;
        W = std::max(W, ceil_div(kv, kBlockSize));
        max_q = std::max(max_q, q);
        max_kv = std::max(max_kv, kv);
    }
    TORCH_CHECK(T <= tokens_.capacity(), "step has ", T, " tokens, more than max_num_batched_tokens");
    TORCH_CHECK(W <= max_blocks_per_seq_, "sequence longer than max_model_len / max_seq_len");

    auto tok = tokens_.host(), pos = positions_.host(), cu = cu_seqlens_.host(), kvl = kv_lens_.host(),
         tab = tables_.host(), rows = rows_.host();
    auto temp = temps_.host();
    int64_t t = 0, R = 0;
    cu[0] = 0;
    row_owner_.clear();
    for (int64_t j = 0; j < S; ++j) {
        const size_t bi = idx[j];
        const Sequence& s = *batch.seqs[bi];
        const int64_t c = s.num_cached_tokens, q = s.num_scheduled_tokens, kv = c + q;
        for (int64_t k = 0; k < q; ++k, ++t) { tok[t] = s.token_ids[c + k]; pos[t] = c + k; }
        cu[j + 1] = t;
        kvl[j] = kv;
        const int64_t nblk = ceil_div(kv, kBlockSize);
        for (int64_t b = 0; b < W; ++b) tab[j * W + b] = b < nblk ? s.block_table[b] : -1;
        if (samples_this_step(s)) {
            rows[R] = j;                         // index of the sequence inside this group
            temp[R] = s.params.temperature;
            row_owner_.push_back(bi);
            ++R;
        }
    }

    FlatBatch fb;
    fb.tokens = tokens_.upload(T);
    fb.positions = positions_.upload(T);
    fb.cu_seqlens_q = cu_seqlens_.upload(S + 1);
    fb.kv_lens = kv_lens_.upload(S);
    fb.block_tables = tables_.upload(S * W).view({S, W});
    fb.max_q = max_q;
    fb.max_kv = max_kv;

    torch::Tensor logits = model_.forward(fb, rows_.upload(R));
    if (R == 0) {
        if (device_.is_cuda()) torch::cuda::synchronize();   // host buffers are reused next step
        return;
    }
    torch::Tensor next = sample(logits, temps_.upload(R)).to(torch::kCPU);   // blocking copy = sync
    const int64_t* p = next.data_ptr<int64_t>();
    for (int64_t r = 0; r < R; ++r) out_[row_owner_[r]] = static_cast<int32_t>(p[r]);
}