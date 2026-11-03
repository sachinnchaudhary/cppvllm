#include "model_runner.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#if __has_include(<cuda_runtime_api.h>)
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDACachingAllocator.h>
#define CPPVLLM_HAS_CUDART 1
#endif

#if __has_include(<ATen/cuda/CUDAGraph.h>)
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#define CPPVLLM_HAS_CUDA_GRAPH 1
#else
namespace at::cuda { struct CUDAGraph {}; }   // CPU-only torch: keeps unique_ptr<CUDAGraph> destructible, never instantiated
#endif

namespace {

int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }
void release_cached_gpu_memory() {
#ifdef CPPVLLM_HAS_CUDART
    c10::cuda::CUDACachingAllocator::emptyCache();
#endif
}
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
// `logits` is a fresh tensor owned by the caller's step, so it is modified in place: with R = 128 rows each [R, V]
// float tensor is 63 MiB, and the old version kept six of them alive at once.
torch::Tensor sample(torch::Tensor logits, const torch::Tensor& temps) {
    torch::Tensor greedy = logits.argmax(-1);
    logits.div_(temps.clamp_min(1e-5).unsqueeze(1));
    torch::Tensor probs = torch::softmax(logits, -1);
    // Gumbel-free exponential-race sampling: argmax(p / Exp(1)) draws from p
    probs.div_(torch::empty_like(probs).exponential_(1.0).clamp_min_(1e-10));
    return torch::where(temps <= 0, greedy, probs.argmax(-1));
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

    torch::manual_seed(cfg_.seed);

    load_weights();
    num_kv_blocks_ = size_kv_cache();
    bind_cache(num_kv_blocks_ + 1);

    out_.reserve(cfg_.max_num_seqs);
    row_owner_.reserve(cfg_.max_num_seqs);

    if (wants_graphs()) capture_graphs(/*scratch_block=*/num_kv_blocks_);   // after the cache is bound: graphs bake in its address
}

ModelRunner::~ModelRunner() = default;

bool ModelRunner::wants_graphs() const { return device_.is_cuda() && !cfg_.enforce_eager && model_.fused_attention(); }

void ModelRunner::load_weights() {
    WeightMap weights = ::load_weights(cfg_.model_path, device_, dtype_);   // free function from utils/tensor.h
    model_.load_weights(weights);
}

void ModelRunner::bind_cache(int64_t num_blocks) {
    const std::vector<int64_t> shape{mcfg_.n_layers, num_blocks, kBlockSize, mcfg_.n_kv_heads, mcfg_.head_dim()};
    const auto opts = torch::TensorOptions().dtype(dtype_).device(device_);
    model_.bind_kv_cache(torch::zeros(shape, opts), torch::zeros(shape, opts));
}

int32_t ModelRunner::size_kv_cache() {
    if (cfg_.num_kv_blocks > 0) return cfg_.num_kv_blocks;
    if (!device_.is_cuda())
        throw std::runtime_error("EngineConfig::num_kv_blocks must be set > 0 for CPU execution");

#ifdef CPPVLLM_HAS_CUDART
    // Loading converts weights (for example fp32 -> bf16); the freed temporaries stay cached by PyTorch and cudaMemGetInfo
    // would count them as used. Hand them back first.
    release_cached_gpu_memory();

    // CUDA graphs live in their own memory pool (about 1-1.5 GiB for 256 sequences) and can only be captured AFTER the real
    // KV cache exists (they bake in its address). So: capture once on a 1-block cache just to MEASURE the pool, destroy the
    // graphs, and keep that many bytes out of the KV budget. The constructor captures the real graphs afterwards.
    size_t graph_bytes = 0;
    if (wants_graphs()) {
        bind_cache(1);
        size_t f0 = 0, f1 = 0, t = 0;
        cudaMemGetInfo(&f0, &t);
        capture_graphs(/*scratch_block=*/0);
        cudaMemGetInfo(&f1, &t);
        graph_bytes = f0 > f1 ? f0 - f1 : 0;
        release_graphs();
        release_cached_gpu_memory();
        std::cerr << "[cppvllm] CUDA graphs need " << static_cast<double>(graph_bytes) / (1ull << 30) << " GiB\n";
    }

    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) throw std::runtime_error("cudaMemGetInfo failed");
    // Weights and the CUDA context are resident, so (total - free) is "used". The (1 - gpu_memory_utilization) share of the
    // GPU stays free for activations (prefill chunks, sampling temporaries) and fragmentation.
    const double used = static_cast<double>(total_b - free_b) + static_cast<double>(graph_bytes);
    const double budget = static_cast<double>(total_b) * cfg_.gpu_memory_utilization - used;
    const double per_block = static_cast<double>(kv_bytes_per_block(mcfg_, c10::elementSize(dtype_)));
    const int64_t n = static_cast<int64_t>(budget / per_block);
    if (n <= 0)
        throw std::runtime_error("weights and CUDA graphs already exceed gpu_memory_utilization of the GPU: raise gpu_memory_utilization or set enforce_eager");
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

    if (can_use_graph(batch)) { run_graph(batch); return out_; }

    // The padded fallback attention wastes work on mixed batches: split decode and prefill when padding would
    // exceed 4x. The fused kernel handles ragged batches in one launch.
    size_t max_q = 0;
    for (const Sequence* s : batch.seqs) max_q = std::max<size_t>(max_q, s->num_scheduled_tokens);
    if (model_.fused_attention() || n * max_q <= 4 * batch.total_tokens) {
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
bool ModelRunner::can_use_graph(const ScheduledBatch& batch) const {
    if (graphs_.empty() || !batch.all_single_token) return false;
    if (static_cast<int64_t>(batch.seqs.size()) > max_graph_batch_) return false;
    for (const Sequence* s : batch.seqs)
        if (static_cast<int64_t>(s->num_cached_tokens) + 1 > graph_width_ * kBlockSize) return false;   // eager path reports it
    return true;
}

void ModelRunner::capture_graphs(int64_t scratch_block) {
#ifdef CPPVLLM_HAS_CUDA_GRAPH
    max_graph_batch_ = std::min<int64_t>(static_cast<int64_t>(cfg_.max_num_seqs), 512);
    // Every replay uploads max_batch x graph_width_ block ids, so keep the table narrow: graphs cover contexts up to 4096
    // tokens (a longer sequence just takes the eager path, see can_use_graph).
    graph_width_ = std::min<int64_t>(max_blocks_per_seq_, ceil_div(4096, kBlockSize));
    graph_scratch_ = scratch_block;                  // idle rows write their garbage K/V here
    const int64_t scratch = scratch_block;

    g_tokens_    = std::make_unique<StagedBuffer<int64_t>>(max_graph_batch_, device_);
    g_positions_ = std::make_unique<StagedBuffer<int64_t>>(max_graph_batch_, device_);
    g_kv_lens_   = std::make_unique<StagedBuffer<int64_t>>(max_graph_batch_, device_, int64_t{1});
    g_tables_    = std::make_unique<StagedBuffer<int64_t>>(max_graph_batch_ * graph_width_, device_, int64_t{-1});
    const auto i64 = torch::TensorOptions().dtype(torch::kLong).device(device_);
    g_cu_     = torch::arange(max_graph_batch_ + 1, i64);
    g_logits_ = torch::empty({max_graph_batch_, static_cast<int64_t>(mcfg_.vocab_size)},
                             torch::TensorOptions().dtype(torch::kFloat).device(device_));

    // batch sizes: 1 2 4 8, then every 16, then the max
    std::vector<int64_t> sizes;
    for (int64_t b : {1, 2, 4, 8}) if (b < max_graph_batch_) sizes.push_back(b);
    for (int64_t b = 16; b < max_graph_batch_; b += 16) sizes.push_back(b);
    sizes.push_back(max_graph_batch_);

    // every row starts idle (token 0, position 0, one key, scratch block)
    {
        auto tok = g_tokens_->host(), pos = g_positions_->host(), kvl = g_kv_lens_->host(), tab = g_tables_->host();
        for (int64_t j = 0; j < max_graph_batch_; ++j) { tok[j] = 0; pos[j] = 0; kvl[j] = 1; tab[j * graph_width_] = scratch; }
        (void)g_tokens_->upload(max_graph_batch_);
        (void)g_positions_->upload(max_graph_batch_);
        (void)g_kv_lens_->upload(max_graph_batch_);
        (void)g_tables_->upload(max_graph_batch_ * graph_width_);
        torch::cuda::synchronize();
    }

    auto pool = at::cuda::graph_pool_handle();       // one pool: graphs replay one at a time and share activation memory
    graphs_.resize(sizes.size());
    for (size_t i = sizes.size(); i-- > 0;) {        // biggest first, so the pool is sized by the biggest
        const int64_t B = sizes[i];
        FlatBatch fb;
        fb.tokens       = g_tokens_->device_view(B);
        fb.positions    = g_positions_->device_view(B);
        fb.cu_seqlens_q = g_cu_.narrow(0, 0, B + 1);
        fb.kv_lens      = g_kv_lens_->device_view(B);
        fb.block_tables = g_tables_->device_view(B * graph_width_).view({B, graph_width_});
        fb.max_q  = 1;
        fb.max_kv = graph_width_ * kBlockSize;

        c10::cuda::CUDAStream side = c10::cuda::getStreamFromPool();   // capture must not run on the default stream
        c10::cuda::CUDAStreamGuard guard(side);

        for (int w = 0; w < 3; ++w) (void)model_.forward(fb);          // warm-up: cuBLAS handles, workspaces
        torch::cuda::synchronize();

        graphs_[i] = std::make_unique<at::cuda::CUDAGraph>();
        graphs_[i]->capture_begin(pool);
        g_logits_.narrow(0, 0, B).copy_(model_.forward(fb));           // output copied inside the graph
        graphs_[i]->capture_end();
        torch::cuda::synchronize();
    }
    graph_sizes_ = std::move(sizes);
    std::cerr << "[cppvllm] captured " << graph_sizes_.size() << " CUDA graphs (batch sizes " << graph_sizes_.front()
              << " .. " << graph_sizes_.back() << ")\n";
#else
    throw std::runtime_error("built without ATen/cuda/CUDAGraph.h: set EngineConfig::enforce_eager = true");
#endif
}

void ModelRunner::release_graphs() {
#ifdef CPPVLLM_HAS_CUDA_GRAPH
    torch::cuda::synchronize();
    graphs_.clear();                                 // destroying the graphs releases their private memory pool
    graph_sizes_.clear();
    g_tokens_.reset(); g_positions_.reset(); g_kv_lens_.reset(); g_tables_.reset();
    g_cu_ = torch::Tensor();
    g_logits_ = torch::Tensor();
    max_graph_batch_ = 0;
#endif
}

void ModelRunner::run_graph(const ScheduledBatch& batch) {
#ifdef CPPVLLM_HAS_CUDA_GRAPH
    const int64_t S = static_cast<int64_t>(batch.seqs.size());
    const size_t gi = static_cast<size_t>(std::lower_bound(graph_sizes_.begin(), graph_sizes_.end(), S) - graph_sizes_.begin());
    const int64_t B = graph_sizes_[gi];
    const int64_t scratch = graph_scratch_;

    auto tok = g_tokens_->host(), pos = g_positions_->host(), kvl = g_kv_lens_->host(), tab = g_tables_->host();
    for (int64_t j = 0; j < S; ++j) {
        const Sequence& s = *batch.seqs[j];
        const int64_t c = s.num_cached_tokens, nblk = ceil_div(c + 1, kBlockSize);
        tok[j] = s.token_ids[c];
        pos[j] = c;
        kvl[j] = c + 1;
        for (int64_t b = 0; b < nblk; ++b) tab[j * graph_width_ + b] = s.block_table[b];   // entries past nblk are never read
    }
    for (int64_t j = S; j < B; ++j) { tok[j] = 0; pos[j] = 0; kvl[j] = 1; tab[j * graph_width_] = scratch; }

    (void)g_tokens_->upload(B);
    (void)g_positions_->upload(B);
    (void)g_kv_lens_->upload(B);
    (void)g_tables_->upload(B * graph_width_);
    graphs_[gi]->replay();

    torch::Tensor logits = g_logits_.narrow(0, 0, S);

    auto rows = rows_.host();
    auto temp = temps_.host();
    int64_t R = 0;
    row_owner_.clear();
    for (int64_t j = 0; j < S; ++j) {
        if (samples_this_step(*batch.seqs[j])) {
            rows[R] = j;
            temp[R] = batch.seqs[j]->params.temperature;
            row_owner_.push_back(static_cast<size_t>(j));
            ++R;
        }
    }
    if (R == 0) { torch::cuda::synchronize(); return; }      // host buffers are reused next step

    torch::Tensor next = sample(logits.index_select(0, rows_.upload(R)), temps_.upload(R)).to(torch::kCPU);   // blocking = sync
    const int64_t* p = next.data_ptr<int64_t>();
    for (int64_t r = 0; r < R; ++r) out_[row_owner_[r]] = static_cast<int32_t>(p[r]);
#else
    (void)batch;
#endif
}