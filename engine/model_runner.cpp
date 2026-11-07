#include "model_runner.h"
#include "kernels/fused_ops.h"

#include <torch/torch.h>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

#include <cuda_runtime_api.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

namespace {

inline int64_t ceil_div(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

torch::ScalarType pick_dtype(const EngineConfig& cfg) {
     switch (cfg.dtype) {
        case DType::BF16: return torch::kBFloat16;
        case DType::FP16: return torch::kHalf;
        case DType::FP32: return torch::kFloat;
        default:          return torch::kBFloat16;
    }
}

} 

struct ModelRunner::Events {
    at::cuda::CUDAEvent slot_done[kStageSlots];     
    at::cuda::CUDAEvent step_done[kMaxInFlight];   
};

ModelRunner::ModelRunner(const EngineConfig& cfg, const ModelConfig& mcfg, torch::Device device,
                         int tp_rank, int tp_size, ncclComm_t comm, bool defer_graph_capture,
                         const WeightMap* host_weights)
    : cfg_(cfg),
      mcfg_(mcfg),
      device_(device),
      dtype_(pick_dtype(cfg)),
      model_(mcfg, tp_rank, tp_size, comm),
      max_blocks_per_seq_(ceil_div(std::min<int64_t>(cfg.max_model_len, mcfg.max_seq_len), kBlockSize)),
      max_graph_batch_(std::min<int64_t>({static_cast<int64_t>(cfg.max_num_seqs), 512, static_cast<int64_t>(cfg.max_num_batched_tokens)})),
      graph_width_(std::min<int64_t>(max_blocks_per_seq_, ceil_div(4096, kBlockSize))),
      tokens_(static_cast<int64_t>(cfg.max_num_batched_tokens), device),
      positions_(static_cast<int64_t>(cfg.max_num_batched_tokens), device),
      cu_seqlens_(static_cast<int64_t>(cfg.max_num_seqs) + 1, device),
      kv_lens_(static_cast<int64_t>(cfg.max_num_seqs), device),
      tables_(static_cast<int64_t>(cfg.max_num_seqs) * max_blocks_per_seq_, device, int64_t{-1}),
      rows_(static_cast<int64_t>(cfg.max_num_seqs), device),
      feed_(static_cast<int64_t>(cfg.max_num_batched_tokens), device, int64_t{-1}),
      temps_(static_cast<int64_t>(cfg.max_num_seqs), device, 1.0f)
{
 
    c10::cuda::CUDAGuard device_guard(device_);
    ev_ = std::make_unique<Events>();
    load_weights(host_weights);
    c10::cuda::CUDACachingAllocator::resetPeakStats(device_.index());
    num_kv_blocks_ = size_kv_cache();
    bind_cache(num_kv_blocks_ + 1);                          // +1: scratch block for idle graph rows
    std::cout << "[cppvllm] KV cache: fp8 e4m3 + fp32 scale per (token, kv head), " << num_kv_blocks_ << " blocks x "
              << kBlockSize << " tokens\n";
    out_.reserve(cfg_.max_num_seqs);

    const int64_t n = static_cast<int64_t>(cfg_.max_num_seqs);
    const auto i64_dev  = torch::TensorOptions().dtype(torch::kLong).device(device_);
    const auto i64_host = torch::TensorOptions().dtype(torch::kLong).pinned_memory(device_.is_cuda());
    for (int i = 0; i < kMaxInFlight; ++i) {
        sampled_dev_[i] = torch::zeros({n}, i64_dev);
        res_host_[i] = torch::zeros({n}, i64_host);
        pending_[i].out_idx.reserve(static_cast<size_t>(n));
    }

    if (!cfg_.enforce_eager && !defer_graph_capture) {
        capture_graphs();
    }
}

ModelRunner::~ModelRunner() = default;

void ModelRunner::capture_graphs_if_enabled() {
    c10::cuda::CUDAGuard device_guard(device_);
    if (!cfg_.enforce_eager && graphs_.empty()) capture_graphs();
}

void ModelRunner::load_weights(const WeightMap* host_weights) {
    if (host_weights) {
        model_.load_weights(*host_weights, device_);
        return;
    }
    WeightMap weights = ::load_weights(cfg_.model_path, device_, dtype_);
    model_.load_weights(weights, device_);
}

void ModelRunner::bind_cache(int64_t num_blocks) {
    const ModelConfig& local_mcfg = model_.config();
    const std::vector<int64_t> shape{local_mcfg.n_layers, num_blocks, kBlockSize,
                                     local_mcfg.n_kv_heads, model_.head_dim()};
    const std::vector<int64_t> sshape{local_mcfg.n_layers, num_blocks, kBlockSize,
                                      local_mcfg.n_kv_heads};
    const auto u8  = torch::TensorOptions().dtype(torch::kUInt8).device(device_);
    const auto f32 = torch::TensorOptions().dtype(torch::kFloat).device(device_);
    model_.bind_kv_cache(torch::zeros(shape, u8), torch::zeros(shape, u8), torch::zeros(sshape, f32), torch::zeros(sshape, f32));
}


int32_t ModelRunner::size_kv_cache() {
    if (cfg_.num_kv_blocks > 0) return cfg_.num_kv_blocks;

    c10::cuda::CUDACachingAllocator::emptyCache();

    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);

    const double total = static_cast<double>(total_b);
    const double used = static_cast<double>(total_b - free_b);

    auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device_.index());
    double peak = static_cast<double>(stats.allocated_bytes[0].peak);
    double current = static_cast<double>(stats.allocated_bytes[0].current);

    const double budget = (total * cfg_.gpu_memory_utilization) - used - peak + current;

    if (budget <= 0) {
        throw std::runtime_error("Calculated KV cache budget is zero or negative");
    }

    const ModelConfig& local_mcfg = model_.config();
    const double per_block = 2.0 * static_cast<double>(local_mcfg.n_layers) * static_cast<double>(kBlockSize) *
                             static_cast<double>(local_mcfg.n_kv_heads) *
                             (static_cast<double>(model_.head_dim()) + static_cast<double>(sizeof(float)));
    const int64_t n = static_cast<int64_t>(budget / per_block);

    return static_cast<int32_t>(std::min<int64_t>(n, INT32_MAX));

}

bool ModelRunner::graph_eligible(const Sequence& s) const {
    return !graphs_.empty() && s.num_scheduled_tokens == 1 &&
           ceil_div(s.num_cached_tokens + 1, kBlockSize) <= graph_width_;
}

void ModelRunner::begin_stage() {
    stage_slot_ = (stage_slot_ + 1) % kStageSlots;
    ev_->slot_done[stage_slot_].synchronize();              
    tokens_.select_slot(stage_slot_);
    positions_.select_slot(stage_slot_);
    cu_seqlens_.select_slot(stage_slot_);
    kv_lens_.select_slot(stage_slot_);
    tables_.select_slot(stage_slot_);
    rows_.select_slot(stage_slot_);
    feed_.select_slot(stage_slot_);
    temps_.select_slot(stage_slot_);
}

void ModelRunner::end_stage() {
    ev_->slot_done[stage_slot_].record(c10::cuda::getCurrentCUDAStream(device_.index()));
}

void ModelRunner::launch(const ScheduledBatch& batch) {
    c10::cuda::CUDAGuard device_guard(device_);
    launch_slot_ = next_launch_;
    Pending& p = pending_[launch_slot_];
    p.out_idx.clear();
    p.n_seqs = batch.seqs.size();
    cur_idx_.clear();
    n_sampled_ = 0;

    decode_.clear();
    fresh_.clear();
    cont_.clear();
    for (size_t i = 0; i < batch.seqs.size(); ++i) {
        Sequence* s = batch.seqs[i];
        Group& g = graph_eligible(*s)                                              ? decode_
                 : (s->num_cached_tokens == 0 && s->num_scheduled_tokens > 1)     ? fresh_
                                                                                   : cont_;
        g.seqs.push_back(s);
        g.slot.push_back(i);
    }

    const size_t cap = static_cast<size_t>(max_graph_batch_);
    for (size_t start = 0; start < decode_.seqs.size(); start += cap) {
        const size_t n = std::min(cap, decode_.seqs.size() - start);
        run_graph(Seqs(decode_.seqs).subspan(start, n), Slots(decode_.slot).subspan(start, n));
    }
    if (!fresh_.seqs.empty()) run_eager(fresh_);
    if (!cont_.seqs.empty())  run_eager(cont_);
    if (n_sampled_ > 0 && model_.tp_rank() == 0) {
        res_host_[launch_slot_].narrow(0, 0, n_sampled_)
            .copy_(sampled_dev_[launch_slot_].narrow(0, 0, n_sampled_), /*non_blocking=*/true);
    }
    ev_->step_done[launch_slot_].record(c10::cuda::getCurrentCUDAStream(device_.index()));
    prev_idx_.swap(cur_idx_);
    prev_slot_ = launch_slot_;
    next_launch_ = (launch_slot_ + 1) % kMaxInFlight;
    ++in_flight_;
}

const std::vector<int32_t>& ModelRunner::collect() {
    c10::cuda::CUDAGuard device_guard(device_);
    const int s = next_collect_;
    const Pending& p = pending_[s];

    ev_->step_done[s].synchronize();                         

    out_.assign(p.n_seqs, -1);
    if (model_.tp_rank() == 0) {
        const int64_t* h = res_host_[s].data_ptr<int64_t>();
        for (size_t i = 0; i < p.out_idx.size(); ++i) out_[p.out_idx[i]] = static_cast<int32_t>(h[i]);
    }

    next_collect_ = (s + 1) % kMaxInFlight;
    --in_flight_;
    return out_;
}

const std::vector<int32_t>& ModelRunner::run(const ScheduledBatch& batch) {
    c10::cuda::CUDAGuard device_guard(device_);
    launch(batch);
    return collect();
}

ModelRunner::SampleStage ModelRunner::stage_sampling(Seqs seqs, Slots slot) {
    SampleStage ss;
    auto rows = rows_.host();
    auto temp = temps_.host();
    Pending& p = pending_[launch_slot_];

    int64_t R = 0;
    for (size_t j = 0; j < seqs.size(); ++j) {
        if (!samples_this_step(*seqs[j])) continue;
        rows[R] = static_cast<int64_t>(j);
        temp[R] = seqs[j]->params.temperature;
        p.out_idx.push_back(static_cast<int32_t>(slot[j]));
        cur_idx_[seqs[j]->seq_id] = static_cast<int32_t>(n_sampled_ + R);
        ++R;
    }
    ss.R = R;
    if (R > 0) {
        ss.rows = rows_.upload(R);
        ss.temps = temps_.upload(R);
    }
    return ss;
}

void ModelRunner::finish_sampling(const SampleStage& ss, const torch::Tensor& logits) {
    if (ss.R == 0) return;
    torch::Tensor out = sampled_dev_[launch_slot_].narrow(0, n_sampled_, ss.R);
    if (model_.tp_rank() == 0) {
        fused::sample_tokens(out, logits, ss.rows, ss.temps,
                             static_cast<uint64_t>(cfg_.seed), sample_step_++);
    }
    model_.broadcast_samples(out);
    n_sampled_ += ss.R;
}

int64_t ModelRunner::feed_source(const Sequence& s) const {
    auto it = prev_idx_.find(s.seq_id);
    return it->second;
}

void ModelRunner::apply_feed(const torch::Tensor& tokens_view, int64_t n) {
    torch::Tensor tokens = tokens_view;
    torch::Tensor src = feed_.device_view(n);
    fused::feed_tokens(tokens, src, sampled_dev_[prev_slot_]);
}

void ModelRunner::run_eager(const Group& g) {
    const int64_t S = static_cast<int64_t>(g.seqs.size());

    int64_t T = 0, W = 0, max_q = 0, max_kv = 0, total_kv = 0;
    for (const Sequence* s : g.seqs) {
        const int64_t q = s->num_scheduled_tokens, kv = s->num_cached_tokens + q;
        T += q;
        total_kv += kv;                                      
        W = std::max(W, ceil_div(kv, kBlockSize));
        max_q = std::max(max_q, q);
        max_kv = std::max(max_kv, kv);
    }

    begin_stage();
    auto tok = tokens_.host(), pos = positions_.host(), cu = cu_seqlens_.host(), kvl = kv_lens_.host(), tab = tables_.host();
    auto feed = feed_.host();
    bool any_feed = false;
    int64_t t = 0;
    cu[0] = 0;
    for (int64_t j = 0; j < S; ++j) {
        const Sequence& s = *g.seqs[j];
        const int64_t c = s.num_cached_tokens, q = s.num_scheduled_tokens, kv = c + q, nblk = ceil_div(kv, kBlockSize);
        for (int64_t k = 0; k < q; ++k, ++t) {
            const int32_t id = s.token_ids[c + k];
            if (id == kPendingToken) { tok[t] = 0; feed[t] = feed_source(s); any_feed = true; }
            else                     { tok[t] = id; feed[t] = -1; }
            pos[t] = c + k;
        }
        cu[j + 1] = t;
        kvl[j] = kv;
        for (int64_t b = 0; b < W; ++b) tab[j * W + b] = b < nblk ? s.block_table[b] : -1;
    }

    FlatBatch fb;
    fb.tokens = tokens_.upload(T);
    fb.positions = positions_.upload(T);
    fb.cu_seqlens_q = cu_seqlens_.upload(S + 1);
    fb.kv_lens = kv_lens_.upload(S);
    fb.block_tables = tables_.upload(S * W).view({S, W});
    fb.max_q = max_q;
    fb.max_kv = max_kv;
    fb.total_kv = total_kv;
    if (any_feed) feed_.upload(T);
    SampleStage ss = stage_sampling(g.seqs, g.slot);
    end_stage();                                            

    if (any_feed) apply_feed(fb.tokens, T);
    torch::Tensor logits = model_.forward(fb);
    finish_sampling(ss, logits);
}
bool ModelRunner::stage_decode_rows(Seqs seqs, int64_t B) {
    const int64_t S = static_cast<int64_t>(seqs.size());
    const int64_t scratch = num_kv_blocks_;                  
    auto tok = tokens_.host(), pos = positions_.host(), cu = cu_seqlens_.host(), kvl = kv_lens_.host(), tab = tables_.host();
    auto feed = feed_.host();
    bool any_feed = false;
    for (int64_t j = 0; j < B; ++j) {
        cu[j] = j;                                           
        feed[j] = -1;
        if (j < S) {
            const Sequence& s = *seqs[j];
            const int64_t c = s.num_cached_tokens, nblk = ceil_div(c + 1, kBlockSize);
            const int32_t id = s.token_ids[c];
            if (id == kPendingToken) { tok[j] = 0; feed[j] = feed_source(s); any_feed = true; }
            else                     { tok[j] = id; }
            pos[j] = c;
            kvl[j] = c + 1;
            for (int64_t b = 0; b < nblk; ++b) tab[j * graph_width_ + b] = s.block_table[b];   
        } else {
            tok[j] = 0; pos[j] = 0; kvl[j] = 1; tab[j * graph_width_] = scratch;
        }
    }
    cu[B] = B;
    tokens_.upload(B);
    positions_.upload(B);
    cu_seqlens_.upload(B + 1);
    kv_lens_.upload(B);
    tables_.upload(B * graph_width_);
    if (any_feed) feed_.upload(B);
    return any_feed;
}

FlatBatch ModelRunner::graph_inputs(int64_t B) const {
    FlatBatch fb;
    fb.tokens       = tokens_.device_view(B);
    fb.positions    = positions_.device_view(B);
    fb.cu_seqlens_q = cu_seqlens_.device_view(B + 1);
    fb.kv_lens      = kv_lens_.device_view(B);
    fb.block_tables = tables_.device_view(B * graph_width_).view({B, graph_width_});
    fb.max_q  = 1;
    fb.max_kv = graph_width_ * kBlockSize;
    return fb;
}

void ModelRunner::capture_graphs() {
    c10::cuda::CUDAGuard device_guard(device_);
    size_t free_before = 0, free_after = 0, total_b = 0;
    cudaMemGetInfo(&free_before, &total_b);

    g_hidden_ = torch::empty({max_graph_batch_, static_cast<int64_t>(mcfg_.dim)},
                             torch::TensorOptions().dtype(dtype_).device(device_));
    begin_stage();
    stage_decode_rows({}, max_graph_batch_);                 
    end_stage();
    torch::cuda::synchronize();

    std::vector<int64_t> sizes;
    for (int64_t b : {1, 2, 4, 8}) if (b < max_graph_batch_) sizes.push_back(b);
    for (int64_t b = 16; b < max_graph_batch_; b += 16) sizes.push_back(b);
    sizes.push_back(max_graph_batch_);

    const at::cuda::MempoolId_t pool = at::cuda::graph_pool_handle();
    c10::cuda::CUDAStream side = c10::cuda::getStreamFromPool();   
    c10::cuda::CUDAStreamGuard guard(side);

    for (auto it = sizes.rbegin(); it != sizes.rend(); ++it) {  
        const int64_t B = *it;
        const FlatBatch fb = graph_inputs(B);

        for (int w = 0; w < 3; ++w) (void)model_.forward_backbone(fb);          
        torch::cuda::synchronize();

        auto g = std::make_unique<at::cuda::CUDAGraph>();
        g->capture_begin(pool, cudaStreamCaptureModeThreadLocal);
        g_hidden_.narrow(0, 0, B).copy_(model_.forward_backbone(fb));
        g->capture_end();
        torch::cuda::synchronize();
        graphs_[B] = std::move(g);
    }
    torch::cuda::synchronize();
    cudaMemGetInfo(&free_after, &total_b);
    std::cout << "[cppvllm] captured " << graphs_.size() << " CUDA graphs (batch sizes " << graphs_.begin()->first
              << " .. " << graphs_.rbegin()->first << "), "
              << (free_before - free_after) / (1024.0 * 1024.0) << " MiB used (pool + graph objects + g_hidden_)\n";

}

void ModelRunner::run_graph(Seqs seqs, Slots slot) {
    const int64_t S = static_cast<int64_t>(seqs.size());
    auto g = graphs_.lower_bound(S);                         
    const int64_t B = g->first;

    begin_stage();
    const bool any_feed = stage_decode_rows(seqs, B);
    SampleStage ss = stage_sampling(seqs, slot);
    end_stage();                                             

    if (any_feed) apply_feed(tokens_.device_view(B), B);     
    g->second->replay();                                    
    torch::Tensor logits = model_.compute_logits(g_hidden_.narrow(0, 0, S));   
    finish_sampling(ss, logits);
}