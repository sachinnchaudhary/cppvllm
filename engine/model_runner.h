#pragma once

#include "config.h"
#include "model_architecture.h"
#include "scheduler.h"
#include "sequence.h"
#include "utils/tensor.h"
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace at::cuda { struct CUDAGraph; }

class ModelRunner {
public:
    ModelRunner(const EngineConfig& cfg, const ModelConfig& mcfg, torch::Device device);
    ~ModelRunner();                                   // defined in the .cpp, where CUDAGraph is a complete type
    ModelRunner(const ModelRunner&) = delete;
    ModelRunner& operator=(const ModelRunner&) = delete;

    // One output per batch.seqs entry: the sampled token, or -1 if that sequence did not sample.
    [[nodiscard]] const std::vector<int32_t>& run(const ScheduledBatch& batch);

    [[nodiscard]] int32_t num_kv_blocks() const noexcept { return num_kv_blocks_; }   // blocks the scheduler may use
    [[nodiscard]] torch::Device device() const noexcept { return device_; }
    [[nodiscard]] torch::ScalarType dtype() const noexcept { return dtype_; }
    [[nodiscard]] bool graphs_enabled() const noexcept { return !graphs_.empty(); }

private:
    static constexpr int64_t kBlockSize = static_cast<int64_t>(Sequence::BLOCK_SIZE);

    void load_weights();
    int32_t size_kv_cache();
    void bind_cache(int64_t num_blocks);
    void run_group(const ScheduledBatch& batch, const std::vector<size_t>& idx);

    [[nodiscard]] bool wants_graphs() const;
    void capture_graphs(int64_t scratch_block);   // scratch_block: cache block that idle rows write into
    void release_graphs();                        // destroy every graph and give their memory pool back
    [[nodiscard]] bool can_use_graph(const ScheduledBatch& batch) const;
    void run_graph(const ScheduledBatch& batch);

    // declaration order = initialisation order
    EngineConfig cfg_;
    ModelConfig mcfg_;
    torch::Device device_;
    torch::ScalarType dtype_;
    Qwen3 model_;
    int32_t num_kv_blocks_ = 0;
    int64_t max_blocks_per_seq_;

    StagedBuffer<int64_t> tokens_, positions_, cu_seqlens_, kv_lens_, tables_, rows_;
    StagedBuffer<float> temps_;

    // CUDA graphs (decode only: every row is one token)
    int64_t graph_scratch_ = 0;
    int64_t max_graph_batch_ = 0;
    int64_t graph_width_ = 0;
    std::unique_ptr<StagedBuffer<int64_t>> g_tokens_, g_positions_, g_kv_lens_, g_tables_;
    torch::Tensor g_cu_;
    torch::Tensor g_logits_;
    std::vector<int64_t> graph_sizes_;
    std::vector<std::unique_ptr<at::cuda::CUDAGraph>> graphs_;

    std::vector<int32_t> out_;
    std::vector<size_t> row_owner_;
};