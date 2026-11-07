#pragma once

#include "config.h"
#include "model/model_architecture.h"
#include "scheduler.h"
#include "sequence.h"
#include "utils/tensor.h"
#include <torch/torch.h>

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

namespace at::cuda { struct CUDAGraph; }

class ModelRunner {
public:
  
    ModelRunner(const EngineConfig& cfg, const ModelConfig& mcfg, torch::Device device,
                int tp_rank = 0, int tp_size = 1, ncclComm_t comm = nullptr,
                bool defer_graph_capture = false, const WeightMap* host_weights = nullptr);
    ~ModelRunner();                                  
    ModelRunner(const ModelRunner&) = delete;
    ModelRunner& operator=(const ModelRunner&) = delete;

    void launch(const ScheduledBatch& batch);
   
    [[nodiscard]] const std::vector<int32_t>& collect();
   
    [[nodiscard]] const std::vector<int32_t>& run(const ScheduledBatch& batch);

    [[nodiscard]] int in_flight() const noexcept { return in_flight_; }
    [[nodiscard]] int32_t num_kv_blocks() const noexcept { return num_kv_blocks_; }
    [[nodiscard]] torch::Device device() const noexcept { return device_; }
    [[nodiscard]] torch::ScalarType dtype() const noexcept { return dtype_; }
    [[nodiscard]] bool graphs_enabled() const noexcept { return !graphs_.empty(); }
   
    void capture_graphs_if_enabled();
    [[nodiscard]] int tp_rank() const noexcept { return model_.tp_rank(); }
    [[nodiscard]] int tp_size() const noexcept { return model_.tp_size(); }

private:
    static constexpr int64_t kBlockSize = static_cast<int64_t>(Sequence::BLOCK_SIZE);
    static constexpr int kMaxInFlight = 2;                                  
    static constexpr int kStageSlots = StagedBuffer<int64_t>::kSlots;      

    void load_weights(const WeightMap* host_weights);
    int32_t size_kv_cache();
    void bind_cache(int64_t num_blocks);

    struct Group {
        std::vector<Sequence*> seqs;
        std::vector<size_t> slot;     
        void clear() { seqs.clear(); slot.clear(); }
    };
    using Seqs = std::span<Sequence* const>;
    using Slots = std::span<const size_t>;

    struct Pending {
        std::vector<int32_t> out_idx; 
        size_t n_seqs = 0;
    };
   
    struct SampleStage {
        int64_t R = 0;
        torch::Tensor rows, temps;
    };

    void begin_stage();
    void end_stage();

    [[nodiscard]] bool graph_eligible(const Sequence& s) const;     
    void run_eager(const Group& g);                                   

    SampleStage stage_sampling(Seqs seqs, Slots slot);               
    void finish_sampling(const SampleStage& ss, const torch::Tensor& logits);   

    [[nodiscard]] int64_t feed_source(const Sequence& s) const;
    void apply_feed(const torch::Tensor& tokens_view, int64_t n);

    // CUDA graphs (decode only: every row is one token)
    void capture_graphs();
    bool stage_decode_rows(Seqs seqs, int64_t B);                    
    [[nodiscard]] FlatBatch graph_inputs(int64_t B) const;
    void run_graph(Seqs seqs, Slots slot);                           

    struct Events;                                                   

    EngineConfig cfg_;
    ModelConfig mcfg_;
    torch::Device device_;
    torch::ScalarType dtype_;
    Qwen3 model_;
    int32_t num_kv_blocks_ = 0;
    int64_t max_blocks_per_seq_;
    const int64_t max_graph_batch_;  
    const int64_t graph_width_;      

    StagedBuffer<int64_t> tokens_, positions_, cu_seqlens_, kv_lens_, tables_, rows_, feed_;
    StagedBuffer<float> temps_;

    std::map<int64_t, std::unique_ptr<at::cuda::CUDAGraph>> graphs_;   
    torch::Tensor g_hidden_;                                          

    Group decode_, fresh_, cont_;                                            
    std::vector<int32_t> out_;
    uint64_t sample_step_ = 0;                                     

    // ---- launch / collect state ----
    std::unique_ptr<Events> ev_;
    int stage_slot_ = 0;
    std::array<Pending, kMaxInFlight> pending_;
    std::array<torch::Tensor, kMaxInFlight> sampled_dev_;  
    std::array<torch::Tensor, kMaxInFlight> res_host_;      
    int next_launch_ = 0, next_collect_ = 0, in_flight_ = 0;
    int launch_slot_ = 0;                                  
    int64_t n_sampled_ = 0;                               
    int prev_slot_ = -1;                                   
    std::unordered_map<uint64_t, int32_t> prev_idx_, cur_idx_;  
};