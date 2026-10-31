#pragma once  

#include "config.h"
#include "model_architecture.h"
#include "scheduler.h"
#include "sequence.h" 
#include <torch/torch.h>
#include "utils/tensor.h"

#include <cstdint>
#include <vector>


class ModelRunner { 
public: 
  ModelRunner(const EngineConfig& cfg, const ModelConfig& mcfg, torch::Device device);  

  [[nodiscard]] const std::vector<int32_t>& run(const ScheduledBatch& batch);  
  
  [[nodiscard]] int32_t   num_kv_block() const noexcept {return num_kv_block_;}
  [[nodiscard]] torch::Device device() const noexcept {return device_;}  
  [[nodiscard]] torch::ScalarType dtype() const noexcept {return dtype_;}  

private:  
  static constexpr int64_t kBlockSize = static_cast<int64_t>(Sequence::BLOCK_SIZE); 

  void load_weights();  
  int32_t size_kv_cache();  
  void   warmup();  
  void  bind_cache(int64_t num_blocks);  
  void  run_group(const ScheduledBatch& batch, const std::vector<size_t>& idx);  

  EngineConfig cfg_;  
  ModelConfig mcfg_;  
  torch::Device device_;  
  torch::ScalarType dtype_; 
  Llama model_;  
  int32_t num_kv_blocks = 0;  
  int64_t max_blocks_per_seq_;  
  
StagedBuffer<int64_t> tokens_, positions_, cu_seqlens_, kv_lens_, tables_, rows_;  
  StagedBuffer<float> temps_;  

  std::vector<int32_t> out_;  
  std::vector<size_t> row_owner_;  

}



 