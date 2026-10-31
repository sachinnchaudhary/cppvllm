#pragma once 

#include "config.h"  
#include "torch/torch.h"  
#include <string> 
#include <unordered_map>
#include <vector> 


using WeightMap = std::unordered_map<std::string, torch::Tensor>;  

struct FlatBatch {  
    torch::Tensor tokens;  // [T]      int64  token ids 
    torch::Tensor positions;  
    torch::Tensor cu_seqlens_q;  
    torch::Tensor kv_lens;
    torch::Tensor block_tables;  
    int64_t max_q = 0; 
    int64_t max_kv = 0;    
};  

class Llama {
public:  
   explicit Llama(const ModelConfig& cfg);  
   
   void load_weights(const WeightMap& w);  
   void bind_kv_cache(torch::Tensor k, torch::Tensor v);  
   
   torch::Tensor forward(const FlatBatch& b, const torch::Tensor& sample_rows = torch::Tensor());  

   [[nodiscard]] int64_t block_size() const {return block_size_;}  
   [[nodiscard]] const ModelConfig& config()  const {return cfg_;}
   
private: 
  struct Layer {  
     torch::Tensor wq, wk, wv, wo;  
     torch::Tensor gate, up, down;  
     torch::Tensor attn_norm, ffn_norm;  
  }

  void build_rope_tables();  

  ModelConfig cfg_;  
  std::vectors<Layer>layers_;
  torch::Tensor embed_, final_norm_, lm_head_;  
  torch::Tensor cos_, sin_;  
  torch::Tensor k_cache_, v_cache_;  
  int64_t block_size = 0;  
  bool loaded_ = false;  
};  

