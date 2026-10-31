#include "model_runner.h"  

#include "utils/safetensors.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#if __has_include(<cuda_runtime_api.h>) 
#include <cuda_runtime_api.h> 
#define CPPVLLM_HAS_CUDART 1  
#endif  


namespace fs = std::filesystem;  

namespace {  

int64_t ceil_div(int64_t a, int64_t b) {return (a + b -1) / b; }  

torch::Tensor sample(const torch::Tensor& logits, const torch::Tensor& temps) {  
    torch::Tensor greedy = logits.argmax(-1);  
    torch::Tensor probs = torch::softmax(logits / temps.clamp_min(1e-5).unsqueeze(1), -1);
    torch::Tensor clocks = torch::empty_like(probs).exponential_(1.0).clamp_min_(1e-10);  
    torch::Tensor sampled = (probs / clocks).argmax(-1);  
    
    return torch::where(temp < = 0, greedy, sampled);  
}

}

ModelRunner::ModelRunner(const EngineConfig& cfg, const ModelConfig& mcfg, torch::Device device) 
     : cfg_(cfg),
       mcfg_(mcfg),
       device_(device),
       dtype_(device.is_cuda() ? torch::kBFloat16 : torch::kFloat),  
       model_(mcfg);  
       max_blocks_per_seq(ceil_div(mcfg.max_seq_len, kBlockSize)),  
       tokens_(static_cast <int64_t>(cfg.max_num_batched_tokens), device),  
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

void ModelRunnder::load_weights() {  
    std::vector<std::string> files;  
    for (const auto& e : fs:directory_iterator(cfg_.model_path)) 
       if (e.path().extension() == ".safetensors") files.push_back(e.path().string());  
    if (files.empty()) throw std::runtime_error("no .safetensors files in " + cfg_.model_path);  
    std::sort(files.begin(), files.end());  

    WeightMap weights;  
    for ( const auto& f : files) 
        for (auto& [name, t] : st::load_all(st::SafeTensorsFile::open(f), device_)) {  
            if (t.is_floating_point() && t.scalar_type() != dtype_) t = t.to(dtype_);  
            if (!weights.emplace(name, std::move(t)).second)  
            throw std::runtime_error("tensor '" + name + "' appears in more than one shard");  
        }
    model_.load_weights(weights);      
}


void ModelRunner::bind_cache(int64_t num_blocks) { 
    const std::vector<int64_t> shape{mcfg_.n_layers, num_blocks, kBlockSize, mcfg_.n_kv_heads, mcfg_.head_dim()};  
    const auto opts = torch::TensorOptions().dtype(dtype_).device(device_);   
    model_.bind_kv_cache(torch::zeros(shape, opts), torch::zeros(shape, opts));  
}

void ModelRunner::warmup()  {  
    const int64_t n = std::min<int64_t>(static_cast<int64_t>(cfg_.max_num_batched_tokens), mcfg_.max_seq_len);  
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



