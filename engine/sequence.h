#pragma once 

#include <algorithm>
#include <vector>  
#include <span>
#include <cassert>
#include <cstddef>
#include <cstdint>

enum class Status: uint8_t {  
        WAITING = 0,
        RUNNING = 1,
        FINISHED = 2
};

inline constexpr int32_t kPendingToken = -1;

struct SamplingParams {  
    
    float temperature = 1.0f; 
    int32_t max_tokens = 256;  
    bool ignore_eos = false;  

};

struct Sequence {  
        
    static constexpr size_t BLOCK_SIZE = 16;  
    
    uint64_t seq_id;  
    Status status = Status::WAITING;  
    SamplingParams params;  

    std::vector<int32_t> token_ids;
    std::vector<int32_t> block_table;   
    
    uint32_t num_prompt_tokens;  
    uint32_t num_cached_tokens = 0; 
    uint32_t num_scheduled_tokens = 0;
    uint32_t num_inflight = 0;   // launched steps not yet postprocessed that contain this sequence

    Sequence(uint64_t id, std::span<const int32_t> prompt, const SamplingParams& p = {})  
    :  
       seq_id(id),
       params(p),
       token_ids(prompt.begin(), prompt.end()),  
       num_prompt_tokens(static_cast<uint32_t>(prompt.size())) {  
        token_ids.reserve(num_prompt_tokens + params.max_tokens);  
        block_table.reserve((num_prompt_tokens + params.max_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE);
       } 
       
       [[nodiscard]] size_t size() const noexcept { return token_ids.size(); }

       [[nodiscard]] size_t num_blocks() const noexcept {  return (size() + BLOCK_SIZE - 1) / BLOCK_SIZE; }
       
       [[nodiscard]] size_t last_block_num_tokens() const noexcept {  return size() - ((num_blocks() - 1) * BLOCK_SIZE);}
       
       [[nodiscard]] size_t num_completion_tokens() const noexcept { return size() - num_prompt_tokens; }
       
       [[nodiscard]] size_t num_uncached_tokens() const noexcept {return size() - num_cached_tokens; }
        
       //Zero copy view of block(i)//  
       [[nodiscard]] std::span<const int32_t> block(size_t i) const noexcept 
         {  
            assert(i < num_blocks());  
            const size_t offset = i * BLOCK_SIZE;  
            const size_t count =  std::min(BLOCK_SIZE, size() - offset); 
            
            return std::span<const int32_t>(token_ids.data() + offset, count);  
         } 
        
       void append_token(int32_t token_id) 
         {
            token_ids.push_back(token_id);  
         }
        
       [[nodiscard]] bool is_finished() const noexcept 
         { 
            return status == Status::FINISHED; 
         }

       void reset_for_preemption() 
         {  
            assert(num_inflight == 0);
            status = Status::WAITING;   
            block_table.clear();  
            num_cached_tokens = 0;  
            num_scheduled_tokens = 0;  
         }
};