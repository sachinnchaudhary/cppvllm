#pragma once 
#include "block_manager.h"  
#include "sequence.h" 

#include <span> 
#include <vector> 
#include <deque>  
#include <cstddef>
#include <cstdint>


struct SchedulerConfig { 
    size_t max_num_batched_tokens = 2048;  
    size_t max_num_seqs = 128;  
    size_t max_chunk_tokens = 0;  
    int32_t eos_token_id = -1;
};


struct ScheduledBatch  
{
    std::vector<Sequence*> seqs;  
    size_t total_tokens = 0;    
    
    //pure decode to play cudagraph 
    bool all_single_token = false;

    [[nodiscard]] bool empty() const noexcept {return seqs.empty();}  
};

[[nodiscard]] inline bool samples_this_step(const Sequence& s) noexcept { 
        return s.num_cached_tokens + s.num_scheduled_tokens == s.size();
}


class Scheduler { 

public:  

  Scheduler(const SchedulerConfig& cfg, int32_t num_kv_blocks); 
  
  void add_request(Sequence* seq);  
  [[nodiscard]] const ScheduledBatch& schedule();  
  void postprocess(const ScheduledBatch& batch, std::span<const int32_t> sampled);  

  [[nodiscard]] bool is_finished() const noexcept     {return waiting_.empty() && running_.empty();}
  [[nodiscard]] size_t num_waiting() const noexcept   {return waiting_.size();}  
  [[nodiscard]] size_t num_running() const noexcept   {return running_.size();}  
  [[nodiscard]] size_t num_preemptions() const noexcept {return num_preemptions_;}
  [[nodiscard]] const BlockManager& block_manager() const noexcept {return bm_;}

private:  

  void preempt(Sequence* seq);
  
  SchedulerConfig cfg_;  
  BlockManager bm_; 
  std::deque<Sequence*> waiting_; 
  std::vector<Sequence*> running_;  
  
  ScheduledBatch batch_;  
  size_t num_preemptions_ = 0; 
};  


