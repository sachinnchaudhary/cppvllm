#pragma once 
#include "sequence.h"  
#include <unordered_map>
#include <vector>  
#include <cstddef>
#include <cstdint>

inline constexpr uint64_t kNoHash = 0; 



class BlockManager {  

public:  
      explicit BlockManager(int32_t num_blocks);  
          
      [[nodiscard]] bool can_allocate(const Sequence& seq) const;  

      void allocate(Sequence& seq); 

      [[nodiscard]] bool can_append(const Sequence& seq) const;  
      void may_append(Sequence& seq);   

      void commit(const Sequence& seq);  
      void deallocate(Sequence& seq);  

      [[nodiscard]] size_t num_free_blocks() const noexcept {return num_free_;}
      [[nodiscard]] int32_t num_blocks() const noexcept {return static_cast <int32_t>(meta_.size());}
      [[nodiscard]] int32_t ref_count(int32_t block_id) const {return meta_[block_id].ref_count;}   
      
private:  
      
  struct BlockMeta {  
    int32_t ref_count = 0;  
    int32_t prev_free = -1;   //intrusive free double linked list//  
    int32_t next_free = -1;  
    uint64_t hash = kNoHash;  
  };
  std::vector<BlockMeta> meta_; 
  std::vector<int32_t> block_tokens_;  

  std::unordered_map<uint64_t, int32_t> hash_to_block_;  
  
  int32_t free_head_ = -1; 
  int32_t free_tail_ = -1;  
  size_t num_free_ = 0; 

  void push_back_free(int32_t id);   
  void unlink_free(int32_t id);  
  int32_t take_fresh_block(); 
  
  [[nodiscard]] int32_t lookup(std::span<const int32_t> tokens, uint64_t hash) const;  
 
};



