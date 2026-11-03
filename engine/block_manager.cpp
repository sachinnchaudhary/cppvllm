#include "block_manager.h"  
#include <algorithm>  
#include <cassert>  


namespace { 
   
constexpr int32_t kBS = Sequence::BLOCK_SIZE;  

uint64_t hash_tokens(std::span<const int32_t> tokens, uint64_t prefix) noexcept { 
    // FNV-1a over the tokens + a splitmix64 finalizer to spread the bits.
    uint64_t h =  0xcbf29ce484222325ULL ^ (prefix * 0x9E3779B97F4A7C15ULL);  

    for (int32_t t : tokens) {  
        h ^= static_cast<uint32_t>(t);
        h *= 0x100000001b3ULL;
    }
    h ^= h >> 30; h *= 0xbf58476d1ce4e5b9ULL;
    h ^= h >> 27; h *= 0x94d049bb133111ebULL;
    h ^= h >> 31;
    return h == kNoHash ? 1 : h;   // 0 is reserved for "no hash"    
}

size_t max_cacheable_blocks(const Sequence& seq) noexcept {return (seq.size() - 1) / kBS;}   

} 

BlockManager::BlockManager(int32_t num_blocks)  
  : meta_(num_blocks), block_tokens_(static_cast<size_t>(num_blocks) * kBS, 0) { 
    hash_to_block_.reserve(num_blocks);  
    for (int32_t id = 0; id < num_blocks; ++id) push_back_free(id);  
}  

//intrusive free list.
void BlockManager::push_back_free(int32_t id) {  
    BlockMeta& b = meta_[id];  
    b.prev_free = free_tail_;  
    b.next_free = -1;  
    if (free_tail_ != -1) meta_[free_tail_].next_free = id;
    else                  free_head_ = id;
    free_tail_ = id;  
    ++num_free_;
}

//removing id with O(1) 
void BlockManager::unlink_free(int32_t id) {  
    BlockMeta& b = meta_[id];  
    const int32_t p = b.prev_free;  
    const int32_t n = b.next_free;  
    if (p != -1) meta_[p].next_free = n; else free_head_ = n;  
    if (n != -1) meta_[n].prev_free = p; else free_tail_ = p;  
    b.prev_free = b.next_free = -1; 
    --num_free_; 
}

//LRF(least recently freed) block. 
int32_t BlockManager::take_fresh_block() {
   const int32_t id = free_head_;
   unlink_free(id);

   BlockMeta& b = meta_[id];
   if (b.hash != kNoHash) {
     auto it = hash_to_block_.find(b.hash);
     if (it != hash_to_block_.end() && it->second == id) hash_to_block_.erase(it);   // keyed by hash, not id
     b.hash = kNoHash;
   }
   b.ref_count = 1;
   return id;
}

int32_t BlockManager::lookup(std::span<const int32_t> tokens, uint64_t hash) const {  
   auto it = hash_to_block_.find(hash);  
   if (it == hash_to_block_.end()) return -1; 
   const int32_t id = it->second; 
   const int32_t* stored = block_tokens_.data() + static_cast<size_t>(id) * kBS;  
   if (tokens.size() != static_cast<size_t>(kBS) || !std::equal(tokens.begin(), tokens.end(), stored)) return -1; 
   return id; 
}

bool BlockManager::can_allocate(const Sequence& seq) const {  
    size_t need = seq.num_blocks();  
    uint64_t prefix = kNoHash;  
    const size_t limit = max_cacheable_blocks(seq);  
    for (size_t i = 0; i < limit; ++i) { 
          const auto tokens = seq.block(i);  
          const uint64_t h = hash_tokens(tokens, prefix);  
          const int32_t  id = lookup(tokens, h); 
          if (id < 0) break;  
          if (meta_[id].ref_count > 0) --need;  
          prefix = h;  
        }
    return num_free_ >= need;
} 

void BlockManager::allocate(Sequence& seq) {  
    assert(seq.block_table.empty());  
    uint64_t prefix =kNoHash;  
    size_t hits = 0; 
    const size_t limit = max_cacheable_blocks(seq);  

    for(size_t i = 0; i < limit; ++i) { 
      const auto tokens = seq.block(i); 
      const uint64_t h = hash_tokens(tokens, prefix);  
      const int32_t id = lookup(tokens, h);  
      if (id < 0) break;  
      if (meta_[id].ref_count == 0) unlink_free(id);  
      ++meta_[id].ref_count;  
      seq.block_table.push_back(id);  
      prefix = h;  
      ++hits;  
    }
    seq.num_cached_tokens = static_cast<uint32_t>(hits * kBS);  
    for (size_t i = hits; i < seq.num_blocks(); ++i) { 
        seq.block_table.push_back(take_fresh_block()); 
    }
}

bool BlockManager::can_append(const Sequence& seq) const {  
     assert(seq.block_table.size() <= seq.num_blocks());  
     return num_free_ >= seq.num_blocks() - seq.block_table.size();  
}

void BlockManager::may_append(Sequence& seq){ 
    while (seq.block_table.size() < seq.num_blocks())  
      seq.block_table.push_back(take_fresh_block());
}

void BlockManager::commit(const Sequence& seq) { 
    const size_t full = seq.num_cached_tokens / kBS;  
    for (size_t i=0; i < full; ++i) {
       const int32_t id = seq.block_table[i];  
       BlockMeta& b = meta_[id];  
       if (b.hash != kNoHash) continue;  

       const uint64_t prefix = (i == 0) ? kNoHash : meta_[seq.block_table[i-1]].hash;  
       const auto tokens = seq.block(i);  
       const uint64_t h = hash_tokens(tokens, prefix);  
       
       std::copy(tokens.begin(), tokens.end(), block_tokens_.begin() + static_cast<size_t>(id) * kBS); 
       b.hash = h; 
       hash_to_block_.try_emplace(h, id);  
    }
}

void BlockManager::deallocate(Sequence& seq) {  
    for (auto it = seq.block_table.rbegin(); it != seq.block_table.rend(); ++it) { 
        BlockMeta& b= meta_[*it];  
        assert(b.ref_count > 0);  
        if (--b.ref_count == 0) push_back_free(*it);  
    } 
    seq.block_table.clear(); 
    seq.num_cached_tokens = 0;  
}





