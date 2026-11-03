#pragma once  
#include <functional>
#include <utility>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector> 


// Qwen3 byte-level BPE tokenizer (reads tokenizer.json). Qwen has no BOS token: prompts start directly with the text.
class Tokenizer {  

public:  
  static Tokenizer load(const std::string& model_dir);  
  [[nodiscard]] std::vector<int32_t> encode(std::string_view text, bool allow_special = false) const;  
  [[nodiscard]] std::string decode(std::span<const int32_t> ids, bool skip_special = false) const;   
  
  [[nodiscard]] int32_t eos_id() const noexcept {return first_special_ + 2;}   // <|im_end|>, ends an assistant turn
  [[nodiscard]] size_t num_ranks() const noexcept {return id_to_bytes_.size();}

private:  
 Tokenizer() = default;  
 struct Pattern;  

 struct SvHash {
    using is_transparent = void;  
    size_t operator()(std::string_view s) const noexcept {return std::hash<std::string_view>{}(s);}
 };  
 void init(std::vector<std::string> id_to_bytes_);  
 void encode_ordinary(std::string_view text, std::vector<int32_t>& out) const;  
 void merge_piece(std::string_view piece, std::vector<int32_t>& out) const;  

 std::unordered_map<std::string, int32_t, SvHash, std::equal_to<>> ranks_;  
 std::vector<std::string> id_to_bytes_;  
 int32_t first_special_ = 0;  
 std::vector<std::pair<std::string, int32_t>> specials_;  
 std::shared_ptr<const Pattern> pattern_;

};