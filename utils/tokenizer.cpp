#include "utils/tokenizer.h"
#include <algorithm>
#include <climits>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <nlohmann/json.hpp>

static const char* kPattern = R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

struct Tokenizer::Pattern {
    pcre2_code* code = nullptr;
    Pattern() {
        int err = 0;
        PCRE2_SIZE off = 0;
        code = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(kPattern), PCRE2_ZERO_TERMINATED, PCRE2_UTF | PCRE2_UCP, &err, &off, nullptr);
        pcre2_jit_compile(code, PCRE2_JIT_COMPLETE); 
    }
    ~Pattern() {
        pcre2_code_free(code);
    }
    Pattern(const Pattern&) = delete;
    Pattern& operator=(const Pattern&) = delete;
};

namespace {
constexpr std::pair<const char*, int> kSpecials[] = {
    {"<|endoftext|>", 0},
    {"<|im_start|>", 1},
    {"<|im_end|>", 2},
    {"<tool_call>", 14},
    {"</tool_call>", 15},
    {"<think>", 24},
    {"</think>", 25},
};

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::vector<std::string> load_hf_json(const std::string& path) {
    const nlohmann::json j = nlohmann::json::parse(read_file(path));
    const nlohmann::json& vocab = j.at("model").at("vocab");
    std::unordered_map<uint32_t, uint8_t> cp_to_byte;
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
        const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        cp_to_byte[printable ? static_cast<uint32_t>(b) : static_cast<uint32_t>(256 + extra++)] = static_cast<uint8_t>(b);
    }
    std::vector<std::string> id_to_bytes;
    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        const std::string& s = it.key();
        std::string bytes;
        for (size_t i = 0; i < s.size();) {
            const unsigned char c = static_cast<unsigned char>(s[i]);
            uint32_t cp;
            int n;
            if (c < 0x80) {
                cp = c;
                n = 1;
            } else if ((c >> 5) == 0x6) {
                cp = c & 0x1f;
                n = 2;
            } else if ((c >> 4) == 0xE) {
                cp = c & 0x0F;
                n = 3;
            } else {
                cp = c & 0x07;
                n = 4;
            }
            for (int k = 1; k < n; ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3f);
            auto m = cp_to_byte.find(cp);
            bytes.push_back(static_cast<char>(m->second));
            i += n;
        }
        const int64_t id = it.value().get<int64_t>();
        if (id < 0) throw std::runtime_error("tokenizer.json: negative id");
        if (static_cast<size_t>(id) >= id_to_bytes.size()) id_to_bytes.resize(static_cast<size_t>(id) + 1);
        id_to_bytes[id] = std::move(bytes);
    }
    return id_to_bytes;
}
} 

Tokenizer Tokenizer::load(const std::string& dir) {
    Tokenizer t;
    t.init(load_hf_json(dir + "/tokenizer.json"));
    return t;
}

void Tokenizer::init(std::vector<std::string> id_to_bytes) {
    id_to_bytes_ = std::move(id_to_bytes);
    ranks_.reserve(id_to_bytes_.size() * 2);
  
    for (size_t id = 0; id < id_to_bytes_.size(); ++id) {
        if (!id_to_bytes_[id].empty()) {
            ranks_.emplace(id_to_bytes_[id], static_cast<int32_t>(id));
        }
    }
    
    first_special_ = static_cast<int32_t>(id_to_bytes_.size());
    for (const auto& [name, off] : kSpecials) specials_.emplace_back(name, first_special_ + off);
    pattern_ = std::make_shared<const Pattern>();
}

void Tokenizer::merge_piece(std::string_view piece, std::vector<int32_t>& out) const {
    if (auto it = ranks_.find(piece); it != ranks_.end()) {
        out.push_back(it->second);
        return;
    }
    std::vector<uint32_t> b(piece.size() + 1); // b = boundaries between current parts
    std::iota(b.begin(), b.end(), 0u);
    while (b.size() > 2) {
        int32_t best_rank = INT32_MAX;
        size_t best_i = 0;
        for (size_t i = 0; i + 2 < b.size(); ++i) {
            auto it = ranks_.find(piece.substr(b[i], b[i + 2] - b[i]));
            if (it != ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i = i;
            }
        }
        if (best_rank == INT32_MAX) break;
        b.erase(b.begin() + static_cast<std::ptrdiff_t>(best_i) + 1);
    }
    
    for (size_t i = 0; i + 1 < b.size(); ++i) {
        auto part = piece.substr(b[i], b[i + 1] - b[i]);
        auto it = ranks_.find(part);
        out.push_back(it->second);
    }
}

void Tokenizer::encode_ordinary(std::string_view text, std::vector<int32_t>& out) const {
    pcre2_match_data* md = pcre2_match_data_create_from_pattern(pattern_->code, nullptr);
    uint32_t flags = 0;
    size_t pos = 0;
    while (pos < text.size()) {
        const int rc = pcre2_match(pattern_->code, reinterpret_cast<PCRE2_SPTR>(text.data()), text.size(), pos, flags, md, nullptr);
        if (rc == PCRE2_ERROR_NOMATCH) break;
        
        if (rc < 0) {
            pcre2_match_data_free(md);
        }
        flags = PCRE2_NO_UTF_CHECK;
        const PCRE2_SIZE* ov = pcre2_get_ovector_pointer(md);
        if (ov[1] == ov[0]) {
            pos = ov[0] + 1;
            continue;
        } 
        merge_piece(text.substr(ov[0], ov[1] - ov[0]), out);
        pos = ov[1];
    }
    pcre2_match_data_free(md);
}

std::vector<int32_t> Tokenizer::encode(std::string_view text, bool allow_special) const {
    std::vector<int32_t> out;
    out.reserve(text.size() / 3 + 2);
    size_t pos = 0;
    while (pos < text.size()) {
        size_t next = std::string_view::npos;
        const std::pair<std::string, int32_t>* hit = nullptr;
        if (allow_special) {
            for (const auto& s : specials_) {
                const size_t f = text.find(s.first, pos);
                if (f != std::string_view::npos && f < next) {
                    next = f;
                    hit = &s;
                }
            }
        }
        const size_t end = hit ? next : text.size();
        if (end > pos) encode_ordinary(text.substr(pos, end - pos), out);
        if (!hit) break;
        out.push_back(hit->second);
        pos = end + hit->first.size();
    }
    return out;
}

std::string Tokenizer::decode(std::span<const int32_t> ids, bool skip_special) const {
    std::string out;
    for (int32_t id : ids) {
        if (id < 0) throw std::out_of_range("tokenizer: negative token id");
        if (id < first_special_) {
            out += id_to_bytes_[id];
            continue;
        }
        if (skip_special) continue;
        for (const auto& s : specials_)
            if (s.second == id) {
                out += s.first;
                break;
            }
    }
    return out;
}
