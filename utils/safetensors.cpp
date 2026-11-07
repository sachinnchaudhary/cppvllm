#include "utils/safetensors.h"
#include <nlohmann/json.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace {
constexpr uint64_t kMaxHeaderBytes = 100ull * 1024 * 1024;

struct DTypeRow { const char* name; STDType dtype; size_t size; };
constexpr DTypeRow kDTypes[] = {
    {"BOOL", STDType::BOOL, 1}, {"U8", STDType::U8, 1},   {"I8", STDType::I8, 1},
    {"I16", STDType::I16, 2},   {"U16", STDType::U16, 2}, {"F16", STDType::F16, 2}, {"BF16", STDType::BF16, 2},
    {"I32", STDType::I32, 4},   {"U32", STDType::U32, 4}, {"F32", STDType::F32, 4},
    {"I64", STDType::I64, 8},   {"U64", STDType::U64, 8}, {"F64", STDType::F64, 8},
};

}  

SafeTensorsFile::SafeTensorsFile(const std::string& path) : path_(path) {
    const int fd = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
    
    struct stat sb;
    ::fstat(fd, &sb);
    map_size_ = static_cast<size_t>(sb.st_size);
    
    void* p = ::mmap(nullptr, map_size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    map_ = p;
    try {
        parse_header();
    } catch (...) {
        release();
        throw;
    }
}

SafeTensorsFile::~SafeTensorsFile() {
    release();
}

void SafeTensorsFile::release() noexcept {
    if (map_) ::munmap(map_, map_size_);
    map_ = nullptr;
}

bool SafeTensorsFile::contains(std::string_view name) const noexcept {
    return index_.find(name) != index_.end();
}

const TensorInfo& SafeTensorsFile::info(std::string_view name) const {
    auto it = index_.find(name);
    return tensors_[it->second];
}

const std::byte* SafeTensorsFile::data(const TensorInfo& t) const noexcept {
    return reinterpret_cast<const std::byte*>(static_cast<const uint8_t*>(map_) + data_offset_ + t.begin);
}

void SafeTensorsFile::parse_header() {
    const auto* base = static_cast<const uint8_t*>(map_);
    uint64_t header_len = 0;
    std::memcpy(&header_len, base, 8);
    data_offset_ = 8 + static_cast<size_t>(header_len);

    nlohmann::json j;
    const char* h = reinterpret_cast<const char*>(base + 8);
    j = nlohmann::json::parse(h, h + header_len);

    struct Raw {
        std::string name;
        STDType dtype;
        std::vector<int64_t> shape;
        uint64_t begin, end;
    };

    std::vector<Raw> raws;
    raws.reserve(j.size());

    for (auto it = j.begin(); it != j.end(); ++it) {
        const std::string& name = it.key();
        if (name == "__metadata__") continue;

        const nlohmann::json& e = it.value();
        const std::string dt = e.at("dtype").get<std::string>();
        const auto* row = std::find_if(std::begin(kDTypes), std::end(kDTypes), [&](const DTypeRow& r) { 
            return dt == r.name; 
        });

        Raw r;
        r.name = name;
        r.dtype = row->dtype;
        r.shape = e.at("shape").get<std::vector<int64_t>>();
        
        const auto offs = e.at("data_offsets").get<std::vector<uint64_t>>();
        r.begin = offs[0];
        r.end = offs[1];

        raws.push_back(std::move(r));
    }

    tensors_.reserve(raws.size());
    for (Raw& r : raws) {
        index_.emplace(r.name, tensors_.size());
        tensors_.push_back(TensorInfo{std::move(r.name), r.dtype, std::move(r.shape), r.begin, r.end});
    }
}

std::vector<std::string> list_safetensors_files(const std::string& path) {
    namespace fs = std::filesystem;
    std::vector<std::string> out;
    if (fs::is_regular_file(path)) {
        out.push_back(path);
        return out;
    }
    for (const auto& e : fs::directory_iterator(path)) {
        if (e.is_regular_file() && e.path().extension() == ".safetensors") {
            out.push_back(e.path().string());
        }
    }
    if (out.empty()) throw std::runtime_error("no .safetensors files in " + path);
    std::sort(out.begin(), out.end());
    return out;
}
