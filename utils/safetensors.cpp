#include "utils/safetensors.h"

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

#include <nlohmann/json.hpp>

static_assert(std::endian::native == std::endian::little, "safetensors assumes a little-endian host");

namespace {
constexpr uint64_t kMaxHeaderBytes = 100ull * 1024 * 1024;

struct DTypeRow { const char* name; STDType dtype; size_t size; };
constexpr DTypeRow kDTypes[] = {
    {"BOOL", STDType::BOOL, 1}, {"U8", STDType::U8, 1},   {"I8", STDType::I8, 1},
    {"I16", STDType::I16, 2},   {"U16", STDType::U16, 2}, {"F16", STDType::F16, 2}, {"BF16", STDType::BF16, 2},
    {"I32", STDType::I32, 4},   {"U32", STDType::U32, 4}, {"F32", STDType::F32, 4},
    {"I64", STDType::I64, 8},   {"U64", STDType::U64, 8}, {"F64", STDType::F64, 8},
};

[[noreturn]] void fail(const std::string& path, const std::string& why) {
    throw std::runtime_error("safetensors: " + path + ": " + why);
}
}  // namespace

size_t sdtype_size(STDType t) noexcept {
    for (const auto& r : kDTypes) if (r.dtype == t) return r.size;
    return 0;
}
const char* sdtype_name(STDType t) noexcept {
    for (const auto& r : kDTypes) if (r.dtype == t) return r.name;
    return "?";
}

SafeTensorsFile::SafeTensorsFile(const std::string& path) : path_(path) {
    const int fd = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) fail(path_, std::string("cannot open: ") + std::strerror(errno));

    struct stat sb;
    if (::fstat(fd, &sb) != 0) { ::close(fd); fail(path_, std::string("fstat failed: ") + std::strerror(errno)); }
    map_size_ = static_cast<size_t>(sb.st_size);
    if (map_size_ < 8) { ::close(fd); fail(path_, "file is smaller than the 8-byte length prefix"); }

    void* p = ::mmap(nullptr, map_size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) fail(path_, std::string("mmap failed: ") + std::strerror(errno));
    map_ = p;

    try { parse_header(); } catch (...) { release(); throw; }
}

SafeTensorsFile::~SafeTensorsFile() { release(); }

void SafeTensorsFile::release() noexcept {
    if (map_) ::munmap(map_, map_size_);
    map_ = nullptr;
}

bool SafeTensorsFile::contains(std::string_view name) const noexcept { return index_.find(name) != index_.end(); }

const TensorInfo& SafeTensorsFile::info(std::string_view name) const {
    auto it = index_.find(name);
    if (it == index_.end()) fail(path_, "no tensor named '" + std::string(name) + "'");
    return tensors_[it->second];
}

const std::byte* SafeTensorsFile::data(const TensorInfo& t) const noexcept {
    return reinterpret_cast<const std::byte*>(static_cast<const uint8_t*>(map_) + data_offset_ + t.begin);
}

void SafeTensorsFile::parse_header() {
    const auto* base = static_cast<const uint8_t*>(map_);
    uint64_t header_len = 0;
    std::memcpy(&header_len, base, 8);
    if (header_len > kMaxHeaderBytes) fail(path_, "header length " + std::to_string(header_len) + " exceeds the 100 MB limit");
    if (header_len > map_size_ - 8) fail(path_, "header length runs past the end of the file (truncated file?)");

    data_offset_ = 8 + static_cast<size_t>(header_len);
    const uint64_t data_len = map_size_ - data_offset_;

    nlohmann::json j;
    try {
        const char* h = reinterpret_cast<const char*>(base + 8);
        j = nlohmann::json::parse(h, h + header_len);
    } catch (const nlohmann::json::exception& e) {
        fail(path_, std::string("header is not valid JSON: ") + e.what());
    }
    if (!j.is_object()) fail(path_, "header JSON is not an object");

    struct Raw { std::string name; STDType dtype; std::vector<int64_t> shape; uint64_t begin, end; };
    std::vector<Raw> raws;
    raws.reserve(j.size());

    for (auto it = j.begin(); it != j.end(); ++it) {
        const std::string& name = it.key();
        if (name == "__metadata__") continue;
        const nlohmann::json& e = it.value();
        try {
            const std::string dt = e.at("dtype").get<std::string>();
            const auto* row = std::find_if(std::begin(kDTypes), std::end(kDTypes),
                                           [&](const DTypeRow& r) { return dt == r.name; });
            if (row == std::end(kDTypes)) fail(path_, "tensor '" + name + "' has unknown dtype '" + dt + "'");

            Raw r;
            r.name = name;
            r.dtype = row->dtype;
            r.shape = e.at("shape").get<std::vector<int64_t>>();
            const auto offs = e.at("data_offsets").get<std::vector<uint64_t>>();
            if (offs.size() != 2) fail(path_, "tensor '" + name + "': data_offsets must have 2 entries");
            r.begin = offs[0];
            r.end = offs[1];
            if (r.begin > r.end) fail(path_, "tensor '" + name + "': data_offsets are reversed");
            if (r.end > data_len) fail(path_, "tensor '" + name + "': data_offsets run past the end of the file");

            uint64_t expect = row->size;
            for (int64_t d : r.shape) {
                if (d < 0) fail(path_, "tensor '" + name + "' has a negative dimension");
                if (d != 0 && expect > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(d))
                    fail(path_, "tensor '" + name + "': shape overflows");
                expect *= static_cast<uint64_t>(d);
            }
            if (r.end - r.begin != expect)
                fail(path_, "tensor '" + name + "': shape and dtype imply " + std::to_string(expect) +
                            " bytes but data_offsets span " + std::to_string(r.end - r.begin));
            raws.push_back(std::move(r));
        } catch (const nlohmann::json::exception& ex) {
            fail(path_, "tensor '" + name + "' has a malformed entry: " + ex.what());
        }
    }

    std::sort(raws.begin(), raws.end(), [](const Raw& a, const Raw& b) {
        return a.begin != b.begin ? a.begin < b.begin : a.end < b.end;
    });

    uint64_t cursor = 0;
    for (const Raw& r : raws) {
        if (r.begin != cursor) fail(path_, "tensor '" + r.name + "': offsets leave a gap or overlap the previous tensor");
        cursor = r.end;
    }
    if (cursor != data_len) fail(path_, "tensor data does not cover the whole data region");

    tensors_.reserve(raws.size());
    for (Raw& r : raws) {
        const size_t esz = sdtype_size(r.dtype);
        if (reinterpret_cast<uintptr_t>(base + data_offset_ + r.begin) % esz != 0)
            fail(path_, "tensor '" + r.name + "' is not aligned to its element size");
        index_.emplace(r.name, tensors_.size());
        tensors_.push_back(TensorInfo{std::move(r.name), r.dtype, std::move(r.shape), r.begin, r.end});
    }
}

std::vector<std::string> list_safetensors_files(const std::string& path) {
    namespace fs = std::filesystem;
    std::vector<std::string> out;
    if (fs::is_regular_file(path)) { out.push_back(path); return out; }
    for (const auto& e : fs::directory_iterator(path))
        if (e.is_regular_file() && e.path().extension() == ".safetensors") out.push_back(e.path().string());
    if (out.empty()) throw std::runtime_error("no .safetensors files in " + path);
    std::sort(out.begin(), out.end());
    return out;
}