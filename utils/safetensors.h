#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

enum class STDType : uint8_t { BOOL, U8, I8, I16, U16, I32, U32, I64, U64, F16, BF16, F32, F64 };

[[nodiscard]] size_t      sdtype_size(STDType t) noexcept;
[[nodiscard]] const char* sdtype_name(STDType t) noexcept;

struct TensorInfo {
    std::string name;
    STDType dtype = STDType::F32;
    std::vector<int64_t> shape;
    uint64_t begin = 0;   // offsets are relative to the start of the data region
    uint64_t end = 0;

    [[nodiscard]] int64_t numel() const noexcept { int64_t n = 1; for (int64_t d : shape) n *= d; return n; }
    [[nodiscard]] size_t nbytes() const noexcept { return static_cast<size_t>(end - begin); }
};

class SafeTensorsFile {
public:
    explicit SafeTensorsFile(const std::string& path);
    ~SafeTensorsFile();
    SafeTensorsFile(const SafeTensorsFile&) = delete;
    SafeTensorsFile& operator=(const SafeTensorsFile&) = delete;

    [[nodiscard]] const std::string& path() const noexcept { return path_; }
    [[nodiscard]] const std::vector<TensorInfo>& tensors() const noexcept { return tensors_; }
    [[nodiscard]] size_t file_size() const noexcept { return map_size_; }

    [[nodiscard]] bool contains(std::string_view name) const noexcept;
    [[nodiscard]] const TensorInfo& info(std::string_view name) const;
    [[nodiscard]] const std::byte* data(const TensorInfo& t) const noexcept;
    [[nodiscard]] const std::byte* data(std::string_view name) const { return data(info(name)); }

private:
    struct StringHash {
        using is_transparent = void;
        size_t operator()(std::string_view s) const noexcept { return std::hash<std::string_view>{}(s); }
    };
    void parse_header();
    void release() noexcept;

    std::string path_;
    void* map_ = nullptr;
    size_t map_size_ = 0;
    size_t data_offset_ = 0;
    std::vector<TensorInfo> tensors_;
    std::unordered_map<std::string, size_t, StringHash, std::equal_to<>> index_;
};

[[nodiscard]] std::vector<std::string> list_safetensors_files(const std::string& path);