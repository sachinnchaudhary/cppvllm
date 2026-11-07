#pragma once
#include "utils/safetensors.h"

#include <torch/torch.h>
#include <vector>  
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>

[[nodiscard]] inline at::ScalarType to_torch_dtype(STDType t) {
    switch (t) {
        case STDType::BOOL: return at::kBool;
        case STDType::U8:   return at::kByte;
        case STDType::I8:   return at::kChar;
        case STDType::I16:  return at::kShort;
        case STDType::I32:  return at::kInt;
        case STDType::I64:  return at::kLong;
        case STDType::F16:  return at::kHalf;
        case STDType::BF16: return at::kBFloat16;
        case STDType::F32:  return at::kFloat;
        case STDType::F64:  return at::kDouble;
        default:
            throw std::invalid_argument("Unsupported STDType");
    }
}

[[nodiscard]] inline torch::Tensor tensor_view(const SafeTensorsFile& f, const TensorInfo& ti) {
    const auto options = torch::TensorOptions().dtype(to_torch_dtype(ti.dtype)).device(torch::kCPU);
    if (ti.numel() == 0) return torch::empty(ti.shape, options);
    void* p = const_cast<void*>(static_cast<const void*>(f.data(ti)));
    return torch::from_blob(p, ti.shape, options);
}

[[nodiscard]] inline torch::Tensor load_tensor(const SafeTensorsFile& f, std::string_view name, torch::Device device,
                                               std::optional<at::ScalarType> dtype = std::nullopt) {
    const TensorInfo& ti = f.info(name);
    const torch::Tensor src = tensor_view(f, ti);
    at::ScalarType target = src.scalar_type();
    if (dtype.has_value() && at::isFloatingType(src.scalar_type())) target = *dtype;
    return src.to(device, target, /*non_blocking=*/false, /*copy=*/true);   
}

using WeightMap = std::unordered_map<std::string, torch::Tensor>;

[[nodiscard]] inline WeightMap load_weights(const std::string& path, torch::Device device,
                                            std::optional<at::ScalarType> dtype = std::nullopt) {
    WeightMap out;
    for (const std::string& file : list_safetensors_files(path)) {
        const SafeTensorsFile st(file);
        for (const TensorInfo& ti : st.tensors()) {
            auto [it, inserted] = out.try_emplace(ti.name);
            if (!inserted) throw std::runtime_error("tensor.h: tensor " + ti.name + " appears in multiple files");
            it->second = load_tensor(st, ti.name, device, dtype);
        }
    }
    return out;
}

template <typename T>
class StagedBuffer {
public:
    static constexpr int kSlots = 4;   // host ring size; ModelRunner uses the same number for its events

    StagedBuffer(int64_t capacity, torch::Device device, T fill_value = T{}) : capacity_(capacity) {
        const auto scalar = c10::CppTypeToScalarType<T>::value;
        hosts_.reserve(kSlots);
        for (int s = 0; s < kSlots; ++s) {
            hosts_.push_back(torch::empty({capacity}, torch::TensorOptions().dtype(scalar).pinned_memory(device.is_cuda())));
            hosts_.back().fill_(fill_value);
        }
        dev_ = torch::empty({capacity}, torch::TensorOptions().dtype(scalar).device(device));
        dev_.fill_(fill_value);
    }

    StagedBuffer(const StagedBuffer&) = delete;
    StagedBuffer& operator=(const StagedBuffer&) = delete;
    StagedBuffer(StagedBuffer&&) = default;
    StagedBuffer& operator=(StagedBuffer&&) = default;

    [[nodiscard]] int64_t capacity() const noexcept { return capacity_; }

    // Which pinned host buffer host() and upload() use. The device buffer is the same for every slot.
    void select_slot(int s) noexcept { slot_ = s % kSlots; }

    [[nodiscard]] std::span<T> host() noexcept {
        return std::span<T>(hosts_[slot_].data_ptr<T>(), static_cast<size_t>(capacity_));
    }

    torch::Tensor upload(int64_t n) {
        torch::Tensor dst = dev_.narrow(0, 0, n);
        if (n > 0) dst.copy_(hosts_[slot_].narrow(0, 0, n), /*non_blocking=*/true);
        return dst;
    }

    [[nodiscard]] torch::Tensor device_view(int64_t n) const {
        return dev_.narrow(0, 0, n);
    }

    [[nodiscard]] T* device_ptr() const { return dev_.data_ptr<T>(); }

private:
    int64_t capacity_ = 0;
    int slot_ = 0;
    std::vector<torch::Tensor> hosts_;
    torch::Tensor dev_;
};