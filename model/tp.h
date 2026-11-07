#pragma once

#include "engine/model_runner.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

class TPGroup {
public:
    TPGroup(const EngineConfig& cfg, const ModelConfig& mcfg, torch::Device base_device,
            int tp_size = 1);
    ~TPGroup();

    TPGroup(const TPGroup&) = delete;
    TPGroup& operator=(const TPGroup&) = delete;

    void launch(const ScheduledBatch& batch);
    [[nodiscard]] const std::vector<int32_t>& collect();
    [[nodiscard]] const std::vector<int32_t>& run(const ScheduledBatch& batch);

    [[nodiscard]] int32_t num_kv_blocks() const noexcept { return num_kv_blocks_; }
    [[nodiscard]] int in_flight() const noexcept { return in_flight_; }
    [[nodiscard]] int tp_size() const noexcept { return tp_size_; }

private:
    struct RankWorker;

    void abort_communicators() noexcept;       
    void destroy_communicators() noexcept;

    int tp_size_ = 1;
    std::vector<ncclComm_t> comms_;
    std::unique_ptr<ModelRunner> single_;     
    std::vector<std::unique_ptr<RankWorker>> workers_;   
    int32_t num_kv_blocks_ = 0;
    int in_flight_ = 0;
    std::atomic<bool> failed_{false};
    std::mutex abort_mutex_;
    bool comms_aborted_ = false;               
};