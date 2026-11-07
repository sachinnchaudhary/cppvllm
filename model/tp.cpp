#include "tp.h"
#include <cuda_runtime_api.h>
#include <nccl.h>
#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

struct TPGroup::RankWorker {
    RankWorker(const EngineConfig& cfg, const ModelConfig& mcfg, torch::Device device, int rank, int world_size, ncclComm_t comm, std::shared_ptr<const WeightMap> host_weights, std::function<void()> on_failure)
        : on_failure_(std::move(on_failure)), ready_future_(ready_promise_.get_future()) {
        thread_ = std::thread([this, cfg, mcfg, device, rank, world_size, comm, host_weights = std::move(host_weights)]() mutable {
            try {
                runner_ = std::make_unique<ModelRunner>(cfg, mcfg, device, rank, world_size, comm, /*defer_graph_capture=*/true, host_weights.get());
                host_weights.reset();
                ready_promise_.set_value(runner_->num_kv_blocks());
            } catch (...) {
                try { ready_promise_.set_exception(std::current_exception()); } catch (...) {}
                return;
            }
            for (;;) {
                std::function<void()> task;
                {
                    std::unique_lock lock(mutex_);
                    cv_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
                    if (stopping_ && tasks_.empty()) break;
                    task = std::move(tasks_.front());
                    tasks_.pop_front();
                }
                task();
            }
            runner_.reset();
        });
    }
    ~RankWorker() { stop(); }
    RankWorker(const RankWorker&) = delete;
    RankWorker& operator=(const RankWorker&) = delete;

    int32_t wait_ready() { return ready_future_.get(); }

    template <class Fn>
    auto submit(Fn&& fn) {
        using Result = std::invoke_result_t<Fn, ModelRunner&>;
        auto packaged = std::make_shared<std::packaged_task<Result()>>(
            [this, callable = std::forward<Fn>(fn)]() mutable -> Result {
                try {
                    return callable(*runner_);
                } catch (...) {
                    if (on_failure_) on_failure_();
                    throw;
                }
            });
        auto future = packaged->get_future();
        {
            std::lock_guard lock(mutex_);
            if (stopping_) throw std::runtime_error("TP rank worker is stopping");
            tasks_.emplace_back([packaged] { (*packaged)(); });
        }
        cv_.notify_one();
        return future;
    }

    void stop() noexcept {
        { std::lock_guard lock(mutex_); stopping_ = true; }
        cv_.notify_one();
        if (thread_.joinable()) thread_.join();
    }

private:
    std::function<void()> on_failure_;
    std::promise<int32_t> ready_promise_;
    std::future<int32_t> ready_future_;
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::function<void()>> tasks_;
    std::unique_ptr<ModelRunner> runner_;
    bool stopping_ = false;
};

TPGroup::TPGroup(const EngineConfig& cfg, const ModelConfig& mcfg, torch::Device base_device, int tp_size) : tp_size_(tp_size) {
    const int requested_index = static_cast<int>(base_device.index());
    const int base_index = requested_index < 0 ? 0 : requested_index;

    if (tp_size == 1) {
        single_ = std::make_unique<ModelRunner>(cfg, mcfg, torch::Device(torch::kCUDA, base_index), /*tp_rank=*/0, /*tp_size=*/1, /*comm=*/nullptr);
        num_kv_blocks_ = single_->num_kv_blocks();
        return;
    }

    std::vector<int> device_ids(tp_size);
    for (int rank = 0; rank < tp_size; ++rank) device_ids[rank] = base_index + rank;

    ::setenv("NCCL_CUMEM_ENABLE", "0", /*overwrite=*/0);
    try {
        comms_.assign(tp_size, nullptr);
        torch::ScalarType s_dtype = (cfg.dtype == DType::FP16) ? torch::kHalf : 
                                    (cfg.dtype == DType::FP32) ? torch::kFloat : torch::kBFloat16;

        std::shared_ptr<const WeightMap> host_weights = std::make_shared<WeightMap>(::load_weights(cfg.model_path, torch::Device(torch::kCPU), s_dtype));
        workers_.reserve(tp_size);
        for (int rank = 0; rank < tp_size; ++rank) {
            workers_.push_back(std::make_unique<RankWorker>(cfg, mcfg, torch::Device(torch::kCUDA, device_ids[rank]), rank, tp_size, comms_[rank], host_weights, [this] { abort_communicators(); }));
        }
        num_kv_blocks_ = std::numeric_limits<int32_t>::max();
        std::exception_ptr load_failure;
        for (auto& worker : workers_) {
            try { num_kv_blocks_ = std::min(num_kv_blocks_, worker->wait_ready()); } 
            catch (...) { if (!load_failure) load_failure = std::current_exception(); }
        }
        if (load_failure) std::rethrow_exception(load_failure);
        host_weights.reset();

        if (!cfg.enforce_eager) {
            std::vector<std::future<void>> tasks;
            tasks.reserve(workers_.size());
            for (auto& worker : workers_) {
                tasks.push_back(worker->submit([](ModelRunner& runner) { runner.capture_graphs_if_enabled(); }));
            }
            for (auto& f : tasks) f.get();
        }
    } catch (...) {
        failed_ = true;
        abort_communicators();
        workers_.clear();
        destroy_communicators();
        throw;
    }
}

TPGroup::~TPGroup() {
    if (!failed_) {
        try { while (in_flight_ > 0) (void)collect(); } 
        catch (...) { failed_ = true; abort_communicators(); }
    } else {
        abort_communicators();
    }
    workers_.clear();
    single_.reset();
    destroy_communicators();
}

void TPGroup::abort_communicators() noexcept {
    failed_ = true;
    std::lock_guard lock(abort_mutex_);
    if (comms_aborted_) return;
    comms_aborted_ = true;
    for (ncclComm_t& comm : comms_) {
        if (comm) { (void)ncclCommAbort(comm); comm = nullptr; }
    }
}

void TPGroup::destroy_communicators() noexcept {
    std::lock_guard lock(abort_mutex_);
    for (ncclComm_t& comm : comms_) {
        if (comm) { (void)ncclCommDestroy(comm); comm = nullptr; }
    }
    comms_.clear();
}

void TPGroup::launch(const ScheduledBatch& batch) {
    if (single_) {
        single_->launch(batch);
        ++in_flight_;
        return;
    }
    std::vector<std::future<void>> tasks;
    tasks.reserve(workers_.size());
    for (auto& worker : workers_) {
        tasks.push_back(worker->submit([&batch](ModelRunner& runner) { runner.launch(batch); }));
    }
    for (auto& f : tasks) f.get();
    ++in_flight_;
}

const std::vector<int32_t>& TPGroup::collect() {
    if (single_) {
        const std::vector<int32_t>& out = single_->collect();
        --in_flight_;
        return out;
    }
    std::vector<std::future<const std::vector<int32_t>*>> tasks;
    tasks.reserve(workers_.size());
    for (auto& worker : workers_) {
        tasks.push_back(worker->submit([](ModelRunner& runner) { return &runner.collect(); }));
    }
    std::vector<const std::vector<int32_t>*> results;
    results.reserve(tasks.size());
    for (auto& f : tasks) results.push_back(f.get()); 
    
    --in_flight_;
    return *results.front();
}

const std::vector<int32_t>& TPGroup::run(const ScheduledBatch& batch) {
    launch(batch);
    return collect();
}
