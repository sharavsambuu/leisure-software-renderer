#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: thread_pool_job_system.hpp
    МОДУЛЬ: job
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн job модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "shs/job/job_system.hpp"

namespace shs
{
    class ThreadPoolJobSystem final : public IJobSystem
    {
    public:
        explicit ThreadPoolJobSystem(size_t worker_count)
            : stop_(false), active_(0)
        {
            const size_t n = worker_count == 0 ? 1 : worker_count;
            workers_.reserve(n);
            for (size_t i = 0; i < n; ++i)
            {
                workers_.emplace_back([this]() { worker_loop(); });
            }
        }

        ~ThreadPoolJobSystem() override
        {
            {
                std::lock_guard<std::mutex> lock(mtx_);
                stop_ = true;
            }
            cv_.notify_all();
            for (auto& w : workers_)
            {
                if (w.joinable()) w.join();
            }
        }

        void enqueue(std::function<void()> job) override
        {
            {
                std::lock_guard<std::mutex> lock(mtx_);
                jobs_.push(std::move(job));
            }
            cv_.notify_one();
        }

        void wait_idle() override
        {
            std::unique_lock<std::mutex> lock(mtx_);
            idle_cv_.wait(lock, [this]() {
                return jobs_.empty() && active_.load(std::memory_order_acquire) == 0;
            });
        }

        size_t worker_count() const override
        {
            return workers_.size();
        }

    private:
        void worker_loop()
        {
            while (true)
            {
                std::function<void()> job{};
                {
                    std::unique_lock<std::mutex> lock(mtx_);
                    cv_.wait(lock, [this]() { return stop_ || !jobs_.empty(); });
                    if (stop_ && jobs_.empty()) return;
                    job = std::move(jobs_.front());
                    jobs_.pop();
                    active_.fetch_add(1, std::memory_order_relaxed);
                }

                job();

                {
                    std::lock_guard<std::mutex> lock(mtx_);
                    active_.fetch_sub(1, std::memory_order_relaxed);
                    if (jobs_.empty() && active_.load(std::memory_order_relaxed) == 0)
                    {
                        idle_cv_.notify_all();
                    }
                }
            }
        }

        std::vector<std::thread> workers_{};
        std::queue<std::function<void()>> jobs_{};
        mutable std::mutex mtx_{};
        std::condition_variable cv_{};
        std::condition_variable idle_cv_{};
        bool stop_ = false;
        std::atomic<int> active_{0};
    };
}

