#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: wait_group.hpp
    МОДУЛЬ: job
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн job модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <atomic>
#include <condition_variable>
#include <mutex>

namespace shs
{
    class WaitGroup
    {
    public:
        void add(int n = 1)
        {
            count_.fetch_add(n, std::memory_order_relaxed);
        }

        void done()
        {
            if (count_.fetch_sub(1, std::memory_order_acq_rel) == 1)
            {
                std::lock_guard<std::mutex> lock(mtx_);
                cv_.notify_all();
            }
        }

        void wait()
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait(lock, [&]() { return count_.load(std::memory_order_acquire) == 0; });
        }

    private:
        std::atomic<int> count_{0};
        std::mutex mtx_{};
        std::condition_variable cv_{};
    };
}

