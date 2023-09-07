#include <iostream>
#include <coroutine>
#include <thread>
#include <utility>
#include <condition_variable>
#include <cstdint>
#include <list>
#include <mutex>
#include <queue>

/**
 * 
 * Reference : 
 *  - Building thread pool with coroutine
 *    https://blog.eiler.eu/posts/20210512/
 *
 */


struct task_promise;

class [[nodiscard]] task
{
public:
    using promise_type = task_promise;

    explicit task(std::coroutine_handle<task_promise> handle)
        : m_handle(handle)
    {
    }

private:
    std::coroutine_handle<task_promise> m_handle;
};

struct task_promise
{
    task get_return_object() noexcept
    {
        return task{std::coroutine_handle<task_promise>::from_promise(*this)};
    };

    std::suspend_never initial_suspend() const noexcept { return {}; }
    std::suspend_never final_suspend() const noexcept { return {}; }

    void return_void() noexcept {}

    void unhandled_exception() noexcept
    {
        std::cerr << "Unhandled exception caught...\n";
        exit(1);
    }
};

class threadpool
{
public:
    explicit threadpool(const std::size_t threadCount)
    {
        for (std::size_t i = 0; i < threadCount; ++i)
        {
            std::thread worker_thread([this]()
                                      { this->thread_loop(); });
            m_threads.push_back(std::move(worker_thread));
        }
    }

    ~threadpool()
    {
        shutdown();
    }

    auto schedule()
    {
        struct awaiter
        {
            threadpool *m_threadpool;

            constexpr bool await_ready() const noexcept { return false; }
            constexpr void await_resume() const noexcept {}
            void await_suspend(std::coroutine_handle<> coro) const noexcept
            {
                m_threadpool->enqueue_task(coro);
            }
        };
        return awaiter{this};
    }

private:
    std::list<std::thread> m_threads;

    std::mutex m_mutex;
    std::condition_variable m_cond;
    std::queue<std::coroutine_handle<>> m_coros;

    bool m_stop_thread = false;

    void thread_loop()
    {
        while (!m_stop_thread)
        {
            std::unique_lock<std::mutex> lock(m_mutex);

            while (!m_stop_thread && m_coros.size() == 0)
            {
                m_cond.wait_for(lock, std::chrono::microseconds(100));
            }

            if (m_stop_thread)
            {
                break;
            }

            auto coro = m_coros.front();
            m_coros.pop();
            lock.unlock();
            coro.resume();
        }
    }

    void enqueue_task(std::coroutine_handle<> coro) noexcept
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_coros.emplace(coro);
        m_cond.notify_one();
    }

    void shutdown()
    {
        m_stop_thread = true;
        while (m_threads.size() > 0)
        {
            std::thread &thread = m_threads.back();
            if (thread.joinable())
            {
                thread.join();
            }
            m_threads.pop_back();
        }
    }
};

task run_async_print(threadpool &pool)
{
    co_await pool.schedule();
    std::cout << "This is a hello from thread: " << std::this_thread::get_id() << "\n";
}

int main()
{
    std::cout << "The main thread id is: " << std::this_thread::get_id() << "\n";
    threadpool pool{8};
    task t = run_async_print(pool);
    std::this_thread::sleep_for(std::chrono::microseconds(1000));
}