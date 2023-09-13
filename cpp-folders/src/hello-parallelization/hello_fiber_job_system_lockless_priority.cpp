
#include <iostream>
#include <vector>
#include <queue>
#include <atomic>
#include <functional>
#include <optional>
#include <boost/fiber/all.hpp>
#include <boost/fiber/fiber.hpp>
#include <boost/fiber/future.hpp>
#include <boost/thread/thread.hpp>
#include <boost/chrono.hpp>

#define CONCURRENCY_COUNT 4

#define PRIORITY_LOW      5
#define PRIORITY_NORMAL   15
#define PRIOTITY_HIGH     30

class AbstractJobSystem
{
public:
    virtual ~AbstractJobSystem() {};
    virtual void submit(std::pair<std::function<void()>, int> task) = 0;
    bool is_running = true;
};

template <typename T>
class LocklessPriorityQueue
{
public:
    LocklessPriorityQueue() : head_(nullptr) {}

    void push(const T &value)
    {
        Node *new_node = new Node(value);
        new_node->next = head_.load(std::memory_order_relaxed);

        while (!head_.compare_exchange_weak(new_node->next, new_node,
                                            std::memory_order_release,
                                            std::memory_order_relaxed))
        {
        }
    }

    std::optional<T> pop()
    {
        Node *old_head = head_.load(std::memory_order_acquire);

        while (old_head && !head_.compare_exchange_weak(old_head, old_head->next,
                                                        std::memory_order_relaxed,
                                                        std::memory_order_relaxed))
        {
        }

        if (old_head)
        {
            T value = old_head->data;
            delete old_head;
            return value;
        }
        else
        {
            return std::nullopt;
        }
    }
    long count()
    {
        Node *current = head_.load(std::memory_order_relaxed);
        long count = 0;

        while (current)
        {
            count++;
            current = current->next;
        }
        return count;
    }

private:
    struct Node
    {
        T data;
        Node *next;

        Node(const T &val) : data(val), next(nullptr) {}
    };

    std::atomic<Node *> head_;
};

class LocklessPriorityJobSystem : public AbstractJobSystem
{
public:
    LocklessPriorityJobSystem(int concurrency_count)
    {
        std::cout << "Lockless priority job system is starting..." << std::endl;

        this->concurrency_count = concurrency_count;
        this->workers.reserve(this->concurrency_count);

        for (int i = 0; i < this->concurrency_count; ++i)
        {
            this->workers[i] = boost::thread([this, i] {
                boost::fibers::use_scheduling_algorithm<boost::fibers::algo::work_stealing>(this->concurrency_count);

                while(this->is_running)
                {
                    auto task_priority = this->job_queue.pop();
                    if (task_priority.has_value())
                    {
                        auto [task, priority] = task_priority.value();
                        boost::fibers::fiber(task).join();
                    }

                } 
            });
        }
    }
    ~LocklessPriorityJobSystem()
    {
        for (auto &worker : this->workers)
        {
            worker.join();
        }
        std::cout << "Lockless job system is shutting down..." << std::endl;
    }
    void submit(std::pair<std::function<void()>, int> task) override
    {
        this->job_queue.push(task);
        //long count = this->job_queue.count();
        //std::cout << "total : " << count << std::endl;
    }

private:
    int concurrency_count;
    std::vector<boost::thread> workers;
    LocklessPriorityQueue<std::pair<std::function<void()>, int>> job_queue;
};

void send_batch_jobs(AbstractJobSystem &job_system, int priority)
{
    for (int i = 0; i < 2000; ++i)
    {
        job_system.submit({[i] {
            std::cout << "Job " << i << " started" << std::endl;
            for (int j=0; j<200; ++j)
            {
                std::cout << "Job " << i << " is working..." << std::endl;
                boost::this_fiber::yield(); // let's be nice with each other
            }
            boost::this_fiber::yield();
            std::cout << "Job " << i << " finished" << std::endl; 
        },
        priority
        });
    }
}

int main()
{
    /*
    LocklessPriorityQueue<std::pair<int, int>> test_queue;
    test_queue.push({ 55,  2});
    test_queue.push({ 33,  1});
    test_queue.push({153,  3});
    test_queue.push({413,  3});
    test_queue.push({  1, 13});
    std::cout << "Queue Size: " << test_queue.count() << std::endl;
    for (int i = 0; i < 10; ++i)
    {
        auto element = test_queue.pop();
        if (element.has_value())
        {
            auto [value, priority] = element.value();
            std::cout << "priority : " << priority << ", value : " << value << std::endl;
        }
        else
        {
            std::cout << "queue is empty." << std::endl;
        }
    }
    */

    AbstractJobSystem *lockless_job_system = new LocklessPriorityJobSystem(CONCURRENCY_COUNT);

    bool is_engine_running = true;

    auto first_stop_time = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    bool is_sent_second_batch = false;
    auto second_stop_time = std::chrono::steady_clock::now() + std::chrono::seconds(30);

    std::cout << ">>>>> sending first batch jobs" << std::endl;
    send_batch_jobs(*lockless_job_system, PRIORITY_LOW);

    while (is_engine_running)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        if (std::chrono::steady_clock::now() > first_stop_time && !is_sent_second_batch)
        {

            std::cout << ">>>>> sending second batch jobs to the lockless priority workers" << std::endl;
            send_batch_jobs(*lockless_job_system, PRIOTITY_HIGH);
            is_sent_second_batch = true;
        }

        if (std::chrono::steady_clock::now() > second_stop_time)
        {
            is_engine_running = false;
            lockless_job_system->is_running = false;
        }
    }

    delete lockless_job_system;

    std::cout << "system is shutting down... bye!" << std::endl;

    return 0;
}