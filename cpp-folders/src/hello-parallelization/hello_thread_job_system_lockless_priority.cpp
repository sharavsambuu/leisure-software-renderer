
#include <iostream>
#include "shs_renderer.hpp"

#define CONCURRENCY_COUNT 4

void send_batch_jobs(shs::Job::AbstractJobSystem &job_system, int priority)
{
    for (int i = 0; i < 2000; ++i)
    {
        job_system.submit({[i] {
            std::cout << "Job " << i << " started" << std::endl;
            for (int j=0; j<200; ++j)
            {
                std::cout << "Job " << i << " is working..." << std::endl;
            }
            std::cout << "Job " << i << " finished" << std::endl; 
        },
        priority
        });
    }
}

int main()
{
    /*
    shs::LocklessPriorityQueue<std::pair<int, int>> test_queue;
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

    shs::Job::AbstractJobSystem *lockless_job_system = new shs::Job::ThreadedLocklessPriorityJobSystem(CONCURRENCY_COUNT);

    bool is_engine_running = true;

    auto first_stop_time = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    bool is_sent_second_batch = false;
    auto second_stop_time = std::chrono::steady_clock::now() + std::chrono::seconds(30);

    std::cout << ">>>>> sending first batch jobs" << std::endl;
    send_batch_jobs(*lockless_job_system, shs::Job::PRIORITY_NORMAL);

    while (is_engine_running)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        if (std::chrono::steady_clock::now() > first_stop_time && !is_sent_second_batch)
        {

            std::cout << ">>>>> sending second batch jobs to the lockless priority workers" << std::endl;
            send_batch_jobs(*lockless_job_system, shs::Job::PRIORITY_HIGH);
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