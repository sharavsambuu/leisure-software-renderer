#include <iostream>
#include "shs_renderer.hpp"

/**
 * 
 * There will be following 5 groups of jobs which are : A, B, C, D, E
 * Computational graph is like 
 * 
 *       B
 *  A -> & -> D -> E
 *       C
 * 
 * Meaning B and C groups of should work in concurrent manner and other group
 * should follow this graph
 * 
*/

#define CONCURRENCY_COUNT 4

void send_batch_jobs(shs::AbstractJobSystem &job_system, int priority)
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

    shs::AbstractJobSystem *lockless_job_system = new shs::LocklessPriorityJobSystem(CONCURRENCY_COUNT);

    bool is_engine_running = true;

    auto first_stop_time = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    bool is_sent_second_batch = false;
    auto second_stop_time = std::chrono::steady_clock::now() + std::chrono::seconds(30);

    std::cout << ">>>>> sending first batch jobs" << std::endl;
    send_batch_jobs(*lockless_job_system, shs::JobPriority::NORMAL);

    while (is_engine_running)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        if (std::chrono::steady_clock::now() > first_stop_time && !is_sent_second_batch)
        {

            std::cout << ">>>>> sending second batch jobs to the lockless priority workers" << std::endl;
            send_batch_jobs(*lockless_job_system, shs::JobPriority::HIGH);
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