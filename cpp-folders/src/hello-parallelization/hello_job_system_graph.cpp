#include <iostream>
#include <chrono>
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
 * Meaning B and C groups should work in concurrent manner and other group
 * should follow the one another in the graph
 * 
 * 
 */


#define CONCURRENCY_COUNT 4

// main task coordinator fiber, spawns a single dedicated fiber
void run_task_manager(shs::Job::AbstractJobSystem &job_system)
{
    auto last_time = std::chrono::steady_clock::now();
    int counter    = 0;

    job_system.submit({[&last_time, &counter] {

        std::cout << "STATUS : Task manager is started. " << std::endl;

        bool is_task_manager_running = true;
        while (is_task_manager_running) 
        {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = now - last_time;
            if (elapsed >= std::chrono::seconds(3))
            {
                std::cout << "STATUS : Task manager is alive...";
                last_time = now;
                ++counter;
            }
            if (counter>5)
            {
                is_task_manager_running = false;
            }
            //boost::this_fiber::sleep_for(boost::chrono::milliseconds(100));
        }

        std::cout << "STATUS : Task manager is finished. Sayunara!" << std::endl;

    }, shs::Job::PRIORITY_HIGH});
}

int main()
{

    shs::Job::AbstractJobSystem *lockless_job_system = new shs::Job::ThreadedLocklessPriorityJobSystem(CONCURRENCY_COUNT);

    bool is_engine_running    = true;
    auto first_stop_time      = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    bool did_run_task_manager = false;
    auto second_stop_time     = std::chrono::steady_clock::now() + std::chrono::seconds(60);

    while (is_engine_running)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        if (std::chrono::steady_clock::now() > first_stop_time && !did_run_task_manager)
        {
            run_task_manager(*lockless_job_system);
            did_run_task_manager = true;
        }

        if (std::chrono::steady_clock::now() > second_stop_time)
        {
            is_engine_running = false;
            lockless_job_system->is_running = false;
        }

        std::cout << "STATUS : Main thread is alive..." << std::endl;
    }

    delete lockless_job_system;

    std::cout << "STATUS : All system is shutting down... BYE!" << std::endl;

    return 0;
}
