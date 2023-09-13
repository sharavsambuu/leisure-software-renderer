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
 * Meaning B and C groups should work in concurrent manner and other group
 * should follow this graph, kind of counting on work stealing mode on those
 * group of workers in order to juicing many CPU cores.
 * 
 * 
 */


#define CONCURRENCY_COUNT 4

void run_task_manager(shs::AbstractJobSystem &job_system)
{
    for (int i = 0; i < 2000; ++i)
    {
        job_system.submit({[i] {
            //std::cout << "Job " << i << " started" << std::endl;
            for (int j=0; j<200; ++j)
            {
                //std::cout << "Job " << i << " is working..." << std::endl;
                boost::this_fiber::yield(); // let's be nice with each other
            }
            boost::this_fiber::yield();
            //std::cout << "Job " << i << " finished" << std::endl; 
        },
        shs::JobPriority::NORMAL
        });
    }
}

int main()
{

    shs::AbstractJobSystem *lockless_job_system = new shs::LocklessPriorityJobSystem(CONCURRENCY_COUNT);

    bool is_engine_running    = true;
    auto first_stop_time      = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    bool did_run_task_manager = false;
    auto second_stop_time     = std::chrono::steady_clock::now() + std::chrono::seconds(60);

    while (is_engine_running)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(3000));

        if (std::chrono::steady_clock::now() > first_stop_time && !did_run_task_manager)
        {

            std::cout << "STATUS : Starting a task manager... " << std::endl;
            run_task_manager(*lockless_job_system);
            did_run_task_manager = true;
        }

        if (std::chrono::steady_clock::now() > second_stop_time)
        {
            is_engine_running = false;
            lockless_job_system->is_running = false;
        }

        std::cout << "STATUS : Main thread is running..." << std::endl;
    }

    delete lockless_job_system;

    std::cout << "STATUS : All system is shutting down... BYE!" << std::endl;

    return 0;
}