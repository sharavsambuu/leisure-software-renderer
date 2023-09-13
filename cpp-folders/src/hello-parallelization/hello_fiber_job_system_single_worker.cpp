#include <iostream>
#include "shs_renderer.hpp"


class FiberJobSystemSingle : public shs::Job::AbstractJobSystem
{
public:
    FiberJobSystemSingle()
    {
        this->worker = boost::thread([this] {
            while (this->is_running)
            {
                std::function<void()> task;

            }
        });
        std::cout << "STATUS : Fiber job system with single worker is started." << std::endl;
    }
    ~FiberJobSystemSingle()
    {
        std::cout << "STATUS : Fiber job system with single worker is shutting down..." << std::endl;
        this->worker.join();
    }
    void submit(std::pair<std::function<void()>, int> task) override
    {
    }

private:
    boost::thread worker;
};

void send_batch_jobs(shs::Job::AbstractJobSystem &job_system)
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
        }, shs::Job::PRIORITY_NORMAL});
    }
}

int main()
{

    shs::Job::AbstractJobSystem *job_system = new FiberJobSystemSingle();

    bool is_engine_running = true;

    auto first_stop_time = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    bool is_sent_second_batch = false;
    auto second_stop_time = std::chrono::steady_clock::now() + std::chrono::seconds(20);

    std::cout << ">>>>> sending first batch jobs" << std::endl;
    send_batch_jobs(*job_system);

    while (is_engine_running)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        if (std::chrono::steady_clock::now() > first_stop_time && !is_sent_second_batch)
        {

            std::cout << ">>>>> sending second batch jobs" << std::endl;
            send_batch_jobs(*job_system);
            is_sent_second_batch = true;
        }

        if (std::chrono::steady_clock::now() > second_stop_time)
        {
            is_engine_running      = false;
            job_system->is_running = false;
        }
    }

    delete job_system;

    std::cout << "system is shutting down... bye!" << std::endl;

    return 0;
}