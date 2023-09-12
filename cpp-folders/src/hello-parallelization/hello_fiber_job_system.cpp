#include <iostream>
#include <vector>
#include <boost/fiber/all.hpp>
#include <boost/fiber/fiber.hpp>
#include <boost/fiber/future.hpp>
#include <boost/thread/thread.hpp>

#define CONCURRENCY_COUNT 4

class JobSystem
{
public:
    JobSystem(int concurrency_count) 
    {
        std::cout << "Job system is starting..." << std::endl;

        this->concurrency_count = concurrency_count;
        this->workers.reserve(this->concurrency_count);

        for (int i = 0; i < this->concurrency_count; ++i)
        {
            this->workers[i] = boost::thread([this, i] {
                boost::fibers::use_scheduling_algorithm<boost::fibers::algo::work_stealing>(this->concurrency_count);
                std::cout << "thread " << i << " is on duty." << std::endl;

            });
        }
    }
    ~JobSystem()
    {
        for (auto &worker : this->workers)
        {
            worker.join();
        }
        std::cout << "Job system is shutting down..." << std::endl;
    }
private:

    int concurrency_count;

    std::vector<boost::thread> workers;
};

int main()
{

    JobSystem *job_system = new JobSystem(CONCURRENCY_COUNT);

    bool is_engine_running = true;
    auto stop_time = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (is_engine_running)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        if (std::chrono::steady_clock::now() > stop_time)
        {
            is_engine_running = false;
        }
    }

    delete job_system;

    std::cout << "system is shutting down... bye!" << std::endl;

    return 0;
}