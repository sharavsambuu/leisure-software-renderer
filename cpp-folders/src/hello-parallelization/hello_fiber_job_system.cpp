#include <iostream>
#include <vector>
#include <queue>
#include <functional>
#include <boost/fiber/all.hpp>
#include <boost/fiber/fiber.hpp>
#include <boost/fiber/future.hpp>
#include <boost/chrono.hpp>
#include <boost/thread/thread.hpp>
#include <boost/lockfree/queue.hpp>

#define CONCURRENCY_COUNT 4


class AbstractJobSystem
{
public:
    virtual ~AbstractJobSystem() {};
    virtual void submit(std::function<void()> job) = 0;
    bool is_running = true;
};


class JobSystem : public AbstractJobSystem
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

                while(this->is_running)
                {
                    std::function<void()> job;

                    {
                        std::unique_lock<std::mutex> lock(this->mutex);
                        if (!this->job_queue.empty())
                        {
                            job = std::move(this->job_queue.front());
                            this->job_queue.pop();
                        }
                    }

                    if (job)
                    {
                        //boost::fibers::fiber(job).detach();
                        boost::fibers::fiber(job).join();
                    }
                }
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
    void submit(std::function<void()> job) override
    {
        {
            std::unique_lock<std::mutex> lock(this->mutex);
            job_queue.push(std::move(job));
        }
    }
private:
    int concurrency_count;
    std::vector<boost::thread>        workers;
    std::queue<std::function<void()>> job_queue;
    std::mutex mutex;
};


class LocklessJobSystem : public AbstractJobSystem
{
public:
    LocklessJobSystem(int concurrency_count) 
    {
        std::cout << "Job system is starting..." << std::endl;

        this->concurrency_count = concurrency_count;
        this->workers.reserve(this->concurrency_count);

        for (int i = 0; i < this->concurrency_count; ++i)
        {
            this->workers[i] = boost::thread([this, i] {
                boost::fibers::use_scheduling_algorithm<boost::fibers::algo::work_stealing>(this->concurrency_count);

                while(this->is_running)
                {
                    std::function<void(void)>* job;
                    if (this->job_queue.pop(job))
                    {
                        boost::fibers::fiber([job]() {
                            (*job)();
                        }).join();
                    }
                }
            });
        }
    }
    ~LocklessJobSystem()
    {
        for (auto &worker : this->workers)
        {
            worker.join();
        }
        std::cout << "Job system is shutting down..." << std::endl;
    }
    void submit(std::function<void()> job) override
    {
        std::function<void(void)>* task = new std::function<void(void)>(job);
        this->job_queue.push(task);
    }
private:
    int concurrency_count;
    std::vector<boost::thread> workers;
    boost::lockfree::queue<std::function<void(void)>*, boost::lockfree::capacity<1024>> job_queue;
};


void send_batch_jobs(AbstractJobSystem &job_system)
{
    for (int i = 0; i < 2000; ++i)
    {
        job_system.submit([i] {
            std::cout << "Job " << i << " started" << std::endl;
            for (int j=0; j<200; ++j)
            {
                std::cout << "Job " << i << " is working..." << std::endl;
                boost::this_fiber::yield(); // let's be nice with each other
            }
            boost::this_fiber::yield();
            std::cout << "Job " << i << " finished" << std::endl;
        });
    }
}


int main()
{

    AbstractJobSystem *job_system          = new JobSystem        (CONCURRENCY_COUNT);
    //AbstractJobSystem *lockless_job_system = new LocklessJobSystem(CONCURRENCY_COUNT);

    bool is_engine_running = true;


    auto first_stop_time  = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    bool is_sent_second_batch = false;
    auto second_stop_time = std::chrono::steady_clock::now() + std::chrono::seconds(30);


    std::cout << ">>>>> sending first batch jobs" << std::endl;
    send_batch_jobs(*job_system);


    while (is_engine_running)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));


        if (std::chrono::steady_clock::now() > first_stop_time && !is_sent_second_batch)
        {

            std::cout << ">>>>> sending second batch jobs to the lockless workers" << std::endl;
            //send_batch_jobs(*lockless_job_system);
            send_batch_jobs(*job_system);
            is_sent_second_batch = true;
        }

        if (std::chrono::steady_clock::now() > second_stop_time)
        {
            is_engine_running               = false;
            job_system->is_running          = false;
            //lockless_job_system->is_running = false;
        }
    }

    delete job_system;
    //delete lockless_job_system;

    std::cout << "system is shutting down... bye!" << std::endl;

    return 0;
}