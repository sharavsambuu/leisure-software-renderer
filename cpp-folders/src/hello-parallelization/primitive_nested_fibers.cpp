#include <iostream>
#include <boost/fiber/all.hpp>
#include <boost/fiber/fiber.hpp>
#include <boost/fiber/future.hpp>
#include <boost/thread/thread.hpp>

#define CONCURRENCY_COUNT 4
#define WORKER_COUNT      5000


void sub_task(boost::fibers::promise<long> &p, long j)
{
    long sum = 0;
    for (long i = 0; i < j; ++i)
    {
        sum += i;
        boost::this_fiber::yield();
    }
    p.set_value(sum);
}

void parent_task(int pidx)
{
    long result = 0;
    for (long j = 0; j < 50; ++j)
    {
        boost::fibers::promise<long> p;
        boost::fibers::fiber([&p, j]() {
            sub_task(p, j);
        }).detach();

        boost::this_fiber::yield();

        long sub_result = p.get_future().get();
        result += sub_result;

        boost::this_fiber::yield();
    }
    std::cout << "parent task" << pidx << " " << result << std::endl << std::flush;

}

int main()
{
    boost::thread worker_threads[CONCURRENCY_COUNT];

    for (int i = 0; i < CONCURRENCY_COUNT; ++i)
    {
        worker_threads[i] = boost::thread([i] {
            boost::fibers::use_scheduling_algorithm<boost::fibers::algo::work_stealing>(CONCURRENCY_COUNT);

            for (int j = i * WORKER_COUNT; j < (i + 1) * WORKER_COUNT; ++j) {
                boost::fibers::fiber(parent_task, j).detach();
            }
            for (int j = i * WORKER_COUNT; j < (i + 1) * WORKER_COUNT; ++j)
            {
                boost::fibers::fiber(parent_task, j).join();
            }
            std::cout << "thread"<< i <<" is done" << std::endl << std::flush; 
        });
    }

    for (int i = 0; i < CONCURRENCY_COUNT; ++i)
    {
        worker_threads[i].join();
    }

    return 0;
}