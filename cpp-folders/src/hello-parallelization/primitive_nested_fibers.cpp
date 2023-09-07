#include <iostream>
#include <boost/fiber/all.hpp>
#include <boost/fiber/fiber.hpp>
#include <boost/fiber/future.hpp>
#include <boost/thread/thread.hpp>


void parent_task(int pidx)
{
    long result = 0;
    for (long j = 0; j < 50; ++j)
    {
        boost::fibers::promise<long> p;
        boost::fibers::fiber([&p, j] {
            long sum = 0;
            for (long i=0; i<j; ++i) {
                sum += i;
                boost::this_fiber::yield();
            }
            p.set_value(sum); 
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
    boost::thread worker_threads[4];
    for (int i = 0; i < 4; ++i)
    {
        worker_threads[i] = boost::thread([i] {
            boost::fibers::use_scheduling_algorithm<boost::fibers::algo::work_stealing>(4);

            for (int j = i * 100; j < (i + 1) * 100; ++j) {
                boost::fibers::fiber(parent_task, j).detach();
            }
            for (int j = i * 100; j < (i + 1) * 100; ++j)
            {
                boost::fibers::fiber(parent_task, j).join();
            }
            std::cout << "thread"<< i <<" is done" << std::endl << std::flush; 

        });
    }

    for (int i = 0; i < 4; ++i)
    {
        worker_threads[i].join();
    }

    return 0;
}