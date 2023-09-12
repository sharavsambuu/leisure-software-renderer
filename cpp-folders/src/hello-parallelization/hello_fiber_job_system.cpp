#include <iostream>
#include <vector>
#include <boost/fiber/all.hpp>
#include <boost/fiber/fiber.hpp>
#include <boost/fiber/future.hpp>
#include <boost/thread/thread.hpp>

#define CONCURRENCY_COUNT 4


int main()
{

    bool is_engine_running = true;

    auto stop_time = std::chrono::steady_clock::now() + std::chrono::seconds(10);

    while (is_engine_running)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        std::cout << "engine is running" << std::endl;

        if (std::chrono::steady_clock::now() > stop_time)
        {
            is_engine_running = false;
        }
    }

    std::cout << "system is shutting down... bye!" << std::endl;

    return 0;
}