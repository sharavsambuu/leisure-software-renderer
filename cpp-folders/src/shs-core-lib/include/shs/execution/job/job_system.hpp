#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: job_system.hpp
    МОДУЛЬ: job
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн job модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <cstddef>
#include <functional>

namespace shs
{
    class IJobSystem
    {
    public:
        virtual ~IJobSystem() = default;
        virtual void enqueue(std::function<void()> job) = 0;
        virtual void wait_idle() = 0;
        virtual size_t worker_count() const = 0;
    };
}

