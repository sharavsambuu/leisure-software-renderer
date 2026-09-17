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
    // Job-system contract (step 4.4, engine_domain_separation_migration.md):
    //
    // THREAD ACCESS
    //   - enqueue() and wait_idle() are safe to call from any thread
    //   concurrently, including while workers are running other jobs.
    //   - Jobs execute on ARBITRARY worker threads (never the enqueuing
    //   thread's); a job must not assume thread affinity or TLS set up by
    //   the submitter.
    //   - A job may enqueue further jobs (including from a worker thread);
    //   wait_idle() observes the whole transitively drained queue.
    //
    // SHUTDOWN ORDERING
    //   - wait_idle() is the only completion guarantee; call it before
    //   signaling teardown that releases memory jobs still reference.
    //   - Destruction drains: every job accepted by enqueue() before the
    //   destructor starts has run before the destructor returns.
    //   - enqueue() after destruction has begun is a caller error (use
    //   wait_idle() + teardown barrier before releasing producers).
    class IJobSystem
    {
    public:
        virtual ~IJobSystem() = default;
        virtual void enqueue(std::function<void()> job) = 0;
        virtual void wait_idle() = 0;
        virtual size_t worker_count() const = 0;
    };
}

