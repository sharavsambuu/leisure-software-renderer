#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: parallel_for.hpp
    МОДУЛЬ: job
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн job модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <algorithm>
#include <cstddef>
#include <functional>

#include "shs/job/job_system.hpp"
#include "shs/job/wait_group.hpp"

namespace shs
{
    template<typename Fn>
    inline void parallel_for_1d(
        IJobSystem* js,
        int begin,
        int end,
        int min_grain,
        Fn&& fn
    )
    {
        if (end <= begin) return;
        const int count = end - begin;
        // Ажил бага эсвэл job system байхгүй үед sync замаар ажиллуулна.
        if (!js || count <= std::max(1, min_grain))
        {
            fn(begin, end);
            return;
        }

        const int workers = (int)std::max<size_t>(1, js->worker_count());
        // Chunk-ийг worker тоонд дөхүүлж, хэт жижиг тасархай үүсгэхээс сэргийлнэ.
        const int chunks = std::max(1, std::min(workers * 2, (count + min_grain - 1) / std::max(1, min_grain)));
        const int chunk_size = (count + chunks - 1) / chunks;

        WaitGroup wg{};
        for (int i = 0; i < chunks; ++i)
        {
            const int b = begin + i * chunk_size;
            const int e = std::min(end, b + chunk_size);
            if (b >= e) continue;

            wg.add(1);
            js->enqueue([b, e, &fn, &wg]() {
                fn(b, e);
                wg.done();
            });
        }
        wg.wait();
    }
}
