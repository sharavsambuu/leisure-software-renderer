#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: time.hpp
    МОДУЛЬ: core
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн core модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <cstdint>

namespace shs
{
    struct FrameClock
    {
        uint64_t ticks_prev = 0;
        double tick_hz = 1.0;

        float begin_frame(uint64_t ticks_now)
        {
            if (ticks_prev == 0)
            {
                ticks_prev = ticks_now;
                return 0.0f;
            }
            const float dt = (float)((double)(ticks_now - ticks_prev) / tick_hz);
            ticks_prev = ticks_now;
            return dt;
        }
    };
}

