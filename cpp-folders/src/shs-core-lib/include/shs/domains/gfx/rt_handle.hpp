#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: rt_handle.hpp
    МОДУЛЬ: gfx
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн gfx модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <cstdint>

namespace shs
{
    struct RTHandle
    {
        uint32_t id = 0; // 0 = invalid
        constexpr bool valid() const { return id != 0; }
    };

    // Small typed wrappers (compile-time type separation only)
    struct RT_Color : RTHandle {};
    struct RT_Depth : RTHandle {};
    struct RT_Motion : RTHandle {};
    struct RT_Shadow : RTHandle {};
}

