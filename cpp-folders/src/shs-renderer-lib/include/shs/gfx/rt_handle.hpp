// File: src/shs-renderer-lib/include/shs/gfx/rt_handle.hpp
#pragma once
/*
    SHS RENDERER LIB - RT HANDLE

    ЗОРИЛГО:
    - Opaque void* оронд type-safe handle ашиглах (demo бүрийн RT struct өөр байж болно)
    - Дараа нь static/dynamic lib болгоход ABI-г хялбар байлгана
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

