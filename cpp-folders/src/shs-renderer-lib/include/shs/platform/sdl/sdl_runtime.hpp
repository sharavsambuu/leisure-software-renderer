#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: sdl_runtime.hpp
    МОДУЛЬ: platform (SDL3 windowing backend adapter)
    ЗОРИЛГО: Compatibility include (retired SDL2-era name). The canonical
             SDL3 adapter is sdl3_runtime.hpp; the SDL2 twin is sdl2_runtime.hpp.
*/

// Compatibility include: definitions live in the platform-owned canonical header.
#include "shs/platform/sdl/sdl3_runtime.hpp"

namespace shs
{
    inline namespace platform
    {
    // Retired SDL2-era class name; the SDL3 backend is canonical.
    using SdlRuntime = Sdl3Runtime;
    }
}
