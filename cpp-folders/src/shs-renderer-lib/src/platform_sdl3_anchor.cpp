/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: platform_sdl3_anchor.cpp
    МОДУЛЬ: platform (edge anchor TU)
    ЗОРИЛГО: SDL3 windowing backend factory + SDL3_image texture dispatch
             target. Compiled unconditionally; the SDL SDK headers are
             included only when SHS_HAS_SDL3=1, so the TU stays SDK-free when
             SDL3 was not discovered. SDL2 and SDL3 headers must never share
             a translation unit — each backend owns its own anchor.
*/

#include "shs/platform/window_backend.hpp"
#include "shs/resources/adapters/texture_loader.hpp"
#include "shs/resources/texture.hpp"

#ifdef SHS_HAS_SDL3
#include "shs/platform/sdl/sdl3_runtime.hpp"
#include "shs/resources/adapters/texture_loader_sdl.hpp"

namespace shs::platform
{
    IPlatformRuntime* create_sdl3_runtime(const WindowDesc& win, const SurfaceDesc& surface)
    {
        auto* runtime = new Sdl3Runtime(win, surface);
        if (!runtime->valid())
        {
            delete runtime;
            return nullptr;
        }
        return runtime;
    }
}

namespace shs::resources
{
    Texture2DData load_texture2d_sdl3_backend(const std::string& path, bool flip_y)
    {
        return load_texture2d_sdl_image(path, flip_y);
    }
}
#else
namespace shs::platform
{
    IPlatformRuntime* create_sdl3_runtime(const WindowDesc&, const SurfaceDesc&)
    {
        return nullptr; // SDL3 backend not compiled in
    }
}

namespace shs::resources
{
    Texture2DData load_texture2d_sdl3_backend(const std::string&, bool)
    {
        return {}; // SDL3_image backend not compiled in
    }
}
#endif
