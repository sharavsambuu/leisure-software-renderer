/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: platform_sdl2_anchor.cpp
    МОДУЛЬ: platform (edge anchor TU)
    ЗОРИЛГО: SDL2 windowing backend factory + SDL2_image texture dispatch
             target. Compiled unconditionally; the SDL SDK headers are
             included only when SHS_HAS_SDL2=1, so the TU stays SDK-free when
             SDL2 was not discovered. SDL2 and SDL3 headers must never share
             a translation unit — each backend owns its own anchor.
*/

#include "shs/platform/window_backend.hpp"
#include "shs/resources/adapters/texture_loader.hpp"
#include "shs/resources/texture.hpp"

#ifdef SHS_HAS_SDL2
#include "shs/platform/sdl/sdl2_runtime.hpp"
#include "shs/resources/adapters/texture_loader_sdl2.hpp"

namespace shs::platform
{
    IPlatformRuntime* create_sdl2_runtime(const WindowDesc& win, const SurfaceDesc& surface)
    {
        auto* runtime = new Sdl2Runtime(win, surface);
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
    Texture2DData load_texture2d_sdl2_backend(const std::string& path, bool flip_y)
    {
        return load_texture2d_sdl2_image(path, flip_y);
    }
}
#else
namespace shs::platform
{
    IPlatformRuntime* create_sdl2_runtime(const WindowDesc&, const SurfaceDesc&)
    {
        return nullptr; // SDL2 backend not compiled in
    }
}

namespace shs::resources
{
    Texture2DData load_texture2d_sdl2_backend(const std::string&, bool)
    {
        return {}; // SDL2_image backend not compiled in
    }
}
#endif
