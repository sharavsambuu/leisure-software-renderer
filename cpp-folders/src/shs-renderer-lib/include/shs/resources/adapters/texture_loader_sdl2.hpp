#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: texture_loader_sdl2.hpp
    МОДУЛЬ: resources (SDL2_image texture adapter)
    ЗОРИЛГО: SDL2_image twin of texture_loader_sdl.hpp (SDL3). SDL2 and SDL3
             headers must never share a translation unit.
*/

#include <string>

#if __has_include(<SDL2/SDL.h>)
#include <SDL2/SDL.h>
#else
#include <SDL.h>
#endif
#if __has_include(<SDL2_image/SDL_image.h>)
#include <SDL2_image/SDL_image.h>
#elif __has_include(<SDL2/SDL_image.h>)
#include <SDL2/SDL_image.h> // system (Debian-style) dev layout: /usr/include/SDL2
#else
#include <SDL_image.h>
#endif

#include "shs/platform/sdl/sdl2_runtime.hpp" // dlopen-dispatched SDL2/SDL2_image API (dual-link hijack shield)
#include "shs/resources/texture.hpp"
#include "shs/resources/storage/resource_registry.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace resources
    {
    inline Texture2DData load_texture2d_sdl2_image(const std::string& path, bool flip_y = true)
    {
        // All SDL2/SDL2_image calls go through the dlopen-dispatched API table
        // so demo binaries dual-linked with static SDL3 cannot hijack them.
        const shs::platform::Sdl2Api* s2 = shs::platform::shs_sdl2_api();
        if (!s2 || !s2->ImgLoad) return Texture2DData{};

        // SDL2_image keeps explicit loader init (SDL3_image removed it).
        s2->ImgInit(IMG_INIT_PNG | IMG_INIT_JPG);

        SDL_Surface* loaded = s2->ImgLoad(path.c_str());
        if (!loaded)
        {
            s2->ImgQuit();
            return Texture2DData{};
        }

        SDL_Surface* rgba = s2->ConvertSurfaceFormat(loaded, SDL_PIXELFORMAT_RGBA32, 0);
        s2->FreeSurface(loaded);
        if (!rgba)
        {
            s2->ImgQuit();
            return Texture2DData{};
        }

        Texture2DData out{rgba->w, rgba->h, Color{0, 0, 0, 0}};
        out.source_path = path;

        auto* pixels = static_cast<uint8_t*>(rgba->pixels);
        const int pitch = rgba->pitch;
        const SDL_PixelFormat* fmt = rgba->format;
        for (int y = 0; y < out.h; ++y)
        {
            const int dst_y = flip_y ? (out.h - 1 - y) : y;
            const uint32_t* row = reinterpret_cast<const uint32_t*>(pixels + y * pitch);
            for (int x = 0; x < out.w; ++x)
            {
                uint8_t r = 0, g = 0, b = 0, a = 0;
                s2->GetRGBA(row[x], fmt, &r, &g, &b, &a);
                out.at(x, dst_y) = Color{r, g, b, a};
            }
        }

        s2->FreeSurface(rgba);
        s2->ImgQuit();
        return out;
    }

    inline TextureAssetHandle import_texture_sdl2(
        ResourceRegistry& reg,
        const std::string& path,
        const std::string& key = {},
        bool flip_y = true
    )
    {
        Texture2DData tex = load_texture2d_sdl2_image(path, flip_y);
        if (!tex.valid()) return 0;
        return reg.add_texture(std::move(tex), key.empty() ? path : key);
    }

    } // inline namespace resources
}
