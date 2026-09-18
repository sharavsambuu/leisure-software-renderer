#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: texture_loader_sdl.hpp
    МОДУЛЬ: execution/platform (edge: SDL image IO; moved from domains/resources P0.2)
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн resources модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <string>

#include <SDL3/SDL.h>
#include <SDL3_image/SDL_image.h>

#include "shs/resources/texture.hpp"
#include "shs/resources/storage/resource_registry.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace resources
    {
    inline Texture2DData load_texture2d_sdl_image(const std::string& path, bool flip_y = true)
    {
        SDL_Surface* loaded = IMG_Load(path.c_str());
        if (!loaded) return Texture2DData{};

        SDL_Surface* rgba = SDL_ConvertSurface(loaded, SDL_PIXELFORMAT_RGBA32);
        SDL_DestroySurface(loaded);
        if (!rgba) return Texture2DData{};

        Texture2DData out{rgba->w, rgba->h, Color{0, 0, 0, 0}};
        out.source_path = path;

        auto* pixels = static_cast<uint8_t*>(rgba->pixels);
        const int pitch = rgba->pitch;
        const SDL_PixelFormatDetails* fmt = SDL_GetPixelFormatDetails(rgba->format);
        if (!fmt)
        {
            SDL_DestroySurface(rgba);
            return Texture2DData{};
        }
        for (int y = 0; y < out.h; ++y)
        {
            const int dst_y = flip_y ? (out.h - 1 - y) : y;
            auto* row = reinterpret_cast<uint32_t*>(pixels + y * pitch);
            for (int x = 0; x < out.w; ++x)
            {
                uint8_t r = 0, g = 0, b = 0, a = 0;
                SDL_GetRGBA(row[x], fmt, nullptr, &r, &g, &b, &a);
                out.at(x, dst_y) = Color{r, g, b, a};
            }
        }

        SDL_DestroySurface(rgba);
        return out;
    }

    inline TextureAssetHandle import_texture_sdl(
        ResourceRegistry& reg,
        const std::string& path,
        const std::string& key = {},
        bool flip_y = true
    )
    {
        Texture2DData tex = load_texture2d_sdl_image(path, flip_y);
        if (!tex.valid()) return 0;
        return reg.add_texture(std::move(tex), key.empty() ? path : key);
    }

    } // inline namespace resources
}
