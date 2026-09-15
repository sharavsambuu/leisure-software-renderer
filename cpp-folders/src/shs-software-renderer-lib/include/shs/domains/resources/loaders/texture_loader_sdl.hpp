#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: texture_loader_sdl.hpp
    МОДУЛЬ: resources
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн resources модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <string>

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>

#include "shs/resources/texture.hpp"

namespace shs
{
    inline Texture2DData load_texture2d_sdl_image(const std::string& path, bool flip_y = true)
    {
        SDL_Surface* loaded = IMG_Load(path.c_str());
        if (!loaded) return Texture2DData{};

        SDL_Surface* rgba = SDL_ConvertSurfaceFormat(loaded, SDL_PIXELFORMAT_RGBA32, 0);
        SDL_FreeSurface(loaded);
        if (!rgba) return Texture2DData{};

        Texture2DData out{rgba->w, rgba->h, Color{0, 0, 0, 0}};
        out.source_path = path;

        auto* pixels = static_cast<uint8_t*>(rgba->pixels);
        const int pitch = rgba->pitch;
        for (int y = 0; y < out.h; ++y)
        {
            const int dst_y = flip_y ? (out.h - 1 - y) : y;
            auto* row = reinterpret_cast<uint32_t*>(pixels + y * pitch);
            for (int x = 0; x < out.w; ++x)
            {
                uint8_t r = 0, g = 0, b = 0, a = 0;
                SDL_GetRGBA(row[x], rgba->format, &r, &g, &b, &a);
                out.at(x, dst_y) = Color{r, g, b, a};
            }
        }

        SDL_FreeSurface(rgba);
        return out;
    }
}
