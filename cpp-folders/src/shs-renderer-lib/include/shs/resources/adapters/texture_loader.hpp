#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: texture_loader.hpp
    МОДУЛЬ: resources (platform-agnostic image-loading facade)
    ЗОРИЛГО: Backend-agnostic entry over the SDL3_image / SDL2_image adapter
             headers. Contains zero SDK includes; dispatch goes through the
             compiled anchor functions (src/platform_sdl3_anchor.cpp /
             src/platform_sdl2_anchor.cpp) because SDL2 and SDL3 headers must
             never share a translation unit.
*/

#include <string>

#include "shs/resources/texture.hpp"
#include "shs/resources/storage/resource_registry.hpp"

namespace shs
{
    // Compiled backend dispatch targets. Always-present symbols; return an
    // empty Texture2DData when the backend is not compiled in or the load
    // failed (honest result, no exception).
    namespace resources
    {
        Texture2DData load_texture2d_sdl3_backend(const std::string& path, bool flip_y);
        Texture2DData load_texture2d_sdl2_backend(const std::string& path, bool flip_y);
    }

// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace resources
    {
    enum class ImageSource
    {
        Auto, // prefer SDL3_image, fall back to SDL2_image
        Sdl3,
        Sdl2,
    };

    inline Texture2DData load_texture2d_image(
        const std::string& path,
        bool flip_y = true,
        ImageSource source = ImageSource::Auto)
    {
        if (source == ImageSource::Sdl3 || source == ImageSource::Auto)
        {
            Texture2DData tex = load_texture2d_sdl3_backend(path, flip_y);
            if (tex.valid() || source == ImageSource::Sdl3) return tex;
        }
        if (source == ImageSource::Sdl2 || source == ImageSource::Auto)
        {
            Texture2DData tex = load_texture2d_sdl2_backend(path, flip_y);
            if (tex.valid() || source == ImageSource::Sdl2) return tex;
        }
        return Texture2DData{};
    }

    inline TextureAssetHandle import_texture_image(
        ResourceRegistry& reg,
        const std::string& path,
        const std::string& key = {},
        bool flip_y = true,
        ImageSource source = ImageSource::Auto
    )
    {
        Texture2DData tex = load_texture2d_image(path, flip_y, source);
        if (!tex.valid()) return 0;
        return reg.add_texture(std::move(tex), key.empty() ? path : key);
    }

    } // inline namespace resources
}
