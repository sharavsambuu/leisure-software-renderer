#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: texture.hpp
    МОДУЛЬ: resources
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн resources модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <cstdint>
#include <string>
#include <vector>

#include "shs/gfx/rt_types.hpp"

namespace shs
{
    using TextureAssetHandle = uint32_t;

    struct Texture2DData
    {
        std::string source_path{};
        int w = 0;
        int h = 0;
        std::vector<Color> texels{};

        Texture2DData() = default;
        Texture2DData(int W, int H, Color clear = {0, 0, 0, 255})
            : w(W), h(H), texels((size_t)W * (size_t)H, clear)
        {}

        bool valid() const
        {
            return w > 0 && h > 0 && texels.size() == (size_t)w * (size_t)h;
        }

        Color& at(int x, int y)
        {
            return texels[(size_t)y * (size_t)w + (size_t)x];
        }

        const Color& at(int x, int y) const
        {
            return texels[(size_t)y * (size_t)w + (size_t)x];
        }
    };
}

