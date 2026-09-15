#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: light_culling_mode.hpp
    МОДУЛЬ: lighting
    ЗОРИЛГО: Гэрлийн culling-ийн горимуудыг backend/technique хооронд
            нийтлэг, өргөтгөхөд хялбар семантикаар тодорхойлно.
*/

#include <cstdint>

namespace shs
{
    enum class LightCullingMode : uint32_t
    {
        None = 0,
        Tiled = 1,
        TiledDepthRange = 2,
        Clustered = 3
    };

    inline const char* light_culling_mode_name(LightCullingMode mode)
    {
        switch (mode)
        {
            case LightCullingMode::None: return "none";
            case LightCullingMode::Tiled: return "tiled";
            case LightCullingMode::TiledDepthRange: return "tiled-depth";
            case LightCullingMode::Clustered: return "clustered";
        }
        return "unknown";
    }

    inline LightCullingMode next_light_culling_mode(LightCullingMode mode)
    {
        switch (mode)
        {
            case LightCullingMode::None: return LightCullingMode::Tiled;
            case LightCullingMode::Tiled: return LightCullingMode::TiledDepthRange;
            case LightCullingMode::TiledDepthRange: return LightCullingMode::Clustered;
            case LightCullingMode::Clustered: return LightCullingMode::None;
        }
        return LightCullingMode::None;
    }
}
