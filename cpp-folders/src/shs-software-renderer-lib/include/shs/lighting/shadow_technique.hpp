#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: shadow_technique.hpp
    МОДУЛЬ: lighting
    ЗОРИЛГО: Shadow system-ийг light type болон rendering composition-оор нь
            backend-д нийтлэг, өргөтгөхөд хялбар семантикаар тодорхойлно.
*/

#include <cstdint>

#include "shs/lighting/light_types.hpp"

namespace shs
{
    enum class ShadowTechnique : uint32_t
    {
        None = 0,
        DirectionalMap2D = 1,
        SpotMap2D = 2,
        PointCube = 3,
        AreaProxySpotMap2D = 4
    };

    enum class ShadowFilter : uint32_t
    {
        Hard = 0,
        PCF3x3 = 1,
        PCF5x5 = 2
    };

    struct ShadowQualityParams
    {
        uint32_t directional_resolution = 2048;
        uint32_t local_resolution = 1024;
        uint32_t point_resolution = 512;
        ShadowFilter filter = ShadowFilter::PCF3x3;
        float pcf_step = 1.0f;
    };

    struct ShadowCasterBudget
    {
        uint32_t max_directional = 1;
        uint32_t max_spot = 4;
        uint32_t max_point = 2;
        uint32_t max_rect_area = 2;
        uint32_t max_tube_area = 2;
    };

    struct ShadowCompositionSettings
    {
        bool enable = true;
        bool directional = true;
        bool point = true;
        bool spot = true;
        bool rect_area_proxy = true;
        bool tube_area_proxy = true;
        ShadowQualityParams quality{};
        ShadowCasterBudget budget{};
    };

    inline const char* shadow_technique_name(ShadowTechnique t)
    {
        switch (t)
        {
            case ShadowTechnique::None: return "none";
            case ShadowTechnique::DirectionalMap2D: return "directional_map_2d";
            case ShadowTechnique::SpotMap2D: return "spot_map_2d";
            case ShadowTechnique::PointCube: return "point_cube";
            case ShadowTechnique::AreaProxySpotMap2D: return "area_proxy_spot_map_2d";
        }
        return "unknown";
    }

    inline const char* shadow_filter_name(ShadowFilter f)
    {
        switch (f)
        {
            case ShadowFilter::Hard: return "hard";
            case ShadowFilter::PCF3x3: return "pcf3x3";
            case ShadowFilter::PCF5x5: return "pcf5x5";
        }
        return "unknown";
    }

    inline bool shadow_technique_uses_cube_map(ShadowTechnique t)
    {
        return t == ShadowTechnique::PointCube;
    }

    inline bool shadow_technique_uses_2d_map(ShadowTechnique t)
    {
        return t == ShadowTechnique::DirectionalMap2D ||
               t == ShadowTechnique::SpotMap2D ||
               t == ShadowTechnique::AreaProxySpotMap2D;
    }

    inline ShadowTechnique default_shadow_technique_for_light(LightType type)
    {
        switch (type)
        {
            case LightType::Directional:
                return ShadowTechnique::DirectionalMap2D;
            case LightType::Point:
                return ShadowTechnique::PointCube;
            case LightType::Spot:
                return ShadowTechnique::SpotMap2D;
            case LightType::RectArea:
            case LightType::TubeArea:
                return ShadowTechnique::AreaProxySpotMap2D;
            case LightType::EnvironmentProbe:
            default:
                return ShadowTechnique::None;
        }
    }

    inline ShadowCompositionSettings make_default_shadow_composition_settings()
    {
        ShadowCompositionSettings s{};
        return s;
    }
}

