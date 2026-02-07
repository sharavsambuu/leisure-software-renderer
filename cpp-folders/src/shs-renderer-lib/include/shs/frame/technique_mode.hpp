#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: technique_mode.hpp
    МОДУЛЬ: frame
    ЗОРИЛГО: Rendering technique mode-ийн суурь enum/mask helper-үүд.
*/


#include <cstdint>

namespace shs
{
    enum class TechniqueMode : uint8_t
    {
        Forward = 0,
        ForwardPlus = 1,
        Deferred = 2,
        TiledDeferred = 3,
        ClusteredForward = 4
    };

    inline const char* technique_mode_name(TechniqueMode m)
    {
        switch (m)
        {
            case TechniqueMode::Forward: return "forward";
            case TechniqueMode::ForwardPlus: return "forward_plus";
            case TechniqueMode::Deferred: return "deferred";
            case TechniqueMode::TiledDeferred: return "tiled_deferred";
            case TechniqueMode::ClusteredForward: return "clustered_forward";
        }
        return "unknown";
    }

    constexpr uint32_t technique_mode_bit(TechniqueMode m)
    {
        return (1u << static_cast<uint32_t>(m));
    }

    constexpr uint32_t technique_mode_mask_all()
    {
        return technique_mode_bit(TechniqueMode::Forward) |
               technique_mode_bit(TechniqueMode::ForwardPlus) |
               technique_mode_bit(TechniqueMode::Deferred) |
               technique_mode_bit(TechniqueMode::TiledDeferred) |
               technique_mode_bit(TechniqueMode::ClusteredForward);
    }

    inline bool technique_mode_in_mask(uint32_t mask, TechniqueMode m)
    {
        return (mask & technique_mode_bit(m)) != 0u;
    }
}
