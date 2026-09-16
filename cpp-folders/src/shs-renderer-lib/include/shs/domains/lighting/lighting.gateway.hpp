#pragma once

/*
    SHS RENDERER SAN

    FILE: lighting.gateway.hpp
    MODULE: domains/lighting
    PURPOSE: CORE 4. GATEWAY — the identity transition (R4, P4.1 house shape).
             The pod owns no mutable state yet (light vocabulary + shading
             terms are values; culling runtime migrates in R5).
*/

#include <memory_resource>
#include <span>

#include "shs/domains/lighting/lighting.command.hpp"
#include "shs/domains/lighting/lighting.contract.hpp"
#include "shs/domains/lighting/lighting.event.hpp"

namespace shs::lighting
{
    struct LightingState
    {
        bool operator==(const LightingState&) const = default;
    };

    struct LightingContext
    {
    };

    inline void lighting_gateway(
        LightingState&                   state,
        std::span<const LightingCommand>  commands,
        const LightingContext&      context,
        std::pmr::vector<LightingEvent>& events)
    {
        (void)state;
        (void)commands;
        (void)context;
        (void)events;
    }
} // namespace shs::lighting
