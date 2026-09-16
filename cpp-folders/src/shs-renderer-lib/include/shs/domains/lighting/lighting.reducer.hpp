#pragma once

/*
    SHS RENDERER SAN

    FILE: lighting.reducer.hpp
    MODULE: domains/lighting
    PURPOSE: CORE 4. REDUCER — the identity transition (R4, P4.1 house shape).
             The pod owns no mutable state yet (light vocabulary + shading
             terms are values; culling runtime migrates in R5).
*/

#include <memory_resource>
#include <span>

#include "shs/domains/lighting/lighting.action.hpp"
#include "shs/domains/lighting/lighting.contract.hpp"
#include "shs/domains/lighting/lighting.event.hpp"

namespace shs::lighting
{
    struct LightingState
    {
        bool operator==(const LightingState&) const = default;
    };

    struct LightingReduceInputs
    {
    };

    inline void reduce_lighting(
        LightingState&                   state,
        std::span<const LightingAction>  actions,
        const LightingReduceInputs&      inputs,
        std::pmr::vector<LightingEvent>& events)
    {
        (void)state;
        (void)actions;
        (void)inputs;
        (void)events;
    }
} // namespace shs::lighting
