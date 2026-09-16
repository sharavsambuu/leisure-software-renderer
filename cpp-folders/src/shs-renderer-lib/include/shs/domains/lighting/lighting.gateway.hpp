#pragma once

/*
    SHS RENDERER SAN

    FILE: lighting.gateway.hpp
    MODULE: domains/lighting
    PURPOSE: CORE 4. GATEWAY — the pinned identity transition, Kleisli house
             shape (Run C, K1.5): (State, span<Commands>, Context, arena)
             -> LightingStep. The pod owns no mutable state yet (light
             vocabulary + shading terms are values); the empty-vocabulary
             static_assert makes the silence provable (§6.1).
             Pinned by lighting_tests.
*/

#include <cstdint>
#include <memory_resource>
#include <span>
#include <type_traits>
#include <variant>

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

    // Batch outcome summary (house shape per kdba_kleisli_migration_plan.md).
    // An identity pod applies nothing — it counts what passed the seam.
    struct LightingStep
    {
        uint32_t commands_observed = 0;

        bool operator==(const LightingStep&) const = default;
    };

    inline LightingStep lighting_gateway(
        LightingState&                   state,
        std::span<const LightingCommand>  commands,
        const LightingContext&      context,
        std::pmr::vector<LightingEvent>& events)
    {
        (void)state;
        (void)context;
        (void)events;
        LightingStep step{};
        for (const LightingCommand& command : commands)
        {
            std::visit([&](const auto& cmd) {
                using T = std::decay_t<decltype(cmd)>;
                static_assert(std::is_same_v<T, std::monostate>,
                    "lighting command vocabulary is empty by law (§6.1): "
                    "land a new intent as a named apply_* arrow first");
                (void)cmd;
            }, command);
            step.commands_observed += 1;
        }
        return step;
    }
} // namespace shs::lighting
