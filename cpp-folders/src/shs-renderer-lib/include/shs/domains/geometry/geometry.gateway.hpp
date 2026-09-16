#pragma once

/*
    SHS RENDERER SAN

    FILE: geometry.gateway.hpp
    MODULE: domains/geometry
    PURPOSE: CORE 4. GATEWAY — the pinned identity transition, Kleisli house
             shape (Run C, K1.5): (State, span<Commands>, Context, arena)
             -> GeometryStep. The pod owns no mutable state yet (shapes/
             adapters are values), so reduction is stability by construction;
             the empty-vocabulary static_assert makes it provable (§6.1).
             Pinned by geometry_tests.
*/

#include <cstdint>
#include <memory_resource>
#include <span>
#include <type_traits>
#include <variant>

#include "shs/domains/geometry/geometry.command.hpp"
#include "shs/domains/geometry/geometry.contract.hpp"
#include "shs/domains/geometry/geometry.event.hpp"

namespace shs::geometry
{
    struct GeometryState
    {
        bool operator==(const GeometryState&) const = default;
    };

    struct GeometryContext
    {
    };

    // Batch outcome summary (house shape per kdba_kleisli_migration_plan.md).
    // An identity pod applies nothing — it counts what passed the seam.
    struct GeometryStep
    {
        uint32_t commands_observed = 0;

        bool operator==(const GeometryStep&) const = default;
    };

    inline GeometryStep geometry_gateway(
        GeometryState&                   state,
        std::span<const GeometryCommand>  commands,
        const GeometryContext&      context,
        std::pmr::vector<GeometryEvent>& events)
    {
        (void)state;
        (void)context;
        (void)events;
        GeometryStep step{};
        for (const GeometryCommand& command : commands)
        {
            std::visit([&](const auto& cmd) {
                using T = std::decay_t<decltype(cmd)>;
                static_assert(std::is_same_v<T, std::monostate>,
                    "geometry command vocabulary is empty by law (§6.1): "
                    "land a new intent as a named apply_* arrow first");
                (void)cmd;
            }, command);
            step.commands_observed += 1;
        }
        return step;
    }
} // namespace shs::geometry
