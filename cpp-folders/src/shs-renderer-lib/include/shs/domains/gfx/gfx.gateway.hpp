#pragma once

/*
    SHS RENDERER SAN

    FILE: gfx.gateway.hpp
    MODULE: domains/gfx
    PURPOSE: CORE 4. GATEWAY — the pinned identity transition, Kleisli house
             shape (Run C, K1.5): (State, span<Commands>, Context, arena)
             -> GfxStep. Handles and pixel buffers are values; allocation
             lifecycles reduce through commands only after the registry edge
             migration; the empty-vocabulary static_assert makes the silence
             provable (§6.1). Pinned by gfx_tests.
*/

#include <cstdint>
#include <memory_resource>
#include <span>
#include <type_traits>
#include <variant>

#include "shs/domains/gfx/gfx.command.hpp"
#include "shs/domains/gfx/gfx.contract.hpp"
#include "shs/domains/gfx/gfx.event.hpp"

namespace shs::gfx
{
    struct GfxState
    {
        bool operator==(const GfxState&) const = default;
    };

    struct GfxContext
    {
    };

    // Batch outcome summary (house shape per kdba_kleisli_migration_plan.md).
    // An identity pod applies nothing — it counts what passed the seam.
    struct GfxStep
    {
        uint32_t commands_observed = 0;

        bool operator==(const GfxStep&) const = default;
    };

    inline GfxStep gfx_gateway(
        GfxState&                   state,
        std::span<const GfxCommand>  commands,
        const GfxContext&      context,
        std::pmr::vector<GfxEvent>& events)
    {
        (void)state;
        (void)context;
        (void)events;
        GfxStep step{};
        for (const GfxCommand& command : commands)
        {
            std::visit([&](const auto& cmd) {
                using T = std::decay_t<decltype(cmd)>;
                static_assert(std::is_same_v<T, std::monostate>,
                    "gfx command vocabulary is empty by law (§6.1): "
                    "land a new intent as a named apply_* arrow first");
                (void)cmd;
            }, command);
            step.commands_observed += 1;
        }
        return step;
    }
} // namespace shs::gfx
