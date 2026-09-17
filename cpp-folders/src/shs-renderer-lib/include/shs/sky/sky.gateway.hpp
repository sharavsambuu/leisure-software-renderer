#pragma once

/*
    SHS RENDERER SAN

    FILE: sky.gateway.hpp
    MODULE: domains/sky
    PURPOSE: CORE 4. GATEWAY — the pinned identity transition, Kleisli house
             shape (Run C, K1.5): (State, span<Commands>, Context, arena)
             -> SkyStep. Sky models are caller-owned values sampled by
             edges/passes; the empty-vocabulary static_assert makes the
             silence provable (§6.1). Pinned by sky_tests.
*/

#include <cstdint>
#include <memory_resource>
#include <span>
#include <type_traits>
#include <variant>

#include "shs/sky/sky.command.hpp"
#include "shs/sky/sky.contract.hpp"
#include "shs/sky/sky.event.hpp"

namespace shs::sky
{
    struct SkyState
    {
        bool operator==(const SkyState&) const = default;
    };

    struct SkyContext
    {
    };

    // Batch outcome summary (house shape per kdba_kleisli_migration_plan.md).
    // An identity pod applies nothing — it counts what passed the seam.
    struct SkyStep
    {
        uint32_t commands_observed = 0;

        bool operator==(const SkyStep&) const = default;
    };

    inline SkyStep sky_gateway(
        SkyState&                   state,
        std::span<const SkyCommand>  commands,
        const SkyContext&      context,
        std::pmr::vector<SkyEvent>& events)
    {
        (void)state;
        (void)context;
        (void)events;
        SkyStep step{};
        for (const SkyCommand& command : commands)
        {
            std::visit([&](const auto& cmd) {
                using T = std::decay_t<decltype(cmd)>;
                static_assert(std::is_same_v<T, std::monostate>,
                    "sky command vocabulary is empty by law (§6.1): "
                    "land a new intent as a named apply_* arrow first");
                (void)cmd;
            }, command);
            step.commands_observed += 1;
        }
        return step;
    }
} // namespace shs::sky
