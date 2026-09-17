#pragma once

/*
    SHS RENDERER SAN

    FILE: frame.gateway.hpp
    MODULE: domains/frame
    PURPOSE: CORE 4. GATEWAY — the pinned identity transition, Kleisli house
             shape (Run C, K1.5): (State, span<Commands>, Context, arena)
             -> FrameStep. The command vocabulary is EMPTY by law (§6.1):
             frame configuration is transition-free. The visit + static_assert
             makes the silence PROVABLE: there is nothing to drop, and a new
             command type cannot compile past this gateway unhandled.
             Pinned by frame_tests (kit: replay + empty log), not by convention.
*/

#include <cstdint>
#include <memory_resource>
#include <span>
#include <type_traits>
#include <variant>

#include "shs/core/contract_guardrails.hpp"
#include "shs/render/frame/frame.command.hpp"
#include "shs/render/frame/frame.contract.hpp"
#include "shs/render/frame/frame.event.hpp"

namespace shs::frame
{
    struct FrameContext
    {
    };

    // Batch outcome summary (house shape per kdba_kleisli_migration_plan.md).
    // An identity pod applies nothing — it counts what passed the seam.
    struct FrameStep
    {
        uint32_t commands_observed = 0;

        bool operator==(const FrameStep&) const = default;
    };

    inline FrameStep frame_gateway(
        FrameParams&                     state,
        std::span<const FrameCommand>     commands,
        const FrameContext&         context,
        std::pmr::vector<FrameEvent>&    events)
    {
        (void)state;
        (void)context;
        (void)events;
        FrameStep step{};
        for (const FrameCommand& command : commands)
        {
            std::visit([&](const auto& cmd) {
                using T = std::decay_t<decltype(cmd)>;
                static_assert(std::is_same_v<T, std::monostate>,
                    "frame command vocabulary is empty by law (§6.1): "
                    "land a new intent as a named apply_* arrow first");
                (void)cmd;
            }, command);
            step.commands_observed += 1;
        }
        // Rim postcondition (W-D frame slice, 2026-09-17): zero-signal-loss
        // for the identity pod — every consumed command was observed, none
        // dropped or transformed. Trivial by construction today; the guard
        // binds the loop when the first real knob (exposure, debug view)
        // replaces monostate and the visit grows real arrows.
        SHS_POST(step.commands_observed == commands.size());
        return step;
    }
} // namespace shs::frame
