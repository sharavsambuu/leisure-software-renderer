#pragma once

/*
    SHS RENDERER SAN

    FILE: logic.command.hpp
    MODULE: domains/logic
    PURPOSE: CORE 2. COMMAND — the closed FSM command vocabulary (R5b).
             Order in the span is execution order (legacy parity notes):
             - Start begins the machine (unknown id -> StartRejected event).
             - Signal fires event-gated rules from the current state.
             - Force jumps immediately (unknown id -> TransitionRejected).
             - Tick advances time, then fires time-gated rules.
*/

#include <cstdint>
#include <variant>

namespace shs::logic
{
    template <typename TStateId>
    struct FsmStart
    {
        TStateId initial{};

        bool operator==(const FsmStart&) const = default;
    };

    template <typename TStateId>
    struct FsmSignal
    {
        uint32_t event_id = 0;

        bool operator==(const FsmSignal&) const = default;
    };

    template <typename TStateId>
    struct FsmForce
    {
        TStateId to{};

        bool operator==(const FsmForce&) const = default;
    };

    struct FsmTick
    {
        // K1.3 (Run B): frame time lives in LogicContext (one dt per batch,
        // matching the input pod) — the tick is a pure time-advance marker.

        bool operator==(const FsmTick&) const = default;
    };

    template <typename TStateId>
    using FsmCommand = std::variant<
        FsmStart<TStateId>,
        FsmSignal<TStateId>,
        FsmForce<TStateId>,
        FsmTick>;
} // namespace shs::logic

namespace shs::logic
{
    // Instantiation example (traffic-light tests pin the generic design).
    enum class TrafficLight : uint8_t
    {
        Red    = 0,
        Green  = 1,
        Yellow = 2
    };

    using TrafficCommand = FsmCommand<TrafficLight>;
} // namespace shs::logic
