#pragma once

/*
    SHS RENDERER SAN

    FILE: logic.event.hpp
    MODULE: domains/logic
    PURPOSE: CORE 3. EVENT — raw transition facts (R5b). Enter/exit pairs let
             edges run the legacy on_enter/on_exit effects without callbacks
             in the hot path. Rejections are observable (the legacy bool
             returns, reified). No per-tick event: edges read state_time via
             selector instead (log spam is not observability).
*/

#include "shs/domains/logic/logic.action.hpp"

namespace shs
{
    template <typename TStateId>
    struct FsmStarted
    {
        TStateId state{};

        bool operator==(const FsmStarted&) const = default;
    };

    template <typename TStateId>
    struct FsmStateEntered
    {
        TStateId state{};

        bool operator==(const FsmStateEntered&) const = default;
    };

    template <typename TStateId>
    struct FsmStateExited
    {
        TStateId state{};

        bool operator==(const FsmStateExited&) const = default;
    };

    template <typename TStateId>
    struct FsmTransitionRejected
    {
        TStateId to{};

        bool operator==(const FsmTransitionRejected&) const = default;
    };

    template <typename TStateId>
    struct FsmStartRejected
    {
        TStateId initial{};

        bool operator==(const FsmStartRejected&) const = default;
    };

    template <typename TStateId>
    using FsmEvent = std::variant<
        FsmStarted<TStateId>,
        FsmStateEntered<TStateId>,
        FsmStateExited<TStateId>,
        FsmTransitionRejected<TStateId>,
        FsmStartRejected<TStateId>>;

    inline const char* fsm_event_name_traffic(const FsmEvent<shs::logic::TrafficLight>& ev)
    {
        if (std::holds_alternative<FsmStarted<shs::logic::TrafficLight>>(ev))          return "started";
        if (std::holds_alternative<FsmStateEntered<shs::logic::TrafficLight>>(ev))     return "state_entered";
        if (std::holds_alternative<FsmStateExited<shs::logic::TrafficLight>>(ev))      return "state_exited";
        if (std::holds_alternative<FsmTransitionRejected<shs::logic::TrafficLight>>(ev)) return "transition_rejected";
        return "start_rejected";
    }
} // namespace shs

namespace shs::logic
{
    using TrafficEvent = shs::FsmEvent<TrafficLight>;
} // namespace shs::logic
