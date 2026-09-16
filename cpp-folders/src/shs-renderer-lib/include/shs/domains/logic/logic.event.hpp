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

#include "shs/domains/logic/logic.command.hpp"

namespace shs::logic
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

    // K3.2 zero-signal-loss facts (Run B; house answer copied from Run A):
    // a consumed command never emits nothing. Unstarted-machine consumptions
    // and no-rule matches are observable facts, not silent drops.
    struct FsmSignalRejected
    {
        uint32_t event_id = 0;  // signal consumed while the machine is not started

        bool operator==(const FsmSignalRejected&) const = default;
    };

    struct FsmSignalNoRule
    {
        uint32_t event_id = 0;  // signal consumed, no rule matched from the current state

        bool operator==(const FsmSignalNoRule&) const = default;
    };

    struct FsmTickUnstarted
    {
        bool operator==(const FsmTickUnstarted&) const = default;
    };

    struct FsmTickNoRule
    {
        bool operator==(const FsmTickNoRule&) const = default;
    };

    template <typename TStateId>
    struct FsmForceUnstarted
    {
        TStateId to{};

        bool operator==(const FsmForceUnstarted&) const = default;
    };

    // K3.3 same-state facts (Run C): the last silent no-op closed. A consumed
    // command that resolves to the state it is already in emits an *Unchanged
    // fact (the Run A K3.2 house answer, mirrored): no exit/enter pair, one
    // observation, every consumption visible.
    struct FsmSignalUnchanged
    {
        uint32_t event_id = 0;  // signal matched a rule back to the current state

        bool operator==(const FsmSignalUnchanged&) const = default;
    };

    struct FsmTickUnchanged
    {
        bool operator==(const FsmTickUnchanged&) const = default;
    };

    template <typename TStateId>
    struct FsmForceUnchanged
    {
        TStateId to{};

        bool operator==(const FsmForceUnchanged&) const = default;
    };

    template <typename TStateId>
    using FsmEvent = std::variant<
        FsmStarted<TStateId>,
        FsmStateEntered<TStateId>,
        FsmStateExited<TStateId>,
        FsmTransitionRejected<TStateId>,
        FsmStartRejected<TStateId>,
        FsmSignalRejected,
        FsmSignalNoRule,
        FsmTickUnstarted,
        FsmTickNoRule,
        FsmForceUnstarted<TStateId>,
        FsmSignalUnchanged,
        FsmTickUnchanged,
        FsmForceUnchanged<TStateId>>;

    inline const char* fsm_event_name_traffic(const FsmEvent<TrafficLight>& ev)
    {
        if (std::holds_alternative<FsmStarted<TrafficLight>>(ev))            return "started";
        if (std::holds_alternative<FsmStateEntered<TrafficLight>>(ev))       return "state_entered";
        if (std::holds_alternative<FsmStateExited<TrafficLight>>(ev))        return "state_exited";
        if (std::holds_alternative<FsmTransitionRejected<TrafficLight>>(ev)) return "transition_rejected";
        if (std::holds_alternative<FsmStartRejected<TrafficLight>>(ev))      return "start_rejected";
        if (std::holds_alternative<FsmSignalRejected>(ev))                   return "signal_rejected";
        if (std::holds_alternative<FsmSignalNoRule>(ev))                     return "signal_no_rule";
        if (std::holds_alternative<FsmTickUnstarted>(ev))                    return "tick_unstarted";
        if (std::holds_alternative<FsmTickNoRule>(ev))                       return "tick_no_rule";
        if (std::holds_alternative<FsmForceUnstarted<TrafficLight>>(ev))     return "force_unstarted";
        if (std::holds_alternative<FsmSignalUnchanged>(ev))                  return "signal_unchanged";
        if (std::holds_alternative<FsmTickUnchanged>(ev))                    return "tick_unchanged";
        return "force_unchanged";
    }

    static_assert(std::variant_size_v<FsmEvent<TrafficLight>> == 13,
        "logic event vocabulary changed: update name table + EVENT_FLOW.md");
} // namespace shs::logic

namespace shs::logic
{
    using TrafficEvent = FsmEvent<TrafficLight>;
} // namespace shs::logic
