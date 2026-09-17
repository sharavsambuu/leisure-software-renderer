#pragma once

/*
    SHS RENDERER SAN

    FILE: logic.contract.hpp
    MODULE: domains/logic
    PURPOSE: CORE 1. TYPES — the logic pod's discoverability seam (R5b P3.10).
             Table-driven FSM values: FsmDesc (states + transition table as
             DATA, no callbacks) and FsmState (current/started/elapsed).
             The legacy callback StateMachine class is untouched beside this;
             new logic must use the value design (callbacks cannot replay).
*/

#include <cstdint>
#include <vector>

namespace shs::logic
{
    template <typename TStateId>
    struct FsmTransition
    {
        TStateId   from{};
        TStateId   to{};
        uint32_t   on_event = 0;     // event-gated rule (0 = no event gate)
        float      after_s  = -1.0f; // time-gated rule (< 0 = no time gate)
        int        priority = 0;     // strictly-greater wins (legacy mirror)
    };

    template <typename TStateId>
    struct FsmDesc
    {
        std::vector<TStateId>                states{};
        std::vector<FsmTransition<TStateId>> transitions{};

        bool has_state(const TStateId& id) const
        {
            for (const auto& s : states)
            {
                if (s == id) return true;
            }
            return false;
        }

        // Table-integrity invariant the gateway rides (W-D logic slice,
        // 2026-09-17): every transition rule must reference registered
        // states on BOTH ends. select_rule trusts the table blindly — a
        // rule->to outside `states` would otherwise commit the FSM into an
        // unregistered state. Pure, allocation-free, O(states·transitions).
        bool rules_reference_states() const
        {
            for (const auto& tr : transitions)
            {
                if (!has_state(tr.from) || !has_state(tr.to)) return false;
            }
            return true;
        }
    };

    template <typename TStateId>
    struct FsmState
    {
        TStateId   current{};
        bool       started     = false;
        float      state_time  = 0.0f;

        bool operator==(const FsmState&) const = default;
    };
} // namespace shs::logic
