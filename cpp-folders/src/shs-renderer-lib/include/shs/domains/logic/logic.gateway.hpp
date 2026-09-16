#pragma once

/*
    SHS RENDERER SAN

    FILE: logic.gateway.hpp
    MODULE: domains/logic
    PURPOSE: CORE 4. GATEWAY — the table-driven FSM transition (R5b, P4.1
             house shape + explicit desc threading, renderpath precedent).
             Legacy parity: priority strictly-greater wins (first max on ties),
             dt clamped >= 0, same-state force is a silent no-op, unknown ids
             reject observably instead of returning false.
*/

#include <limits>
#include <memory_resource>
#include <span>

#include "shs/domains/logic/logic.command.hpp"
#include "shs/domains/logic/logic.contract.hpp"
#include "shs/domains/logic/logic.event.hpp"

namespace shs
{
    template <typename TStateId>
    struct LogicContext
    {
    };

    namespace fsm_detail
    {
        template <typename TStateId>
        inline void enter_state(
            FsmState<TStateId>&            state,
            const TStateId&                to,
            std::pmr::vector<FsmEvent<TStateId>>& events)
        {
            state.current    = to;
            state.state_time = 0.0f;
            events.push_back(FsmStateEntered<TStateId>{to});
        }

        template <typename TStateId>
        inline const FsmTransition<TStateId>* select_rule(
            const FsmDesc<TStateId>&  desc,
            const FsmState<TStateId>& state,
            uint32_t                  event_id,
            bool                      use_event,
            float                     elapsed)
        {
            const FsmTransition<TStateId>* selected          = nullptr;
            int                            selected_priority = std::numeric_limits<int>::min();
            for (const auto& tr : desc.transitions)
            {
                if (tr.from != state.current) continue;
                if (use_event)
                {
                    if (tr.on_event == 0 || tr.on_event != event_id) continue;
                }
                else
                {
                    if (tr.after_s < 0.0f || elapsed < tr.after_s) continue;
                }
                if (!selected || tr.priority > selected_priority)
                {
                    selected          = &tr;
                    selected_priority = tr.priority;
                }
            }
            return selected;
        }

        template <typename TStateId>
        inline void apply_transition(
            FsmState<TStateId>&            state,
            const TStateId&                to,
            std::pmr::vector<FsmEvent<TStateId>>& events)
        {
            if (state.current == to) return; // silent no-op (legacy mirror)
            events.push_back(FsmStateExited<TStateId>{state.current});
            enter_state(state, to, events);
        }
    } // namespace fsm_detail

    template <typename TStateId>
    inline void logic_gateway(
        FsmState<TStateId>&                 state,
        std::span<const FsmCommand<TStateId>> commands,
        const FsmDesc<TStateId>&            desc,
        const LogicContext<TStateId>&          context,
        std::pmr::vector<FsmEvent<TStateId>>& events)
    {
        (void)context;
        for (const auto& command : commands)
        {
            if (const auto* start = std::get_if<FsmStart<TStateId>>(&command))
            {
                if (!desc.has_state(start->initial))
                {
                    events.push_back(FsmStartRejected<TStateId>{start->initial});
                    continue;
                }
                state.started    = true;
                state.state_time = 0.0f;
                state.current    = start->initial;
                events.push_back(FsmStarted<TStateId>{start->initial});
                events.push_back(FsmStateEntered<TStateId>{start->initial});
            }
            else if (const auto* signal = std::get_if<FsmSignal<TStateId>>(&command))
            {
                if (!state.started) continue;
                const auto* rule = fsm_detail::select_rule(desc, state, signal->event_id, true, state.state_time);
                if (!rule) continue;
                fsm_detail::apply_transition(state, rule->to, events);
            }
            else if (const auto* force = std::get_if<FsmForce<TStateId>>(&command))
            {
                if (!state.started || !desc.has_state(force->to))
                {
                    if (state.started) events.push_back(FsmTransitionRejected<TStateId>{force->to});
                    continue;
                }
                fsm_detail::apply_transition(state, force->to, events);
            }
            else if (const auto* tick = std::get_if<FsmTick>(&command))
            {
                if (!state.started) continue;
                const float clamped_dt = tick->dt >= 0.0f ? tick->dt : 0.0f;
                state.state_time += clamped_dt;
                const auto* rule = fsm_detail::select_rule(desc, state, 0, false, state.state_time);
                if (!rule) continue;
                fsm_detail::apply_transition(state, rule->to, events);
            }
        }
    }
} // namespace shs
