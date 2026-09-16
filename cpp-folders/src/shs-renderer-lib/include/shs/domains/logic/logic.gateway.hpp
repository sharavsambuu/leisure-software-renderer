#pragma once

/*
    SHS RENDERER SAN

    FILE: logic.gateway.hpp
    MODULE: domains/logic
    PURPOSE: CORE 4. GATEWAY — the table-driven FSM transition, Kleisli house
             shape (Run B, K1.3): (State, span<Commands>, Desc, Context, arena)
             -> FsmStep. Zero-signal-loss (K3.2, house answer copied from
             Run A): every consumed command emits at least one fact —
             transitions emit enter/exit, unstarted consumptions emit
             Unstarted/Rejected facts, no-rule matches emit NoRule facts,
             invalid ids emit rejections. Legacy parity: priority
             strictly-greater wins (first max on ties), dt clamped >= 0,
             same-state force is a documented silent no-op (legacy mirror),
             unknown ids reject observably instead of returning false.
*/

#include <cstdint>
#include <limits>
#include <memory_resource>
#include <span>
#include <type_traits>
#include <variant>

#include "shs/domains/logic/logic.command.hpp"
#include "shs/domains/logic/logic.contract.hpp"
#include "shs/domains/logic/logic.event.hpp"

namespace shs::logic
{
    // Batch context: frame time arrives here (K1.3 — one dt per batch,
    // matching the input pod); the tick command is a pure advance marker.
    template <typename TStateId>
    struct LogicContext
    {
        float dt = 0.0f;
    };

    // Batch outcome summary (house shape per kdba_kleisli_migration_plan.md;
    // the batch rim is infallible — invalid ids and no-rule matches are
    // materialized facts, never an invented error enum).
    struct FsmStep
    {
        uint32_t commands_applied  = 0;  // commands that mutated pod state
        uint32_t facts_observed    = 0;  // consumed with a fact, no transition
        uint32_t commands_rejected = 0;  // invalid lifecycle/identity consumption

        bool operator==(const FsmStep&) const = default;
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
        inline bool apply_transition(
            FsmState<TStateId>&            state,
            const TStateId&                to,
            std::pmr::vector<FsmEvent<TStateId>>& events)
        {
            if (state.current == to) return false; // documented silent no-op (legacy mirror)
            events.push_back(FsmStateExited<TStateId>{state.current});
            enter_state(state, to, events);
            return true;
        }

        // --- named per-intent arrows (K2.2: transition bodies live here; the
        // public gateway below is only the assembly point) ----------------

        template <typename TStateId>
        inline void apply_start(
            FsmState<TStateId>& state,
            const FsmStart<TStateId>& cmd,
            const FsmDesc<TStateId>& desc,
            std::pmr::vector<FsmEvent<TStateId>>& events,
            FsmStep& step)
        {
            if (!desc.has_state(cmd.initial))
            {
                step.commands_rejected += 1;
                events.push_back(FsmStartRejected<TStateId>{cmd.initial});
                return;
            }
            state.started    = true;
            state.state_time = 0.0f;
            state.current    = cmd.initial;
            step.commands_applied += 1;
            events.push_back(FsmStarted<TStateId>{cmd.initial});
            events.push_back(FsmStateEntered<TStateId>{cmd.initial});
        }

        template <typename TStateId>
        inline void apply_signal(
            FsmState<TStateId>& state,
            const FsmSignal<TStateId>& cmd,
            const FsmDesc<TStateId>& desc,
            std::pmr::vector<FsmEvent<TStateId>>& events,
            FsmStep& step)
        {
            if (!state.started)
            {
                step.commands_rejected += 1;
                events.push_back(FsmSignalRejected{cmd.event_id});
                return;
            }
            const auto* rule = select_rule(desc, state, cmd.event_id, true, state.state_time);
            if (!rule)
            {
                step.facts_observed += 1;
                events.push_back(FsmSignalNoRule{cmd.event_id});
                return;
            }
            if (apply_transition(state, rule->to, events))
            {
                step.commands_applied += 1;
            }
            else
            {
                step.facts_observed += 1; // same-state rule: documented legacy no-op
            }
        }

        template <typename TStateId>
        inline void apply_force(
            FsmState<TStateId>& state,
            const FsmForce<TStateId>& cmd,
            const FsmDesc<TStateId>& desc,
            std::pmr::vector<FsmEvent<TStateId>>& events,
            FsmStep& step)
        {
            if (!state.started)
            {
                step.commands_rejected += 1;
                events.push_back(FsmForceUnstarted<TStateId>{cmd.to});
                return;
            }
            if (!desc.has_state(cmd.to))
            {
                step.commands_rejected += 1;
                events.push_back(FsmTransitionRejected<TStateId>{cmd.to});
                return;
            }
            if (apply_transition(state, cmd.to, events))
            {
                step.commands_applied += 1;
            }
            else
            {
                step.facts_observed += 1; // same-state force: documented legacy no-op
            }
        }

        template <typename TStateId>
        inline void apply_tick(
            FsmState<TStateId>& state,
            const FsmTick& cmd,
            const LogicContext<TStateId>& context,
            const FsmDesc<TStateId>& desc,
            std::pmr::vector<FsmEvent<TStateId>>& events,
            FsmStep& step)
        {
            (void)cmd;
            if (!state.started)
            {
                step.commands_rejected += 1;
                events.push_back(FsmTickUnstarted{});
                return;
            }
            const float clamped_dt = context.dt >= 0.0f ? context.dt : 0.0f;
            state.state_time += clamped_dt;
            step.commands_applied += 1; // time advanced (state mutated)
            const auto* rule = select_rule(desc, state, 0, false, state.state_time);
            if (!rule)
            {
                events.push_back(FsmTickNoRule{});
                return;
            }
            apply_transition(state, rule->to, events);
        }
    } // namespace fsm_detail

    template <typename TStateId>
    inline FsmStep logic_gateway(
        FsmState<TStateId>& state,
        std::span<const FsmCommand<TStateId>> commands,
        const FsmDesc<TStateId>& desc,
        const LogicContext<TStateId>& context,
        std::pmr::vector<FsmEvent<TStateId>>& events)
    {
        FsmStep step{};
        for (const FsmCommand<TStateId>& command : commands)
        {
            std::visit([&](const auto& cmd) {
                using T = std::decay_t<decltype(cmd)>;

                if constexpr (std::is_same_v<T, FsmStart<TStateId>>)
                {
                    fsm_detail::apply_start(state, cmd, desc, events, step);
                }
                else if constexpr (std::is_same_v<T, FsmSignal<TStateId>>)
                {
                    fsm_detail::apply_signal(state, cmd, desc, events, step);
                }
                else if constexpr (std::is_same_v<T, FsmForce<TStateId>>)
                {
                    fsm_detail::apply_force(state, cmd, desc, events, step);
                }
                else if constexpr (std::is_same_v<T, FsmTick>)
                {
                    fsm_detail::apply_tick(state, cmd, context, desc, events, step);
                }
            }, command);
        }
        return step;
    }
} // namespace shs::logic
