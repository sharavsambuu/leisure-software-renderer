#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: state_machine.hpp
    МОДУЛЬ: logic
    ЗОРИЛГО: Төрөл (enum/class)-оор ID-лагдсан төлөвүүдтэй, callback дээр суурилсан
            өргөтгөх боломжтой finite state machine (FSM) хэрэгжүүлэлт.
*/


#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace shs
{
    template <typename TStateId, typename TContext>
    class StateMachine
    {
    public:
        using StateId = TStateId;

        struct StateCallbacks
        {
            std::function<void(TContext&)> on_enter{};
            std::function<void(TContext&, float dt, float elapsed)> on_update{};
            std::function<void(TContext&)> on_exit{};
        };

        struct TransitionRule
        {
            StateId from{};
            StateId to{};
            std::function<bool(const TContext&, float elapsed)> predicate{};
            int priority = 0;
        };

        bool add_state(StateId id, StateCallbacks callbacks = {})
        {
            if (find_state_index(id).has_value()) return false;
            states_.push_back(StateEntry{id, std::move(callbacks)});
            return true;
        }

        bool has_state(StateId id) const
        {
            return find_state_index(id).has_value();
        }

        bool add_transition(TransitionRule rule)
        {
            if (!rule.predicate) return false;
            transitions_.push_back(std::move(rule));
            return true;
        }

        bool add_transition(
            StateId from,
            StateId to,
            std::function<bool(const TContext&, float elapsed)> predicate,
            int priority = 0
        )
        {
            TransitionRule rule{};
            rule.from = from;
            rule.to = to;
            rule.predicate = std::move(predicate);
            rule.priority = priority;
            return add_transition(std::move(rule));
        }

        void clear()
        {
            states_.clear();
            transitions_.clear();
            started_ = false;
            state_time_ = 0.0f;
            pending_transition_.reset();
        }

        bool start(StateId initial_state, TContext& ctx)
        {
            if (!has_state(initial_state)) return false;
            started_ = true;
            current_state_ = initial_state;
            state_time_ = 0.0f;
            pending_transition_.reset();
            call_on_enter(ctx, initial_state);
            return true;
        }

        bool started() const
        {
            return started_;
        }

        std::optional<StateId> current_state() const
        {
            if (!started_) return std::nullopt;
            return current_state_;
        }

        float state_time() const
        {
            return state_time_;
        }

        void request_transition(StateId to)
        {
            pending_transition_ = to;
        }

        bool transition_to(StateId to, TContext& ctx)
        {
            if (!started_) return false;
            if (!has_state(to)) return false;
            if (current_state_ == to)
            {
                pending_transition_.reset();
                return true;
            }
            call_on_exit(ctx, current_state_);
            current_state_ = to;
            state_time_ = 0.0f;
            pending_transition_.reset();
            call_on_enter(ctx, current_state_);
            return true;
        }

        void tick(TContext& ctx, float dt)
        {
            if (!started_) return;

            const float clamped_dt = std::max(0.0f, dt);
            call_on_update(ctx, current_state_, clamped_dt, state_time_);
            state_time_ += clamped_dt;

            if (pending_transition_.has_value())
            {
                transition_to(*pending_transition_, ctx);
                return;
            }

            const TransitionRule* selected = select_transition(ctx);
            if (!selected) return;
            transition_to(selected->to, ctx);
        }

    private:
        struct StateEntry
        {
            StateId id{};
            StateCallbacks callbacks{};
        };

        std::optional<std::size_t> find_state_index(StateId id) const
        {
            for (std::size_t i = 0; i < states_.size(); ++i)
            {
                if (states_[i].id == id) return i;
            }
            return std::nullopt;
        }

        StateCallbacks* find_callbacks(StateId id)
        {
            const auto idx = find_state_index(id);
            if (!idx.has_value()) return nullptr;
            return &states_[*idx].callbacks;
        }

        const StateCallbacks* find_callbacks(StateId id) const
        {
            const auto idx = find_state_index(id);
            if (!idx.has_value()) return nullptr;
            return &states_[*idx].callbacks;
        }

        void call_on_enter(TContext& ctx, StateId id)
        {
            StateCallbacks* cb = find_callbacks(id);
            if (!cb || !cb->on_enter) return;
            cb->on_enter(ctx);
        }

        void call_on_update(TContext& ctx, StateId id, float dt, float elapsed)
        {
            StateCallbacks* cb = find_callbacks(id);
            if (!cb || !cb->on_update) return;
            cb->on_update(ctx, dt, elapsed);
        }

        void call_on_exit(TContext& ctx, StateId id)
        {
            StateCallbacks* cb = find_callbacks(id);
            if (!cb || !cb->on_exit) return;
            cb->on_exit(ctx);
        }

        const TransitionRule* select_transition(const TContext& ctx) const
        {
            const TransitionRule* selected = nullptr;
            int selected_priority = std::numeric_limits<int>::min();
            for (const TransitionRule& tr : transitions_)
            {
                if (tr.from != current_state_) continue;
                if (!tr.predicate || !tr.predicate(ctx, state_time_)) continue;
                if (!selected || tr.priority > selected_priority)
                {
                    selected = &tr;
                    selected_priority = tr.priority;
                }
            }
            return selected;
        }

        std::vector<StateEntry> states_{};
        std::vector<TransitionRule> transitions_{};
        bool started_ = false;
        StateId current_state_{};
        float state_time_ = 0.0f;
        std::optional<StateId> pending_transition_{};
    };
}

