#pragma once

/*
    SHS RENDERER SAN

    FILE: pod_test_kit.hpp
    MODULE: domains (shared test utility; R3 P4.2)
    PURPOSE: Three-line pod conformance: replay determinism + empty-log
             stability over the P4.1 house signature
             (State, span<const Action>, Context, arena Events).
             Requires State and Event operator== (value semantics proof).
*/

#include <cstddef>
#include <memory_resource>
#include <span>
#include <vector>

namespace shs::pod_test
{
    namespace detail
    {
        inline constexpr std::size_t k_test_arena_bytes = 4096;
    } // namespace detail

    // Same command log applied twice must yield equal states + equal event logs.
    template<typename State, typename Action, typename Context, typename Event, typename GatewayFn>
    inline bool replay_is_deterministic(
        GatewayFn&&                       reduce,
        const State&                     s0,
        std::span<const Action>          commands,
        const Context&                    context)
    {
        std::pmr::monotonic_buffer_resource arena_a{detail::k_test_arena_bytes};
        std::pmr::monotonic_buffer_resource arena_b{detail::k_test_arena_bytes};
        std::pmr::vector<Event> events_a{&arena_a};
        std::pmr::vector<Event> events_b{&arena_b};

        State sa = s0;
        State sb = s0;
        reduce(sa, commands, context, events_a);
        reduce(sb, commands, context, events_b);

        return sa == sb && events_a == events_b;
    }

    // Empty command log must leave state bit-identical and emit nothing.
    template<typename State, typename Action, typename Context, typename Event, typename GatewayFn>
    inline bool empty_log_is_stable(
        GatewayFn&&                       reduce,
        const State&                     s0,
        const Context&                    context)
    {
        std::pmr::monotonic_buffer_resource arena{detail::k_test_arena_bytes};
        std::pmr::vector<Event> events{&arena};

        State s = s0;
        const std::vector<Action> none{};
        reduce(s, std::span<const Action>{none.data(), none.size()}, context, events);

        return s == s0 && events.empty();
    }
} // namespace shs::pod_test
