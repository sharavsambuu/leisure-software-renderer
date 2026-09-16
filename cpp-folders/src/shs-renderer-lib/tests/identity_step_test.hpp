#pragma once

#include <array>
#include <memory_resource>
#include <span>
#include <variant>
#include <vector>

// Run C: identity gateways observe placeholder commands, never mutate state
// or the caller's event prefix. Replay includes the returned Step value.
template<typename State, typename Command, typename Context, typename Event, typename Step,
         typename Gateway>
bool identity_step_summary(Gateway gateway, const State& initial = {})
{
    static_assert(std::variant_size_v<Command> == 1);
    static_assert(std::variant_size_v<Event> == 1);
    const std::array<Command, 3> commands{};
    const Context context{};
    std::pmr::monotonic_buffer_resource arena_a{4096};
    std::pmr::monotonic_buffer_resource arena_b{4096};
    std::pmr::vector<Event> events{&arena_a};
    std::pmr::vector<Event> replay_events{&arena_b};
    events.push_back(std::monostate{});
    replay_events.push_back(std::monostate{});
    auto state = initial;
    auto replay = initial;
    const auto step = gateway(state, std::span<const Command>{commands}, context, events);
    if (step != Step{3} || state != initial || events.size() != 1) return false;
    if (!std::holds_alternative<std::monostate>(events.front())) return false;
    const auto replay_step = gateway(
        replay, std::span<const Command>{commands}, context, replay_events);
    if (replay_step != step || replay != state || replay_events != events) return false;
    const auto empty_step = gateway(state, std::span<const Command>{}, context, events);
    if (empty_step != Step{0} || state != initial || events != replay_events) return false;
    // Summary is per batch, not accumulated across invocations.
    const auto one_step = gateway(state, std::span<const Command>{commands}.first(1), context, events);
    return one_step == Step{1} && state == initial && events == replay_events;
}
