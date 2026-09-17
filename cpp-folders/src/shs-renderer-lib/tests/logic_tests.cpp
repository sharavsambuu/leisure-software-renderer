#include <cstdint>
#include <cstdio>
#include <memory_resource>
#include <variant>
#include <vector>

#include "shs/logic/logic.contract.hpp"
#include "shs/logic/logic.gateway.hpp"
#include "shs/core/testing/pod_test_kit.hpp"

// Headless tests for the logic pod (R5b P3.10: table-driven FSM pins).
// Links only shs::renderer-values. No callbacks anywhere on this path.
namespace
{
    using Id      = shs::logic::TrafficLight;
    using State   = shs::logic::FsmState<Id>;
    using Action  = shs::logic::FsmCommand<Id>;
    using Event   = shs::logic::FsmEvent<Id>;
    using Desc    = shs::logic::FsmDesc<Id>;
    using Context = shs::logic::LogicContext<Id>;

    constexpr uint32_t k_timer = 1;

    Desc make_lights()
    {
        Desc desc{};
        desc.states = { Id::Red, Id::Green, Id::Yellow };
        desc.transitions = {
            { Id::Green, Id::Yellow, k_timer, -1.0f, 0 },
            { Id::Yellow, Id::Red, 0, 2.0f, 0 },
            { Id::Red, Id::Green, k_timer, -1.0f, 0 },
        };
        return desc;
    }

    std::pmr::vector<Event> run(State& s, const Desc& d, const std::vector<Action>& commands,
                                std::pmr::monotonic_buffer_resource& arena, const Context& in = Context{})
    {
        std::pmr::vector<Event> events{&arena};
        shs::logic::logic_gateway(s, std::span<const Action>{commands.data(), commands.size()}, d, in, events);
        return events;
    }

    // Start + signal cycle: Green -> Yellow with exit/enter facts.
    bool test_signal_cycle()
    {
        const Desc desc = make_lights();
        State s{};
        std::pmr::monotonic_buffer_resource arena{4096};
        std::pmr::vector<Event> e0 = run(s, desc, { Action{shs::logic::FsmStart<Id>{Id::Green}} }, arena);
        if (!s.started || s.current != Id::Green || e0.size() != 2) return false;

        std::pmr::vector<Event> e1 = run(s, desc, { Action{shs::logic::FsmSignal<Id>{k_timer}} }, arena);
        if (s.current != Id::Yellow || e1.size() != 2) return false;
        if (!std::holds_alternative<shs::logic::FsmStateExited<Id>>(e1[0])) return false;
        if (!std::holds_alternative<shs::logic::FsmStateEntered<Id>>(e1[1])) return false;
        return true;
    }

    // Time gate: 2s tick in Yellow reaches Red; short tick does nothing.
    bool test_time_gate()
    {
        const Desc desc = make_lights();
        State s{};
        std::pmr::monotonic_buffer_resource arena{4096};
        run(s, desc, { Action{shs::logic::FsmStart<Id>{Id::Yellow}} }, arena);

        std::pmr::vector<Event> e1 = run(s, desc, { Action{shs::logic::FsmTick{}} }, arena, Context{1.0f});
        if (s.current != Id::Yellow || e1.size() != 1) return false; // K3.2: no-rule is a fact now
        if (!std::holds_alternative<shs::logic::FsmTickNoRule>(e1[0])) return false;

        std::pmr::vector<Event> e2 = run(s, desc, { Action{shs::logic::FsmTick{}} }, arena, Context{1.5f});
        return s.current == Id::Red && e2.size() == 2;
    }

    // Priority: strictly-greater wins; rejections are observable.
    bool test_priority_and_rejection()
    {
        Desc desc = make_lights();
        desc.transitions.push_back({ Id::Red, Id::Yellow, k_timer, -1.0f, 5 });
        State s{};
        std::pmr::monotonic_buffer_resource arena{4096};
        run(s, desc, { Action{shs::logic::FsmStart<Id>{Id::Red}} }, arena);

        std::pmr::vector<Event> e1 = run(s, desc, { Action{shs::logic::FsmSignal<Id>{k_timer}} }, arena);
        if (s.current != Id::Yellow) return false; // pri-5 beats pri-0 Green rule

        std::pmr::vector<Event> e2 = run(s, desc, { Action{shs::logic::FsmForce<Id>{Id::Green}} }, arena);
        if (s.current != Id::Green) return false;

        State stuck{};
        std::pmr::vector<Event> e3 = run(stuck, desc,
            { Action{shs::logic::FsmStart<Id>{static_cast<Id>(9)}} }, arena);
        if (stuck.started || e3.size() != 1) return false;
        return std::holds_alternative<shs::logic::FsmStartRejected<Id>>(e3[0]);
    }

    // Kit: same signal log twice -> identical states + logs.
    bool test_replay_deterministic()
    {
        const Desc desc = make_lights();
        const State s0{};
        const std::vector<Action> commands{
            Action{shs::logic::FsmStart<Id>{Id::Green}},
            Action{shs::logic::FsmSignal<Id>{k_timer}},
            Action{shs::logic::FsmTick{}},
        };
        auto reduce = [&](State& s, std::span<const Action> a, const Context& in,
                          std::pmr::vector<Event>& e)
        {
            shs::logic::logic_gateway(s, a, desc, in, e);
        };
        return shs::pod_test::replay_is_deterministic<State, Action, Context, Event>(
            reduce, s0, std::span<const Action>{commands.data(), commands.size()}, Context{2.5f});
    }

    bool test_empty_log_stable()
    {
        const Desc desc = make_lights();
        const State s0{};
        auto reduce = [&](State& s, std::span<const Action> a, const Context& in,
                          std::pmr::vector<Event>& e)
        {
            shs::logic::logic_gateway(s, a, desc, in, e);
        };
        return shs::pod_test::empty_log_is_stable<State, Action, Context, Event>(
            reduce, s0, Context{});
    }

    // K3.2 (Run B): zero-signal-loss — unstarted consumptions and no-rule
    // matches emit facts, never silence.
    bool test_zero_signal_loss()
    {
        const Desc desc = make_lights();
        State s{};
        std::pmr::monotonic_buffer_resource arena{4096};

        // Unstarted machine: signal / tick / force are each rejected with a fact.
        std::pmr::vector<Event> e1 = run(s, desc, { Action{shs::logic::FsmSignal<Id>{k_timer}} }, arena);
        if (e1.size() != 1 || !std::holds_alternative<shs::logic::FsmSignalRejected>(e1[0])) return false;

        std::pmr::vector<Event> e2 = run(s, desc, { Action{shs::logic::FsmTick{}} }, arena);
        if (e2.size() != 1 || !std::holds_alternative<shs::logic::FsmTickUnstarted>(e2[0])) return false;

        std::pmr::vector<Event> e3 = run(s, desc, { Action{shs::logic::FsmForce<Id>{Id::Green}} }, arena);
        if (e3.size() != 1 || !std::holds_alternative<shs::logic::FsmForceUnstarted<Id>>(e3[0])) return false;
        if (s.started) return false;

        // Started machine: a no-rule signal is observed, not silent.
        run(s, desc, { Action{shs::logic::FsmStart<Id>{Id::Green}} }, arena);
        std::pmr::vector<Event> e4 = run(s, desc, { Action{shs::logic::FsmSignal<Id>{77}} }, arena);
        if (e4.size() != 1 || !std::holds_alternative<shs::logic::FsmSignalNoRule>(e4[0])) return false;
        return true;
    }
    bool test_same_state_facts()
    {
        Desc desc{};
        desc.states = {Id::Red};
        desc.transitions = {
            {Id::Red, Id::Red, k_timer, -1.0f, 0},
            {Id::Red, Id::Red, 0, 0.0f, 0},
        };
        State s{};
        std::pmr::monotonic_buffer_resource arena{4096};
        run(s, desc, {Action{shs::logic::FsmStart<Id>{Id::Red}}}, arena);
        s.state_time = 2.0f;
        const State before = s;
        const std::vector<Action> commands{
            shs::logic::FsmSignal<Id>{k_timer},
            shs::logic::FsmForce<Id>{Id::Red},
            shs::logic::FsmTick{},
        };
        std::pmr::vector<Event> events{&arena};
        const auto step = shs::logic::logic_gateway(
            s, std::span<const Action>{commands}, desc, Context{0.5f}, events);
        if (step != shs::logic::FsmStep{1, 2, 0} || events.size() != 3) return false;
        if (events[0] != Event{shs::logic::FsmSignalUnchanged{k_timer}}) return false;
        if (events[1] != Event{shs::logic::FsmForceUnchanged<Id>{Id::Red}}) return false;
        if (events[2] != Event{shs::logic::FsmTickUnchanged{}}) return false;
        if (!s.started || s.current != before.current || s.state_time != 2.5f) return false;

        State replay = before;
        std::pmr::vector<Event> replay_events{&arena};
        const auto replay_step = shs::logic::logic_gateway(
            replay, std::span<const Action>{commands}, desc, Context{0.5f}, replay_events);
        if (replay != s || replay_events != events || replay_step != step) return false;
        events.clear();
        const auto empty_step = shs::logic::logic_gateway(
            s, std::span<const Action>{}, desc, Context{0.5f}, events);
        return empty_step == shs::logic::FsmStep{} && events.empty() && s == replay;
    }
} // namespace

int main()
{
    bool ok = true;
    auto run = [&](const char* name, bool result)
    {
        std::fprintf(stderr, "[logic-tests] %s: %s\n", name, result ? "pass" : "FAIL");
        ok = result && ok;
    };

    run("signal_cycle", test_signal_cycle());
    run("time_gate", test_time_gate());
    run("priority_and_rejection", test_priority_and_rejection());
    run("replay_deterministic", test_replay_deterministic());
    run("empty_log_stable", test_empty_log_stable());
    run("zero_signal_loss", test_zero_signal_loss());
    run("same_state_facts", test_same_state_facts());

    if (!ok)
    {
        std::fprintf(stderr, "[logic-tests] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[logic-tests] all tests passed\n");
    return 0;
}
