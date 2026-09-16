#include <cstdint>
#include <cstdio>
#include <memory_resource>
#include <variant>
#include <vector>

#include "shs/domains/logic/logic.contract.hpp"
#include "shs/domains/logic/logic.gateway.hpp"
#include "shs/domains/pod_test_kit.hpp"

// Headless tests for the logic pod (R5b P3.10: table-driven FSM pins).
// Links only shs::renderer-values. No callbacks anywhere on this path.
namespace
{
    using Id     = shs::logic::TrafficLight;
    using State  = shs::FsmState<Id>;
    using Action = shs::FsmCommand<Id>;
    using Event  = shs::FsmEvent<Id>;
    using Desc   = shs::FsmDesc<Id>;
    using Context = shs::LogicContext<Id>;

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
                                std::pmr::monotonic_buffer_resource& arena)
    {
        std::pmr::vector<Event> events{&arena};
        shs::logic_gateway(s, std::span<const Action>{commands.data(), commands.size()}, d, Context{}, events);
        return events;
    }

    // Start + signal cycle: Green -> Yellow with exit/enter facts.
    bool test_signal_cycle()
    {
        const Desc desc = make_lights();
        State s{};
        std::pmr::monotonic_buffer_resource arena{4096};
        std::pmr::vector<Event> e0 = run(s, desc, { Action{shs::FsmStart<Id>{Id::Green}} }, arena);
        if (!s.started || s.current != Id::Green || e0.size() != 2) return false;

        std::pmr::vector<Event> e1 = run(s, desc, { Action{shs::FsmSignal<Id>{k_timer}} }, arena);
        if (s.current != Id::Yellow || e1.size() != 2) return false;
        if (!std::holds_alternative<shs::FsmStateExited<Id>>(e1[0])) return false;
        if (!std::holds_alternative<shs::FsmStateEntered<Id>>(e1[1])) return false;
        return true;
    }

    // Time gate: 2s tick in Yellow reaches Red; short tick does nothing.
    bool test_time_gate()
    {
        const Desc desc = make_lights();
        State s{};
        std::pmr::monotonic_buffer_resource arena{4096};
        run(s, desc, { Action{shs::FsmStart<Id>{Id::Yellow}} }, arena);

        std::pmr::vector<Event> e1 = run(s, desc, { Action{shs::FsmTick{1.0f}} }, arena);
        if (s.current != Id::Yellow || !e1.empty()) return false;

        std::pmr::vector<Event> e2 = run(s, desc, { Action{shs::FsmTick{1.5f}} }, arena);
        return s.current == Id::Red && e2.size() == 2;
    }

    // Priority: strictly-greater wins; rejections are observable.
    bool test_priority_and_rejection()
    {
        Desc desc = make_lights();
        desc.transitions.push_back({ Id::Red, Id::Yellow, k_timer, -1.0f, 5 });
        State s{};
        std::pmr::monotonic_buffer_resource arena{4096};
        run(s, desc, { Action{shs::FsmStart<Id>{Id::Red}} }, arena);

        std::pmr::vector<Event> e1 = run(s, desc, { Action{shs::FsmSignal<Id>{k_timer}} }, arena);
        if (s.current != Id::Yellow) return false; // pri-5 beats pri-0 Green rule

        std::pmr::vector<Event> e2 = run(s, desc, { Action{shs::FsmForce<Id>{Id::Green}} }, arena);
        if (s.current != Id::Green) return false;

        State stuck{};
        std::pmr::vector<Event> e3 = run(stuck, desc,
            { Action{shs::FsmStart<Id>{static_cast<Id>(9)}} }, arena);
        if (stuck.started || e3.size() != 1) return false;
        return std::holds_alternative<shs::FsmStartRejected<Id>>(e3[0]);
    }

    // Kit: same signal log twice -> identical states + logs.
    bool test_replay_deterministic()
    {
        const Desc desc = make_lights();
        const State s0{};
        const std::vector<Action> commands{
            Action{shs::FsmStart<Id>{Id::Green}},
            Action{shs::FsmSignal<Id>{k_timer}},
            Action{shs::FsmTick{2.5f}},
        };
        auto reduce = [&](State& s, std::span<const Action> a, const Context& in,
                          std::pmr::vector<Event>& e)
        {
            shs::logic_gateway(s, a, desc, in, e);
        };
        return shs::pod_test::replay_is_deterministic<State, Action, Context, Event>(
            reduce, s0, std::span<const Action>{commands.data(), commands.size()}, Context{});
    }

    bool test_empty_log_stable()
    {
        const Desc desc = make_lights();
        const State s0{};
        auto reduce = [&](State& s, std::span<const Action> a, const Context& in,
                          std::pmr::vector<Event>& e)
        {
            shs::logic_gateway(s, a, desc, in, e);
        };
        return shs::pod_test::empty_log_is_stable<State, Action, Context, Event>(
            reduce, s0, Context{});
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

    if (!ok)
    {
        std::fprintf(stderr, "[logic-tests] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[logic-tests] all tests passed\n");
    return 0;
}
