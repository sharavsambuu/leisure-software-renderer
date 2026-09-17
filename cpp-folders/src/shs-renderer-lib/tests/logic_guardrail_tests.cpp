#include <cstdint>
#include <cstdio>
#include <memory_resource>
#include <span>
#include <string>
#include <vector>

#include "shs/core/contract_guardrails.hpp"
#include "shs/logic/logic.gateway.hpp"

// W-D logic-pod negative tests (slice 1 of the per-pod traversal). Compiled
// twice by CMake like the C2 renderpath pilot twins:
// - shs_renderer_logic_guardrail_tests        (SHS_CONTRACTS_ENFORCED: hand-
//   broken inputs reach the violation handler with the right kind)
// - shs_renderer_logic_guardrail_release_tests (assume path: valid batches pass
//   through with checks discarded; verified by compilation + run)

namespace
{
    enum class TestState { Idle, Walk, Dead, Ghost /* registered nowhere */ };

    shs::logic::FsmDesc<TestState> make_desc()
    {
        shs::logic::FsmDesc<TestState> desc{};
        desc.states = {TestState::Idle, TestState::Walk, TestState::Dead};
        desc.transitions = {
            shs::logic::FsmTransition<TestState>{TestState::Idle, TestState::Walk, 7, -1.0f, 1},
        };
        return desc;
    }

#if defined(SHS_CONTRACTS_ENFORCED)
    shs::core::contract_kind g_kind{};
    std::string g_expr;
    int g_calls = 0;

    void capture_handler(shs::core::contract_kind kind, const char* expr, const char*, int) noexcept
    {
        ++g_calls;
        g_kind = kind;
        g_expr = expr;
    }
#endif

#if defined(SHS_CONTRACTS_ENFORCED)
    // Negative 1: a transition rule whose target is NOT a registered state is
    // rejected at the rim by the table-integrity precondition (kind=pre),
    // before any command is consumed.
    bool test_rim_rejects_unregistered_rule_target()
    {
        shs::logic::FsmDesc<TestState> desc = make_desc();
        desc.transitions[0].to = TestState::Ghost; // hand-broken table row

        shs::logic::FsmState<TestState> state{};
        state.current = TestState::Idle;
        state.started = true;
        const std::span<const shs::logic::FsmCommand<TestState>> commands{};
        std::pmr::monotonic_buffer_resource arena{1024};
        std::pmr::vector<shs::logic::FsmEvent<TestState>> events{&arena};
        const shs::logic::LogicContext<TestState> context{};

        const int calls_before = g_calls;
        (void)shs::logic::logic_gateway(state, commands, desc, context, events);
        return g_calls == calls_before + 1
            && g_kind == shs::core::contract_kind::pre
            && g_expr.find("rules_reference_states") != std::string::npos;
    }

    // Negative 2: a caller-owned pre-started state resting in an unregistered
    // state with an EMPTY batch — the table precondition passes, and the
    // commit-rim postcondition fires (kind=post).
    bool test_rim_rejects_unregistered_resting_state()
    {
        const shs::logic::FsmDesc<TestState> desc = make_desc();

        shs::logic::FsmState<TestState> state{};
        state.current    = TestState::Ghost; // registered nowhere
        state.started    = true;             // rim postcondition now applies
        const std::span<const shs::logic::FsmCommand<TestState>> commands{};
        std::pmr::monotonic_buffer_resource arena{1024};
        std::pmr::vector<shs::logic::FsmEvent<TestState>> events{&arena};
        const shs::logic::LogicContext<TestState> context{};

        const int calls_before = g_calls;
        (void)shs::logic::logic_gateway(state, commands, desc, context, events);
        return g_calls == calls_before + 1
            && g_kind == shs::core::contract_kind::post
            && g_expr.find("has_state(state.current)") != std::string::npos;
    }
#endif

    // Silence proof (both twins): house-shaped input violates nothing.
    bool test_valid_batch_is_silent()
    {
        shs::logic::FsmDesc<TestState> desc = make_desc();
        shs::logic::FsmState<TestState> state{};
        const shs::logic::FsmCommand<TestState> commands[]
            = {shs::logic::FsmStart<TestState>{TestState::Idle},
               shs::logic::FsmSignal<TestState>{7}}; // matches the Idle->Walk rule
        const std::span<const shs::logic::FsmCommand<TestState>> span{commands};
        std::pmr::monotonic_buffer_resource arena{1024};
        std::pmr::vector<shs::logic::FsmEvent<TestState>> events{&arena};
        const shs::logic::LogicContext<TestState> context{};

#if defined(SHS_CONTRACTS_ENFORCED)
        const int calls_before = g_calls;
#endif
        const auto step = shs::logic::logic_gateway(state, span, desc, context, events);
        return step.commands_applied == 2 && events.size() == 4
#if defined(SHS_CONTRACTS_ENFORCED)
            && g_calls == calls_before
#endif
            ;
    }
} // namespace

int main()
{
#if defined(SHS_CONTRACTS_ENFORCED)
    shs::core::set_contract_violation_handler(&capture_handler);
    bool ok = true;
    ok = test_rim_rejects_unregistered_rule_target() && ok;
    ok = test_rim_rejects_unregistered_resting_state() && ok;
    ok = test_valid_batch_is_silent() && ok;
    shs::core::set_contract_violation_handler(nullptr); // restore the default
    if (!ok)
    {
        std::fprintf(stderr, "[logic-guardrail-tests] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[logic-guardrail-tests] all logic negative tests passed (enforced)\n");
#else
    // Release semantics: house input compiles through the annotated seam with
    // checks discarded (assume path) and behaves identically (Rule 4.1).
    if (!test_valid_batch_is_silent())
    {
        std::fprintf(stderr, "[logic-guardrail-tests-release] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[logic-guardrail-tests-release] all logic tests passed\n");
#endif
    return 0;
}
