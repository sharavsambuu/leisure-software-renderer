#include <cstdint>
#include <cstdio>
#include <memory_resource>
#include <span>
#include <vector>

#include "shs/core/contract_guardrails.hpp"
#include "shs/app/session_orchestrator.gateway.hpp"

// W-D input-pod guardrail tests (slice 9). The input pod owns TRANSLATION
// only (latch + emitters); the application rim for its intents is
// shs::app::session_orchestrate. Compiled twice by CMake like the logic and
// frame twins:
// - shs_renderer_input_guardrail_tests        (SHS_CONTRACTS_ENFORCED: a
//   valid mixed batch applies every command, emits one fact per command,
//   and never reaches the violation handler — the rim is infallible by law)
// - shs_renderer_input_guardrail_release_tests (assume path: checks
//   discarded, identical application behavior — Rule 4.1)
//
// The negative leg is not reachable through the public seam: the command
// variant is closed (5 alternatives, static_assert-pinned) and every one is
// valid, so no input can violate the rim POST. The exhaustiveness tail is
// compile-time: a new alternative fails to build past the dispatch.

namespace
{
    shs::app::SessionState make_session()
    {
        shs::app::SessionState state{};
        state.camera.pos = glm::vec3(0.0f);
        return state;
    }

#if defined(SHS_CONTRACTS_ENFORCED)
    int g_calls = 0;

    void counting_handler(shs::core::contract_kind, const char*, const char*, int) noexcept
    {
        ++g_calls;
    }
#endif

    // Silence + completeness proof (both twins): every intent in the closed
    // vocabulary is valid, so a mixed batch is fully applied with exactly
    // one fact per command and (enforced) zero handler calls.
    bool test_valid_mixed_batch_is_silent_and_complete()
    {
        shs::app::SessionState state = make_session();
        const shs::RuntimeCommand commands[] = {
            shs::make_move_local_intent(glm::vec3(0.0f, 0.0f, 1.0f), 2.0f),
            shs::make_look_intent(0.5f, 0.25f, 0.01f),
            shs::make_toggle_light_shafts_intent(),
            shs::make_toggle_bot_intent(),
            shs::make_quit_intent(),
        };
        const std::span<const shs::RuntimeCommand> span{commands};
        const shs::input::InputContext context{0.016f};
        std::pmr::monotonic_buffer_resource arena{4096};
        std::pmr::vector<shs::input::InputEvent> events{&arena};

#if defined(SHS_CONTRACTS_ENFORCED)
        const int calls_before = g_calls;
#endif
        const auto step = shs::app::session_orchestrate(state, span, context, events);
        const bool moved   = state.camera.pos.z > 0.0f;
        const bool toggled = !state.enable_light_shafts && state.bot_enabled
            && state.quit_requested;
        return step.commands_applied == 5 && events.size() == 5 && moved && toggled
#if defined(SHS_CONTRACTS_ENFORCED)
            && g_calls == calls_before
#endif
            ;
    }
} // namespace

int main()
{
#if defined(SHS_CONTRACTS_ENFORCED)
    shs::core::set_contract_violation_handler(&counting_handler);
    const bool ok = test_valid_mixed_batch_is_silent_and_complete();
    shs::core::set_contract_violation_handler(nullptr); // restore the default
    if (!ok)
    {
        std::fprintf(stderr, "[input-guardrail-tests] FAILED\\n");
        return 1;
    }
    std::fprintf(stderr, "[input-guardrail-tests] all input guardrail tests passed (enforced)\\n");
#else
    if (!test_valid_mixed_batch_is_silent_and_complete())
    {
        std::fprintf(stderr, "[input-guardrail-tests-release] FAILED\\n");
        return 1;
    }
    std::fprintf(stderr, "[input-guardrail-tests-release] all input tests passed\\n");
#endif
    return 0;
}