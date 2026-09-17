#include <cstdint>
#include <cstdio>
#include <memory_resource>
#include <span>
#include <variant>
#include <vector>

#include "shs/core/contract_guardrails.hpp"
#include "shs/render/frame/frame.gateway.hpp"

// W-D frame-pod guardrail tests (slice 2 of the per-pod traversal). The frame
// pod is the pinned identity transition: the command vocabulary is EMPTY by
// law (§6.1), so no hand-broken-input negative leg is reachable through the
// public seam. The guards live at the layers that can fail:
//   - compile time: the visit's static_assert makes any new command
//     alternative fail to build past this gateway (proven by these twins
//     compiling against the real header), and frame.event.hpp pins the event
//     vocabulary;
//   - run time (enforced twin): the rim POST proves zero-signal-loss
//     (commands_observed == commands.size()) with the violation handler
//     silent on valid batches — the release twin compiles the assume path
//     and proves identical identity behavior (Rule 4.1).

namespace
{
#if defined(SHS_CONTRACTS_ENFORCED)
    int g_calls = 0;

    void counting_handler(shs::core::contract_kind, const char*, const char*, int) noexcept
    {
        ++g_calls;
    }
#endif

    // Silence + completeness proof (both twins): a batch of monostate
    // commands passes the seam fully observed, emits nothing, and (enforced)
    // never reaches the violation handler.
    bool test_identity_batch_is_silent_and_complete()
    {
        shs::frame::FrameParams state{};
        const shs::frame::FrameCommand commands[]
            = {std::monostate{}, std::monostate{}, std::monostate{}};
        const std::span<const shs::frame::FrameCommand> span{commands};
        const shs::frame::FrameContext context{};
        std::pmr::monotonic_buffer_resource arena{1024};
        std::pmr::vector<shs::frame::FrameEvent> events{&arena};

#if defined(SHS_CONTRACTS_ENFORCED)
        const int calls_before = g_calls;
#endif
        const auto step = shs::frame::frame_gateway(state, span, context, events);
        return step.commands_observed == 3 && events.empty()
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
    const bool ok = test_identity_batch_is_silent_and_complete();
    shs::core::set_contract_violation_handler(nullptr); // restore the default
    if (!ok)
    {
        std::fprintf(stderr, "[frame-guardrail-tests] FAILED\\n");
        return 1;
    }
    std::fprintf(stderr, "[frame-guardrail-tests] all frame guardrail tests passed (enforced)\\n");
#else
    if (!test_identity_batch_is_silent_and_complete())
    {
        std::fprintf(stderr, "[frame-guardrail-tests-release] FAILED\\n");
        return 1;
    }
    std::fprintf(stderr, "[frame-guardrail-tests-release] all frame tests passed\\n");
#endif
    return 0;
}