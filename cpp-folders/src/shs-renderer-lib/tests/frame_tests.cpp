#include <cstdio>
#include <memory_resource>
#include <variant>
#include <vector>

#include "shs/render/frame/frame.contract.hpp"
#include "shs/render/frame/frame.gateway.hpp"
#include "shs/core/testing/pod_test_kit.hpp"

// Headless tests for the frame pod (R3 P3.2: identity transition pins).
// Links only shs::renderer-values + glm.
namespace
{
    auto run_gateway = [](shs::FrameParams& s,
                             std::span<const shs::frame::FrameCommand> a,
                             const shs::frame::FrameContext& in,
                             std::pmr::vector<shs::frame::FrameEvent>& e)
    {
        shs::frame::frame_gateway(s, a, in, e);
    };

    // Identity: no commands exist, so any state survives any (empty) log.
    bool test_identity_stable()
    {
        shs::FrameParams s0{};
        s0.w = 640;
        s0.h = 480;
        s0.exposure = 1.25f;
        s0.technique.mode = shs::TechniqueMode::Deferred;
        return shs::pod_test::empty_log_is_stable<shs::FrameParams, shs::frame::FrameCommand,
            shs::frame::FrameContext, shs::frame::FrameEvent>(
            run_gateway, s0, shs::frame::FrameContext{});
    }

    // Replay over the empty vocabulary is trivially deterministic (kit smoke).
    bool test_replay_deterministic()
    {
        const shs::FrameParams s0{};
        const std::vector<shs::frame::FrameCommand> none{};
        return shs::pod_test::replay_is_deterministic<shs::FrameParams, shs::frame::FrameCommand,
            shs::frame::FrameContext, shs::frame::FrameEvent>(
            run_gateway, s0,
            std::span<const shs::frame::FrameCommand>{none.data(), none.size()},
            shs::frame::FrameContext{});
    }

    // Step counts monostate inputs, not mutations or emitted facts.
    bool test_identity_step_summary()
    {
        shs::FrameParams initial{};
        initial.w = 640;
        initial.h = 480;
        initial.exposure = 1.25f;
        const std::vector<shs::frame::FrameCommand> commands(3);
        const shs::frame::FrameContext context{};
        std::pmr::monotonic_buffer_resource arena{4096};
        std::pmr::vector<shs::frame::FrameEvent> events{&arena};
        // Existing caller-owned facts must not be cleared or replaced.
        events.push_back(std::monostate{});
        const auto original_events = events;
        auto state = initial;
        const auto step = shs::frame::frame_gateway(
            state, std::span<const shs::frame::FrameCommand>{commands}, context, events);
        if (step != shs::frame::FrameStep{3}) return false;
        if (state != initial || events != original_events) return false;

        auto replay = initial;
        std::pmr::vector<shs::frame::FrameEvent> replay_events{&arena};
        replay_events.push_back(std::monostate{});
        const auto replay_step = shs::frame::frame_gateway(
            replay, std::span<const shs::frame::FrameCommand>{commands}, context, replay_events);
        if (replay_step != step || replay != state || replay_events != events) return false;

        const auto empty_step = shs::frame::frame_gateway(
            state, std::span<const shs::frame::FrameCommand>{}, context, events);
        return empty_step == shs::frame::FrameStep{0}
            && state == initial && events == original_events;
    }

    // Closed vocabularies stay closed until a real knob lands.
    bool test_vocabularies_closed()
    {
        static_assert(std::variant_size_v<shs::frame::FrameCommand> == 1);
        static_assert(std::variant_size_v<shs::frame::FrameEvent> == 1);
        static_assert(std::holds_alternative<std::monostate>(shs::frame::FrameCommand{}));
        return true;
    }
} // namespace

int main()
{
    bool ok = true;

    ok = test_identity_stable() && ok;
    ok = test_replay_deterministic() && ok;
    ok = test_vocabularies_closed() && ok;
    ok = test_identity_step_summary() && ok;

    if (!ok)
    {
        std::fprintf(stderr, "[frame-tests] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[frame-tests] all tests passed\n");
    return 0;
}
