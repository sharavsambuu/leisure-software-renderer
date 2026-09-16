#include <cstdio>
#include <memory_resource>
#include <variant>
#include <vector>

#include "shs/domains/frame/frame.contract.hpp"
#include "shs/domains/frame/frame.gateway.hpp"
#include "shs/domains/pod_test_kit.hpp"

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

    if (!ok)
    {
        std::fprintf(stderr, "[frame-tests] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[frame-tests] all tests passed\n");
    return 0;
}
