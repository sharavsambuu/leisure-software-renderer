#include <cmath>
#include <cstdio>
#include <memory_resource>
#include <variant>
#include <vector>

#include "shs/app/session_orchestrator.hpp"
#include "shs/camera/camera_rig.hpp"
#include "shs/input/input.contract.hpp"
#include "shs/core/testing/pod_test_kit.hpp"

// Step 4.1 regression suite (engine_domain_separation_migration.md).
// The app orchestrator is the single application path for input intents.
// Pins are goldens carried over from the retired in-pod gateway
// (core_tests / input_tests) plus a bit-parity check against the pre-split
// application math preserved inline below. Links only shs::renderer-values
// + glm: no SDL, no Vulkan, no Context.
namespace
{
    using shs::app::SessionState;

    auto run_orchestrator = [](shs::app::SessionState& s,
        std::span<const shs::RuntimeCommand> a,
        const shs::input::InputContext& in,
        std::pmr::vector<shs::input::InputEvent>& e)
    {
        return shs::app::session_orchestrate(s, a, in, e);
    };

    bool approx_eq(float a, float b)
    {
        return std::fabs(a - b) < 1.0e-5f;
    }

    // Pre-split reference math: verbatim copy of the retired
    // shs::input::input_gateway arrows (the behavior contract step 4.1
    // must preserve bit-for-bit).
    struct ReferenceState
    {
        shs::CameraRig camera{};
        bool enable_light_shafts = true;
        bool quit_requested = false;
        bool bot_enabled = false;
    };

    ReferenceState reference_apply(
        ReferenceState s,
        std::span<const shs::RuntimeCommand> commands,
        float dt,
        std::vector<shs::input::InputEvent>& events)
    {
        for (const shs::RuntimeCommand& command : commands)
        {
            std::visit([&](const auto& cmd) {
                using T = std::decay_t<decltype(cmd)>;
                if constexpr (std::is_same_v<T, shs::MoveLocalIntent>)
                {
                    const glm::vec3 fwd = s.camera.forward();
                    const glm::vec3 right = s.camera.right();
                    const glm::vec3 world_delta =
                        right * cmd.local_dir.x + glm::vec3(0.0f, 1.0f, 0.0f) * cmd.local_dir.y + fwd * cmd.local_dir.z;
                    s.camera.pos += world_delta * (cmd.meters_per_sec * dt);
                    events.push_back(shs::input::CameraTranslatedEvent{world_delta * (cmd.meters_per_sec * dt)});
                }
                else if constexpr (std::is_same_v<T, shs::LookIntent>)
                {
                    const float old_pitch = s.camera.pitch;
                    s.camera.yaw += cmd.dx * cmd.sensitivity;
                    s.camera.pitch -= cmd.dy * cmd.sensitivity;
                    s.camera.pitch = glm::clamp(s.camera.pitch,
                        glm::radians(-85.0f), glm::radians(85.0f));
                    events.push_back(shs::input::CameraRotatedEvent{
                        cmd.dx * cmd.sensitivity, s.camera.pitch - old_pitch});
                }
                else if constexpr (std::is_same_v<T, shs::ToggleLightShaftsIntent>)
                {
                    s.enable_light_shafts = !s.enable_light_shafts;
                    events.push_back(shs::input::RuntimeFlagToggledEvent{
                        shs::input::RuntimeFlagId::LightShafts, s.enable_light_shafts});
                }
                else if constexpr (std::is_same_v<T, shs::ToggleBotIntent>)
                {
                    s.bot_enabled = !s.bot_enabled;
                    events.push_back(shs::input::RuntimeFlagToggledEvent{
                        shs::input::RuntimeFlagId::Bot, s.bot_enabled});
                }
                else if constexpr (std::is_same_v<T, shs::QuitIntent>)
                {
                    s.quit_requested = true;
                    events.push_back(shs::input::QuitRequestedEvent{});
                }
            }, command);
        }
        return s;
    }

    std::vector<shs::RuntimeCommand> make_mixed_log()
    {
        std::vector<shs::RuntimeCommand> commands{};
        commands.push_back(shs::make_move_local_intent(glm::vec3(0.0f, 0.0f, 1.0f), 4.0f));
        commands.push_back(shs::make_look_intent(10.0f, -5.0f, 0.01f));
        commands.push_back(shs::make_toggle_light_shafts_intent());
        commands.push_back(shs::make_toggle_bot_intent());
        commands.push_back(shs::make_quit_intent());
        return commands;
    }

    // Golden mixed-log pins (carried from the retired in-pod gateway:
    // core_tests::test_runtime_command_gateway values).
    bool test_golden_mixed_log()
    {
        SessionState s{};
        s.camera.pos = glm::vec3(0.0f, 0.0f, 0.0f);
        s.camera.yaw = glm::half_pi<float>();
        s.camera.pitch = 0.0f;
        s.enable_light_shafts = true;
        s.bot_enabled = false;
        s.quit_requested = false;

        const std::vector<shs::RuntimeCommand> commands = make_mixed_log();
        std::pmr::monotonic_buffer_resource arena{4096};
        std::pmr::vector<shs::input::InputEvent> events{&arena};

        const shs::input::InputStep step = shs::app::session_orchestrate(s,
            std::span<const shs::RuntimeCommand>{commands.data(), commands.size()},
            shs::input::InputContext{0.5f}, events);

        if (!approx_eq(s.camera.pos.z, 2.0f)) return false;
        if (!approx_eq(s.camera.yaw, glm::half_pi<float>() + 0.1f)) return false;
        if (!approx_eq(s.camera.pitch, 0.05f)) return false;
        if (s.enable_light_shafts) return false;
        if (!s.bot_enabled) return false;
        if (!s.quit_requested) return false;
        if (step.commands_applied != 5) return false;
        if (events.size() != 5) return false;
        return true;
    }

    // Bit-parity: orchestrator outcome == pre-split application math.
    bool test_reference_parity()
    {
        SessionState s{};
        s.camera.pos = glm::vec3(1.5f, -2.0f, 0.0f);
        s.camera.yaw = 0.7f;
        s.camera.pitch = 0.2f;

        const std::vector<shs::RuntimeCommand> commands = make_mixed_log();
        std::pmr::monotonic_buffer_resource arena{4096};
        std::pmr::vector<shs::input::InputEvent> events{&arena};
        shs::app::session_orchestrate(s,
            std::span<const shs::RuntimeCommand>{commands.data(), commands.size()},
            shs::input::InputContext{0.25f}, events);

        ReferenceState r{};
        r.camera.pos = glm::vec3(1.5f, -2.0f, 0.0f);
        r.camera.yaw = 0.7f;
        r.camera.pitch = 0.2f;
        std::vector<shs::input::InputEvent> reference_events{};
        const ReferenceState out = reference_apply(r,
            std::span<const shs::RuntimeCommand>{commands.data(), commands.size()},
            0.25f, reference_events);

        if (!(s.camera == out.camera)) return false;
        if (s.enable_light_shafts != out.enable_light_shafts) return false;
        if (s.quit_requested != out.quit_requested) return false;
        if (s.bot_enabled != out.bot_enabled) return false;
        if (events.size() != reference_events.size()) return false;
        for (size_t i = 0; i < events.size(); ++i)
        {
            if (!(events[i] == reference_events[i])) return false;
        }
        return true;
    }

    // Independent host instances share nothing (step 4.2 seed).
    bool test_host_independence()
    {
        SessionState a{};
        SessionState b{};
        a.camera.yaw = 1.0f;

        const std::vector<shs::RuntimeCommand> commands = make_mixed_log();
        std::pmr::monotonic_buffer_resource arena{4096};
        std::pmr::vector<shs::input::InputEvent> events{&arena};
        shs::app::session_orchestrate(a,
            std::span<const shs::RuntimeCommand>{commands.data(), commands.size()},
            shs::input::InputContext{0.5f}, events);

        return b == SessionState{} && a != b
            && a.quit_requested && !b.quit_requested;
    }

    // Recorded-input replay: same log on two fresh hosts -> identical
    // session state AND identical fact log.
    bool test_recorded_replay()
    {
        const std::vector<shs::RuntimeCommand> commands = make_mixed_log();
        const std::span<const shs::RuntimeCommand> span{
            commands.data(), commands.size()};

        SessionState a{};
        SessionState b{};
        std::pmr::monotonic_buffer_resource arena{4096};
        std::pmr::vector<shs::input::InputEvent> events_a{&arena};
        std::pmr::vector<shs::input::InputEvent> events_b{&arena};
        shs::app::session_orchestrate(a, span, shs::input::InputContext{0.5f}, events_a);
        shs::app::session_orchestrate(b, span, shs::input::InputContext{0.5f}, events_b);

        return a == b && events_a == events_b;
    }

    // Kit parity: replay determinism + empty-batch stability, pod kit forms.
    bool test_kit_determinism()
    {
        const SessionState s0{};
        const std::vector<shs::RuntimeCommand> commands = make_mixed_log();
        const shs::input::InputContext in{0.5f};
        const bool deterministic = shs::pod_test::replay_is_deterministic<
            SessionState, shs::RuntimeCommand, shs::input::InputContext,
            shs::input::InputEvent>(
            run_orchestrator, s0,
            std::span<const shs::RuntimeCommand>{commands.data(), commands.size()}, in);
        const bool stable = shs::pod_test::empty_log_is_stable<
            SessionState, shs::RuntimeCommand, shs::input::InputContext,
            shs::input::InputEvent>(run_orchestrator, s0, in);
        return deterministic && stable;
    }

    // Explicit orchestration pipeline: latch -> translation -> application.
    bool test_pipeline_composition()
    {
        shs::InputState in{};
        in.forward = true;
        in.boost = true;
        in.look_active = true;
        in.look_dx = 10.0f;
        in.look_dy = -5.0f;

        std::vector<shs::RuntimeCommand> commands{};
        shs::emit_human_commands(in, commands, 4.0f, 2.0f, 0.01f);

        SessionState s{};
        s.camera.pos = glm::vec3(0.0f, 0.0f, 0.0f);
        std::pmr::monotonic_buffer_resource arena{4096};
        std::pmr::vector<shs::input::InputEvent> events{&arena};
        shs::app::session_orchestrate(s,
            std::span<const shs::RuntimeCommand>{commands.data(), commands.size()},
            shs::input::InputContext{0.5f}, events);

        // boost multiplies base speed: 4.0 * 2.0 * dt 0.5 = 4.0 along +Z.
        if (!approx_eq(s.camera.pos.z, 4.0f)) return false;
        if (!approx_eq(s.camera.yaw, glm::half_pi<float>() + 0.1f)) return false;
        if (!approx_eq(s.camera.pitch, 0.05f)) return false;
        return true;
    }

    // Look clamp pin: violent pitch input saturates at +-85 degrees.
    bool test_clamp_saturation()
    {
        SessionState up{};
        const std::vector<shs::RuntimeCommand> drive_up{
            shs::make_look_intent(0.0f, -100000.0f, 1.0f)};
        std::pmr::monotonic_buffer_resource arena{4096};
        std::pmr::vector<shs::input::InputEvent> events{&arena};
        shs::app::session_orchestrate(up,
            std::span<const shs::RuntimeCommand>{drive_up.data(), drive_up.size()},
            shs::input::InputContext{0.016f}, events);
        if (up.camera.pitch != glm::radians(85.0f)) return false;

        SessionState down{};
        const std::vector<shs::RuntimeCommand> drive_down{
            shs::make_look_intent(0.0f, 100000.0f, 1.0f)};
        std::pmr::vector<shs::input::InputEvent> events2{&arena};
        shs::app::session_orchestrate(down,
            std::span<const shs::RuntimeCommand>{drive_down.data(), drive_down.size()},
            shs::input::InputContext{0.016f}, events2);
        return down.camera.pitch == glm::radians(-85.0f);
    }
} // namespace

int main()
{
    bool ok = true;

    ok = test_golden_mixed_log() && ok;
    ok = test_reference_parity() && ok;
    ok = test_host_independence() && ok;
    ok = test_recorded_replay() && ok;
    ok = test_kit_determinism() && ok;
    ok = test_pipeline_composition() && ok;
    ok = test_clamp_saturation() && ok;

    if (!ok)
    {
        std::fprintf(stderr, "[session-orchestration-tests] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[session-orchestration-tests] all tests passed\n");
    return 0;
}