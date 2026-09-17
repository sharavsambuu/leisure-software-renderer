#include <cmath>
#include <cstdio>
#include <memory_resource>
#include <variant>
#include <vector>

#include "shs/app/session_orchestrator.gateway.hpp"
#include "shs/app/session_settings_sync.hpp"
#include "shs/camera/camera_rig.hpp"
#include "shs/input/input.contract.hpp"
#include "shs/renderpath/planning/render_technique_presets.hpp"
#include "shs/scene/scene_types.hpp"
#include "shs/core/testing/pod_test_kit.hpp"

// Step 4.1 regression suite (engine_domain_separation_migration.md).
// The app orchestrator is the single application path for input intents.
// Pins are goldens carried over from the retired in-pod gateway
// (core_tests / input_tests) plus a bit-parity check against the pre-split
// application math preserved inline below. Links only shs::renderer-values
// + glm: no SDL, no Vulkan, no Context.
//
// Step 4.2 additions: SessionState is the ONE authoritative owner of
// session camera settings (rig + projection) and session render settings
// (light-shafts toggle); sync_session_to_scene / apply_session_render_settings
// are the only write paths into the scene camera and per-frame FrameParams.
// New pins: settings-carrying host independence, settings recorded-replay
// determinism (toggles included), session->scene camera sync vs an inline
// ViewCamera reference, recipe-then-session render-settings precedence.
namespace
{
    using shs::app::SessionState;

    auto run_orchestrator = [](shs::app::SessionState& s,
        std::span<const shs::input::RuntimeCommand> a,
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
        shs::camera::CameraRig camera{};
        bool enable_light_shafts = true;
        bool quit_requested = false;
        bool bot_enabled = false;
    };

    ReferenceState reference_apply(
        ReferenceState s,
        std::span<const shs::input::RuntimeCommand> commands,
        float dt,
        std::vector<shs::input::InputEvent>& events)
    {
        for (const shs::input::RuntimeCommand& command : commands)
        {
            std::visit([&](const auto& cmd) {
                using T = std::decay_t<decltype(cmd)>;
                if constexpr (std::is_same_v<T, shs::input::MoveLocalIntent>)
                {
                    const glm::vec3 fwd = s.camera.forward();
                    const glm::vec3 right = s.camera.right();
                    const glm::vec3 world_delta =
                        right * cmd.local_dir.x + glm::vec3(0.0f, 1.0f, 0.0f) * cmd.local_dir.y + fwd * cmd.local_dir.z;
                    s.camera.pos += world_delta * (cmd.meters_per_sec * dt);
                    events.push_back(shs::input::CameraTranslatedEvent{world_delta * (cmd.meters_per_sec * dt)});
                }
                else if constexpr (std::is_same_v<T, shs::input::LookIntent>)
                {
                    const float old_pitch = s.camera.pitch;
                    s.camera.yaw += cmd.dx * cmd.sensitivity;
                    s.camera.pitch -= cmd.dy * cmd.sensitivity;
                    s.camera.pitch = glm::clamp(s.camera.pitch,
                        glm::radians(-85.0f), glm::radians(85.0f));
                    events.push_back(shs::input::CameraRotatedEvent{
                        cmd.dx * cmd.sensitivity, s.camera.pitch - old_pitch});
                }
                else if constexpr (std::is_same_v<T, shs::input::ToggleLightShaftsIntent>)
                {
                    s.enable_light_shafts = !s.enable_light_shafts;
                    events.push_back(shs::input::RuntimeFlagToggledEvent{
                        shs::input::RuntimeFlagId::LightShafts, s.enable_light_shafts});
                }
                else if constexpr (std::is_same_v<T, shs::input::ToggleBotIntent>)
                {
                    s.bot_enabled = !s.bot_enabled;
                    events.push_back(shs::input::RuntimeFlagToggledEvent{
                        shs::input::RuntimeFlagId::Bot, s.bot_enabled});
                }
                else if constexpr (std::is_same_v<T, shs::input::QuitIntent>)
                {
                    s.quit_requested = true;
                    events.push_back(shs::input::QuitRequestedEvent{});
                }
            }, command);
        }
        return s;
    }

    std::vector<shs::input::RuntimeCommand> make_mixed_log()
    {
        std::vector<shs::input::RuntimeCommand> commands{};
        commands.push_back(shs::input::make_move_local_intent(glm::vec3(0.0f, 0.0f, 1.0f), 4.0f));
        commands.push_back(shs::input::make_look_intent(10.0f, -5.0f, 0.01f));
        commands.push_back(shs::input::make_toggle_light_shafts_intent());
        commands.push_back(shs::input::make_toggle_bot_intent());
        commands.push_back(shs::input::make_quit_intent());
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

        const std::vector<shs::input::RuntimeCommand> commands = make_mixed_log();
        std::pmr::monotonic_buffer_resource arena{4096};
        std::pmr::vector<shs::input::InputEvent> events{&arena};

        const shs::input::InputStep step = shs::app::session_orchestrate(s,
            std::span<const shs::input::RuntimeCommand>{commands.data(), commands.size()},
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

        const std::vector<shs::input::RuntimeCommand> commands = make_mixed_log();
        std::pmr::monotonic_buffer_resource arena{4096};
        std::pmr::vector<shs::input::InputEvent> events{&arena};
        shs::app::session_orchestrate(s,
            std::span<const shs::input::RuntimeCommand>{commands.data(), commands.size()},
            shs::input::InputContext{0.25f}, events);

        ReferenceState r{};
        r.camera.pos = glm::vec3(1.5f, -2.0f, 0.0f);
        r.camera.yaw = 0.7f;
        r.camera.pitch = 0.2f;
        std::vector<shs::input::InputEvent> reference_events{};
        const ReferenceState out = reference_apply(r,
            std::span<const shs::input::RuntimeCommand>{commands.data(), commands.size()},
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

        const std::vector<shs::input::RuntimeCommand> commands = make_mixed_log();
        std::pmr::monotonic_buffer_resource arena{4096};
        std::pmr::vector<shs::input::InputEvent> events{&arena};
        shs::app::session_orchestrate(a,
            std::span<const shs::input::RuntimeCommand>{commands.data(), commands.size()},
            shs::input::InputContext{0.5f}, events);

        return b == SessionState{} && a != b
            && a.quit_requested && !b.quit_requested;
    }

    // Recorded-input replay: same log on two fresh hosts -> identical
    // session state AND identical fact log.
    bool test_recorded_replay()
    {
        const std::vector<shs::input::RuntimeCommand> commands = make_mixed_log();
        const std::span<const shs::input::RuntimeCommand> span{
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
        const std::vector<shs::input::RuntimeCommand> commands = make_mixed_log();
        const shs::input::InputContext in{0.5f};
        const bool deterministic = shs::pod_test::replay_is_deterministic<
            SessionState, shs::input::RuntimeCommand, shs::input::InputContext,
            shs::input::InputEvent>(
            run_orchestrator, s0,
            std::span<const shs::input::RuntimeCommand>{commands.data(), commands.size()}, in);
        const bool stable = shs::pod_test::empty_log_is_stable<
            SessionState, shs::input::RuntimeCommand, shs::input::InputContext,
            shs::input::InputEvent>(run_orchestrator, s0, in);
        return deterministic && stable;
    }

    // Explicit orchestration pipeline: latch -> translation -> application.
    bool test_pipeline_composition()
    {
        shs::input::InputState in{};
        in.forward = true;
        in.boost = true;
        in.look_active = true;
        in.look_dx = 10.0f;
        in.look_dy = -5.0f;

        std::vector<shs::input::RuntimeCommand> commands{};
        shs::input::emit_human_commands(in, commands, 4.0f, 2.0f, 0.01f);

        SessionState s{};
        s.camera.pos = glm::vec3(0.0f, 0.0f, 0.0f);
        std::pmr::monotonic_buffer_resource arena{4096};
        std::pmr::vector<shs::input::InputEvent> events{&arena};
        shs::app::session_orchestrate(s,
            std::span<const shs::input::RuntimeCommand>{commands.data(), commands.size()},
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
        const std::vector<shs::input::RuntimeCommand> drive_up{
            shs::input::make_look_intent(0.0f, -100000.0f, 1.0f)};
        std::pmr::monotonic_buffer_resource arena{4096};
        std::pmr::vector<shs::input::InputEvent> events{&arena};
        shs::app::session_orchestrate(up,
            std::span<const shs::input::RuntimeCommand>{drive_up.data(), drive_up.size()},
            shs::input::InputContext{0.016f}, events);
        if (up.camera.pitch != glm::radians(85.0f)) return false;

        SessionState down{};
        const std::vector<shs::input::RuntimeCommand> drive_down{
            shs::input::make_look_intent(0.0f, 100000.0f, 1.0f)};
        std::pmr::vector<shs::input::InputEvent> events2{&arena};
        shs::app::session_orchestrate(down,
            std::span<const shs::input::RuntimeCommand>{drive_down.data(), drive_down.size()},
            shs::input::InputContext{0.016f}, events2);
        return down.camera.pitch == glm::radians(-85.0f);
    }

    // ---- Step 4.2: one authoritative owner for camera/render settings ----

    // Host independence with settings: two hosts carrying different camera
    // projection settings and render toggles evolve independently under the
    // same recorded log; projection settings are untouched by intent
    // application (the orchestrator owns pose, not projection).
    bool test_settings_host_independence()
    {
        const std::vector<shs::input::RuntimeCommand> commands = make_mixed_log();
        const std::span<const shs::input::RuntimeCommand> span{
            commands.data(), commands.size()};

        SessionState a{};
        a.fov_y_radians = glm::radians(90.0f);
        a.znear = 0.1f;
        a.zfar = 500.0f;
        a.enable_light_shafts = false;

        SessionState b{}; // all defaults

        std::pmr::monotonic_buffer_resource arena{4096};
        std::pmr::vector<shs::input::InputEvent> events{&arena};
        shs::app::session_orchestrate(a, span, shs::input::InputContext{0.5f}, events);
        std::pmr::vector<shs::input::InputEvent> events_b{&arena};
        shs::app::session_orchestrate(b, span, shs::input::InputContext{0.5f}, events_b);

        // Projection settings never move through intent application.
        if (a.fov_y_radians != glm::radians(90.0f)) return false;
        if (a.znear != 0.1f || a.zfar != 500.0f) return false;
        if (b.fov_y_radians != glm::radians(60.0f)) return false;
        if (b.znear != 0.1f || b.zfar != 200.0f) return false;
        // Toggles evolve per host: the mixed log flips light shafts once,
        // so each host ends in the opposite of its start value.
        if (a.enable_light_shafts != true) return false;
        if (b.enable_light_shafts != false) return false;
        // Pose evolves independently (same log, aligned start pose).
        a.camera.yaw = 1.0f;
        b.camera.yaw = 1.0f;
        a.camera.pos.y = 0.0f;
        b.camera.pos.y = 0.0f;
        return a.camera == b.camera && a.quit_requested && b.quit_requested;
    }

    // Recorded replay with toggles: a toggle-heavy log replayed on three
    // fresh hosts yields bit-identical session state (settings included)
    // and identical fact logs.
    bool test_settings_recorded_replay()
    {
        std::vector<shs::input::RuntimeCommand> commands{};
        commands.push_back(shs::input::make_toggle_light_shafts_intent());
        commands.push_back(shs::input::make_toggle_light_shafts_intent());
        commands.push_back(shs::input::make_toggle_bot_intent());
        commands.push_back(shs::input::make_move_local_intent(glm::vec3(0.0f, 0.0f, 1.0f), 2.0f));
        commands.push_back(shs::input::make_toggle_light_shafts_intent());
        const std::span<const shs::input::RuntimeCommand> span{
            commands.data(), commands.size()};

        SessionState a{};
        SessionState b{};
        SessionState c{};
        std::pmr::monotonic_buffer_resource arena{4096};
        std::pmr::vector<shs::input::InputEvent> events_a{&arena};
        std::pmr::vector<shs::input::InputEvent> events_b{&arena};
        std::pmr::vector<shs::input::InputEvent> events_c{&arena};
        shs::app::session_orchestrate(a, span, shs::input::InputContext{0.25f}, events_a);
        shs::app::session_orchestrate(b, span, shs::input::InputContext{0.25f}, events_b);
        shs::app::session_orchestrate(c, span, shs::input::InputContext{0.25f}, events_c);

        // Odd toggle count: light shafts must end OFF on every host.
        return a == b && b == c && events_a == events_b && events_b == events_c
            && !a.enable_light_shafts;
    }

    // Session -> scene camera sync: canonical funnel projects rig pose AND
    // session projection settings; matrices bit-match an independent
    // ViewCamera reference built from the session settings.
    bool test_session_scene_camera_sync()
    {
        shs::app::SessionState s{};
        s.camera.pos = glm::vec3(1.5f, 2.0f, -3.0f);
        s.camera.yaw = 0.8f;
        s.camera.pitch = -0.15f;
        s.fov_y_radians = glm::radians(75.0f);
        s.znear = 0.25f;
        s.zfar = 500.0f;

        shs::scene::Scene scene{};
        scene.cam.viewproj = glm::mat4{1.0f};

        shs::app::sync_session_to_scene(s, scene, 1.7777f);

        // Session is authoritative: scene projection settings are OVERWRITTEN
        // from the session, not preserved.
        if (scene.cam.fov_y_radians != s.fov_y_radians) return false;
        if (scene.cam.znear != s.znear || scene.cam.zfar != s.zfar) return false;

        // Independent reference: same deterministic math path.
        shs::camera::ViewCamera vc{};
        vc.pos = s.camera.pos;
        vc.target = s.camera.pos + s.camera.forward();
        vc.up = {0.0f, 1.0f, 0.0f};
        vc.fov_y_radians = s.fov_y_radians;
        vc.znear = s.znear;
        vc.zfar = s.zfar;
        vc.viewproj = glm::mat4{1.0f};
        vc.update_matrices(1.7777f);

        return scene.cam.pos == vc.pos && scene.cam.target == vc.target
            && scene.cam.up == vc.up && scene.cam.view == vc.view
            && scene.cam.proj == vc.proj && scene.cam.viewproj == vc.viewproj
            && scene.cam.prev_viewproj == vc.prev_viewproj;
    }

    // Session -> FrameParams render settings: the canonical funnel writes
    // both the legacy flat toggle and the pass-block field the light-shafts
    // pass consumes; applied AFTER the technique recipe, the session owner
    // wins; unrelated FrameParams fields are untouched.
    bool test_session_render_settings_sync()
    {
        shs::app::SessionState s{};
        s.enable_light_shafts = true; // session default

        shs::render::FrameParams fp{};
        const float exposure_before = fp.exposure;
        const int steps_before = fp.pass.light_shafts.steps;

        // Planning-level recipe defaults light shafts OFF; apply it first.
        const shs::renderpath::RenderTechniqueRecipe recipe =
            shs::renderpath::make_builtin_render_technique_recipe(shs::RenderTechniquePreset::PBR);
        shs::renderpath::apply_render_technique_recipe_to_frame_params(recipe, fp);
        if (fp.pass.light_shafts.enable) return false; // recipe default applied

        // Session funnel: runtime owner wins over the recipe default.
        shs::app::apply_session_render_settings(s, fp);
        if (!fp.pass.light_shafts.enable) return false;
        if (!fp.enable_light_shafts) return false;

        // Toggle in the session (as the orchestrator would), re-project.
        s.enable_light_shafts = false;
        shs::app::apply_session_render_settings(s, fp);
        if (fp.pass.light_shafts.enable) return false;
        if (fp.enable_light_shafts) return false;

        // Funnel touches ONLY the two toggle fields.
        if (fp.exposure != exposure_before) return false;
        if (fp.pass.light_shafts.steps != steps_before) return false;
        if (fp.pass.tonemap.exposure != recipe.tonemap_exposure) return false;
        return true;
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
    ok = test_settings_host_independence() && ok;
    ok = test_settings_recorded_replay() && ok;
    ok = test_session_scene_camera_sync() && ok;
    ok = test_session_render_settings_sync() && ok;

    if (!ok)
    {
        std::fprintf(stderr, "[session-orchestration-tests] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[session-orchestration-tests] all tests passed\n");
    return 0;
}