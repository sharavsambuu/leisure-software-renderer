#pragma once

/*
    SHS RENDERER SAN

    FILE: session_orchestrator.hpp
    MODULE: app
    PURPOSE: EXPLICIT APP ORCHESTRATION (step 4.1,
             docs/outdated/engine_domain_separation_migration.md):
             the orchestration host. Input intents are APPLIED here, to
             state owned HERE. Input translation (latch -> intents,
             value_commands) stays in the input pod; camera/render/session
             application leaves it. The camera rig moves out of the input
             pod's state aggregate, resolving the K1.4 interim note
             (docs/outdated/kdba_kleisli_migration_plan.md: "until an
             orchestrator host exists").

             Zero-signal-loss and the Kleisli shape are preserved verbatim
             from the retired shs::input::input_gateway: (State,
             span<Commands>, Context, arena) -> Step, std::visit + if
             constexpr over the closed RuntimeCommand variant, one fact per
             applied intent. Application math (local->world basis, dt
             scaling, +-85 deg pitch clamp, toggle semantics) is
             byte-identical; regression suites (input_tests,
             session_orchestration_tests, core_tests) pin it.
*/

#include <cstdint>
#include <memory_resource>
#include <span>
#include <type_traits>
#include <variant>

#include <glm/glm.hpp>

#include "shs/core/contract_guardrails.hpp"
#include "shs/core/step_shape.hpp"
#include "shs/camera/camera_rig.hpp"
#include "shs/input/input.command.hpp"
#include "shs/input/input.event.hpp"
#include "shs/input/input.gateway.hpp"

namespace shs::app
{
    // Session-level runtime state: camera rig + render/session settings.
    // One authoritative owner: app (step 4.1). Fields and defaults are
    // verbatim from the retired shs::input::RuntimeState so old behavior
    // is bit-preserved.
    //
    // Step 4.2 (engine_domain_separation_migration.md): this aggregate is
    // also the ONE authoritative owner of session-scoped camera settings
    // (projection: fov/znear/zfar, defaults identical to shs::Camera and
    // shs::ViewCamera) and session render settings (light-shafts toggle).
    // Scene shs::Camera and FrameParams are per-frame renderer PROJECTIONS
    // of this state, written only through the canonical sync funnels in
    // shs/app/session_settings_sync.hpp. Hosts must not edit the
    // projections directly; toggles applied here win over technique-level
    // recipe defaults.
    struct SessionState
    {
        CameraRig camera{};
        float fov_y_radians = glm::radians(60.0f);
        float znear = 0.1f;
        float zfar  = 200.0f;
        bool enable_light_shafts = true;
        bool quit_requested = false;
        bool bot_enabled = false;

        bool operator==(const SessionState&) const = default;
    };

    namespace detail
    {
        // --- named per-intent application arrows (moved verbatim from the
        // retired input gateway; transition bodies live here, the public
        // orchestrator below is only the assembly point) -----------------

        inline void apply_move_local(
            SessionState& state,
            const MoveLocalIntent& cmd,
            const shs::input::InputContext& context,
            std::pmr::vector<shs::input::InputEvent>& events,
            shs::input::InputStep& step)
        {
            const glm::vec3 fwd         = state.camera.forward();
            const glm::vec3 right       = state.camera.right();
            const glm::vec3 up          = glm::vec3(0.0f, 1.0f, 0.0f);
            const glm::vec3 world_delta = right * cmd.local_dir.x + up * cmd.local_dir.y + fwd * cmd.local_dir.z;
            const glm::vec3 applied     = world_delta * (cmd.meters_per_sec * context.dt);
            state.camera.pos += applied;
            events.push_back(shs::input::CameraTranslatedEvent{applied});
            step.commands_applied += 1;
        }

        inline void apply_look(
            SessionState& state,
            const LookIntent& cmd,
            std::pmr::vector<shs::input::InputEvent>& events,
            shs::input::InputStep& step)
        {
            const float old_pitch = state.camera.pitch;
            state.camera.yaw   += cmd.dx * cmd.sensitivity;
            state.camera.pitch -= cmd.dy * cmd.sensitivity;
            state.camera.pitch  = glm::clamp(
                state.camera.pitch,
                glm::radians(-85.0f),
                glm::radians(85.0f));
            events.push_back(shs::input::CameraRotatedEvent{
                cmd.dx * cmd.sensitivity,
                state.camera.pitch - old_pitch});
            step.commands_applied += 1;
        }
        inline void apply_toggle_light_shafts(
            SessionState& state,
            std::pmr::vector<shs::input::InputEvent>& events,
            shs::input::InputStep& step)
        {
            state.enable_light_shafts = !state.enable_light_shafts;
            events.push_back(shs::input::RuntimeFlagToggledEvent{
                shs::input::RuntimeFlagId::LightShafts, state.enable_light_shafts});
            step.commands_applied += 1;
        }

        inline void apply_toggle_bot(
            SessionState& state,
            std::pmr::vector<shs::input::InputEvent>& events,
            shs::input::InputStep& step)
        {
            state.bot_enabled = !state.bot_enabled;
            events.push_back(shs::input::RuntimeFlagToggledEvent{
                shs::input::RuntimeFlagId::Bot, state.bot_enabled});
            step.commands_applied += 1;
        }

        inline void apply_quit(
            SessionState& state,
            std::pmr::vector<shs::input::InputEvent>& events,
            shs::input::InputStep& step)
        {
            state.quit_requested = true;
            events.push_back(shs::input::QuitRequestedEvent{});
            step.commands_applied += 1;
        }
    } // namespace detail

    // Apply one batch of input intents against the session-owned state
    // (assembly point only — application bodies live in the named
    // per-intent arrows above, Rule 2 as amended). Events land on the
    // caller's arena. This is the single canonical application path for
    // input intents (step 4.1).
    // R3 (ROP-3.2): the orchestrator rim reuses shs::input::InputStep —
    // pinned here so the app rim stays Step-shaped if the shared step forks.
    static_assert(shs::core::StepShape<shs::input::InputStep>);
    inline shs::input::InputStep session_orchestrate(
        SessionState& state,
        std::span<const RuntimeCommand> commands,
        const shs::input::InputContext& context,
        std::pmr::vector<shs::input::InputEvent>& events)
    {
        shs::input::InputStep step{};
        for (const RuntimeCommand& command : commands)
        {
            std::visit([&](const auto& cmd) {
                using T = std::decay_t<decltype(cmd)>;

                if constexpr (std::is_same_v<T, MoveLocalIntent>)
                {
                    detail::apply_move_local(state, cmd, context, events, step);
                }
                else if constexpr (std::is_same_v<T, LookIntent>)
                {
                    detail::apply_look(state, cmd, events, step);
                }
                else if constexpr (std::is_same_v<T, ToggleLightShaftsIntent>)
                {
                    detail::apply_toggle_light_shafts(state, events, step);
                }
                else if constexpr (std::is_same_v<T, ToggleBotIntent>)
                {
                    detail::apply_toggle_bot(state, events, step);
                }
                else if constexpr (std::is_same_v<T, QuitIntent>)
                {
                    detail::apply_quit(state, events, step);
                }
                else
                {
                    // P5 exhaustiveness (W-D input slice, 2026-09-17): the
                    // command variant is closed — an unhandled alternative
                    // must fail to compile here, never silently swallow
                    // (no default:). The gate 9 scan covers *.gateway.hpp
                    // only; this orchestrator rim is annotated by the sweep.
                    static_assert(sizeof(T) == 0,
                        "unhandled RuntimeCommand alternative in session_orchestrate dispatch");
                }
            }, command);
        }
        // Rim postcondition (W-D input slice, 2026-09-17): the rim is
        // infallible by law — every intent in the closed vocabulary is valid
        // for this pod (input.gateway.hpp: "the rim is infallible") — so
        // zero-signal-loss (K3.2) means every consumed command was applied
        // and emitted exactly one fact.
        SHS_POST(step.commands_applied == commands.size());
        return step;
    }
} // namespace shs::app

namespace shs
{
// namespace-cutover: app compat wrapper (step 7; shs::app pre-exists, cannot be inline)
    namespace app
    {
    // Root compatibility alias (step 4.1): the session aggregate moved from
    // the input pod to app ownership; the old root symbol stays valid.
    using RuntimeState = shs::app::SessionState;

    } // namespace app

    // namespace-cutover compatibility (step 7): root spelling of app symbol
    using app::RuntimeState;
} // namespace shs
