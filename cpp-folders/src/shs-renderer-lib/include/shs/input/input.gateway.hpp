#pragma once

/*
    SHS RENDERER SAN

    FILE: input.gateway.hpp
    MODULE: domains/input
    PURPOSE: CORE 4. GATEWAY — the Kleisli house shape (Run B, K2.1):
             (State, span<Commands>, Context, arena) -> InputStep. Dispatch is
             std::visit + if constexpr over the closed RuntimeCommand variant
             (the switch monolith is dead: no kind enum, no payload fishing).
             Zero-signal-loss: every intent emits its fact. Camera note
             (K1.4 decision, Run B): the camera rig lives in this pod's
             RuntimeState aggregate until an orchestrator host exists; the
             camera pod remains the contract/builder seam. See
             docs/backlog/kdba_kleisli_migration_plan.md.
*/

#include <cstdint>
#include <memory_resource>
#include <span>
#include <type_traits>
#include <variant>

#include <glm/glm.hpp>

#include "shs/input/input.command.hpp"
#include "shs/input/input.event.hpp"
#include "shs/input/input_state.hpp"

namespace shs::input
{
    struct InputContext
    {
        float dt = 0.0f;
    };

    // Batch outcome summary (house shape per kdba_kleisli_migration_plan.md;
    // the rim is infallible — every intent is valid for this pod).
    struct InputStep
    {
        uint32_t commands_applied = 0;  // commands that mutated pod state

        bool operator==(const InputStep&) const = default;
    };

    namespace detail
    {
        // --- named per-intent arrows (K2.2: transition bodies live here; the
        // public gateway below is only the assembly point) ----------------

        inline void apply_move_local(
            RuntimeState& state,
            const MoveLocalIntent& cmd,
            const InputContext& context,
            std::pmr::vector<InputEvent>& events,
            InputStep& step)
        {
            const glm::vec3 fwd         = state.camera.forward();
            const glm::vec3 right       = state.camera.right();
            const glm::vec3 up          = glm::vec3(0.0f, 1.0f, 0.0f);
            const glm::vec3 world_delta = right * cmd.local_dir.x + up * cmd.local_dir.y + fwd * cmd.local_dir.z;
            const glm::vec3 applied     = world_delta * (cmd.meters_per_sec * context.dt);
            state.camera.pos += applied;
            events.push_back(CameraTranslatedEvent{applied});
            step.commands_applied += 1;
        }

        inline void apply_look(
            RuntimeState& state,
            const LookIntent& cmd,
            std::pmr::vector<InputEvent>& events,
            InputStep& step)
        {
            const float old_pitch = state.camera.pitch;
            state.camera.yaw   += cmd.dx * cmd.sensitivity;
            state.camera.pitch -= cmd.dy * cmd.sensitivity;
            state.camera.pitch  = glm::clamp(
                state.camera.pitch,
                glm::radians(-85.0f),
                glm::radians(85.0f));
            events.push_back(CameraRotatedEvent{
                cmd.dx * cmd.sensitivity,
                state.camera.pitch - old_pitch});
            step.commands_applied += 1;
        }

        inline void apply_toggle_light_shafts(
            RuntimeState& state,
            std::pmr::vector<InputEvent>& events,
            InputStep& step)
        {
            state.enable_light_shafts = !state.enable_light_shafts;
            events.push_back(RuntimeFlagToggledEvent{
                RuntimeFlagId::LightShafts, state.enable_light_shafts});
            step.commands_applied += 1;
        }

        inline void apply_toggle_bot(
            RuntimeState& state,
            std::pmr::vector<InputEvent>& events,
            InputStep& step)
        {
            state.bot_enabled = !state.bot_enabled;
            events.push_back(RuntimeFlagToggledEvent{
                RuntimeFlagId::Bot, state.bot_enabled});
            step.commands_applied += 1;
        }

        inline void apply_quit(
            RuntimeState& state,
            std::pmr::vector<InputEvent>& events,
            InputStep& step)
        {
            state.quit_requested = true;
            events.push_back(QuitRequestedEvent{});
            step.commands_applied += 1;
        }
    } // namespace detail

    // Apply one batch of input commands against the runtime state (assembly
    // point only — transition bodies live in the named per-intent arrows
    // above, Rule 2 as amended). Events land on the caller's arena.
    inline InputStep input_gateway(
        RuntimeState& state,
        std::span<const RuntimeCommand> commands,
        const InputContext& context,
        std::pmr::vector<InputEvent>& events)
    {
        InputStep step{};
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
            }, command);
        }
        return step;
    }
} // namespace shs::input
