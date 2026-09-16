#pragma once

/*
    SHS RENDERER SAN

    FILE: input.reducer.hpp
    MODULE: domains/input
    PURPOSE: CORE 4. REDUCER — the canonical input transition (R3, P4.1).
             House signature: (State, span<const Action>, Inputs, arena Events).
             reduce_runtime_state() in value_actions.hpp delegates here, so one
             logic home serves both the legacy and the evented path.
*/

#include <memory_resource>
#include <span>

#include <glm/glm.hpp>

#include "shs/domains/input/input.action.hpp"
#include "shs/domains/input/input.event.hpp"
#include "shs/domains/input/input_state.hpp"

namespace shs::input
{
    struct InputReduceInputs
    {
        float dt = 0.0f;
    };

    inline void reduce_input(
        RuntimeState&                        state,
        std::span<const RuntimeAction>       actions,
        const InputReduceInputs&             inputs,
        std::pmr::vector<InputEvent>&        events)
    {
        for (const RuntimeAction& action : actions)
        {
            switch (action.type)
            {
                case RuntimeActionType::MoveLocal:
                {
                    const MoveLocalAction* mv = std::get_if<MoveLocalAction>(&action.payload);
                    if (!mv) break;

                    const glm::vec3 fwd         = state.camera.forward();
                    const glm::vec3 right       = state.camera.right();
                    const glm::vec3 up          = glm::vec3(0.0f, 1.0f, 0.0f);
                    const glm::vec3 world_delta = right * mv->local_dir.x + up * mv->local_dir.y + fwd * mv->local_dir.z;
                    const glm::vec3 applied     = world_delta * (mv->meters_per_sec * inputs.dt);
                    state.camera.pos += applied;
                    events.push_back(CameraTranslatedEvent{applied});
                    break;
                }
                case RuntimeActionType::Look:
                {
                    const LookAction* look = std::get_if<LookAction>(&action.payload);
                    if (!look) break;

                    const float old_pitch = state.camera.pitch;
                    state.camera.yaw   += look->dx * look->sensitivity;
                    state.camera.pitch -= look->dy * look->sensitivity;
                    state.camera.pitch  = glm::clamp(
                        state.camera.pitch,
                        glm::radians(-85.0f),
                        glm::radians(85.0f));
                    events.push_back(CameraRotatedEvent{
                        look->dx * look->sensitivity,
                        state.camera.pitch - old_pitch});
                    break;
                }
                case RuntimeActionType::ToggleLightShafts:
                {
                    state.enable_light_shafts = !state.enable_light_shafts;
                    events.push_back(RuntimeFlagToggledEvent{
                        RuntimeFlagId::LightShafts, state.enable_light_shafts});
                    break;
                }
                case RuntimeActionType::ToggleBot:
                {
                    state.bot_enabled = !state.bot_enabled;
                    events.push_back(RuntimeFlagToggledEvent{
                        RuntimeFlagId::Bot, state.bot_enabled});
                    break;
                }
                case RuntimeActionType::Quit:
                {
                    state.quit_requested = true;
                    events.push_back(QuitRequestedEvent{});
                    break;
                }
            }
        }
    }
} // namespace shs::input
