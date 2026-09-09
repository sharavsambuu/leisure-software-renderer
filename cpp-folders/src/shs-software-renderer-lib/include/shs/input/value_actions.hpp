#pragma once

/*
    SHS RENDERER SAN

    FILE: value_actions.hpp
    MODULE: input
    PURPOSE: Value-oriented input actions and reducers for runtime state updates.
*/


#include <cmath>
#include <cstdint>
#include <span>
#include <variant>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include "shs/app/runtime_state.hpp"
#include "shs/input/input_state.hpp"

namespace shs
{
    struct MoveLocalAction
    {
        glm::vec3 local_dir{0.0f};
        float meters_per_sec = 0.0f;
    };

    struct LookAction
    {
        float dx = 0.0f;
        float dy = 0.0f;
        float sensitivity = 0.0f;
    };

    struct ToggleFlagAction
    {
        bool value = false;
    };

    enum class RuntimeActionType : uint8_t
    {
        MoveLocal = 0,
        Look = 1,
        ToggleLightShafts = 2,
        ToggleBot = 3,
        Quit = 4
    };

    using RuntimeActionPayload = std::variant<std::monostate, MoveLocalAction, LookAction, ToggleFlagAction>;

    struct RuntimeAction
    {
        RuntimeActionType type = RuntimeActionType::MoveLocal;
        RuntimeActionPayload payload{};
    };

    inline RuntimeAction make_move_local_action(glm::vec3 local_dir, float meters_per_sec)
    {
        RuntimeAction out{};
        out.type = RuntimeActionType::MoveLocal;
        out.payload = MoveLocalAction{local_dir, meters_per_sec};
        return out;
    }

    inline RuntimeAction make_look_action(float dx, float dy, float sensitivity)
    {
        RuntimeAction out{};
        out.type = RuntimeActionType::Look;
        out.payload = LookAction{dx, dy, sensitivity};
        return out;
    }

    inline RuntimeAction make_toggle_light_shafts_action()
    {
        RuntimeAction out{};
        out.type = RuntimeActionType::ToggleLightShafts;
        out.payload = ToggleFlagAction{};
        return out;
    }

    inline RuntimeAction make_toggle_bot_action()
    {
        RuntimeAction out{};
        out.type = RuntimeActionType::ToggleBot;
        out.payload = ToggleFlagAction{};
        return out;
    }

    inline RuntimeAction make_quit_action()
    {
        RuntimeAction out{};
        out.type = RuntimeActionType::Quit;
        out.payload = ToggleFlagAction{};
        return out;
    }

    inline RuntimeState reduce_runtime_state(
        RuntimeState state,
        std::span<const RuntimeAction> actions,
        float dt)
    {
        for (const RuntimeAction& action : actions)
        {
            switch (action.type)
            {
                case RuntimeActionType::MoveLocal:
                {
                    const MoveLocalAction* mv = std::get_if<MoveLocalAction>(&action.payload);
                    if (!mv) break;

                    const glm::vec3 fwd = state.camera.forward();
                    const glm::vec3 right = state.camera.right();
                    const glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
                    const glm::vec3 world_delta = right * mv->local_dir.x + up * mv->local_dir.y + fwd * mv->local_dir.z;
                    state.camera.pos += world_delta * (mv->meters_per_sec * dt);
                    break;
                }
                case RuntimeActionType::Look:
                {
                    const LookAction* look = std::get_if<LookAction>(&action.payload);
                    if (!look) break;

                    state.camera.yaw += look->dx * look->sensitivity;
                    state.camera.pitch -= look->dy * look->sensitivity;
                    state.camera.pitch = glm::clamp(
                        state.camera.pitch,
                        glm::radians(-85.0f),
                        glm::radians(85.0f));
                    break;
                }
                case RuntimeActionType::ToggleLightShafts:
                {
                    state.enable_light_shafts = !state.enable_light_shafts;
                    break;
                }
                case RuntimeActionType::ToggleBot:
                {
                    state.bot_enabled = !state.bot_enabled;
                    break;
                }
                case RuntimeActionType::Quit:
                {
                    state.quit_requested = true;
                    break;
                }
            }
        }
        return state;
    }

    inline void emit_human_actions(
        const InputState& in,
        std::vector<RuntimeAction>& out,
        float base_speed,
        float boost_multiplier,
        float look_sensitivity)
    {
        const float speed = base_speed * (in.boost ? boost_multiplier : 1.0f);
        if (in.forward) out.push_back(make_move_local_action(glm::vec3(0.0f, 0.0f, 1.0f), speed));
        if (in.backward) out.push_back(make_move_local_action(glm::vec3(0.0f, 0.0f, -1.0f), speed));
        if (in.left) out.push_back(make_move_local_action(glm::vec3(-1.0f, 0.0f, 0.0f), speed));
        if (in.right) out.push_back(make_move_local_action(glm::vec3(1.0f, 0.0f, 0.0f), speed));
        if (in.ascend) out.push_back(make_move_local_action(glm::vec3(0.0f, 1.0f, 0.0f), speed));
        if (in.descend) out.push_back(make_move_local_action(glm::vec3(0.0f, -1.0f, 0.0f), speed));

        if (in.look_active && (in.look_dx != 0.0f || in.look_dy != 0.0f))
        {
            out.push_back(make_look_action(in.look_dx, in.look_dy, look_sensitivity));
        }

        if (in.toggle_light_shafts) out.push_back(make_toggle_light_shafts_action());
        if (in.toggle_bot) out.push_back(make_toggle_bot_action());
        if (in.quit) out.push_back(make_quit_action());
    }

    inline void emit_orbit_bot_actions(float time_s, std::vector<RuntimeAction>& out)
    {
        const float sway = std::sin(time_s * 0.5f);
        out.push_back(make_look_action(0.35f + 0.25f * sway, 0.0f, 0.01f));
        out.push_back(make_move_local_action(glm::vec3(0.0f, 0.0f, 0.4f + 0.2f * std::sin(time_s * 0.8f)), 2.0f));
    }
}
