#pragma once

/*
    SHS RENDERER SAN

    FILE: value_commands.hpp
    MODULE: input
    PURPOSE: Pure command emitters (InputState latch -> RuntimeCommand spans).
    PURPOSE: Pure command emitters (InputState latch -> RuntimeCommand spans).
             Step 4.1 (domain separation): this pod owns TRANSLATION only —
             the state-mutating gateway retired to the explicit app
             orchestrator shs::app::session_orchestrate, which the host
             drives with the batch emitted here.
*/


#include <cmath>
#include <cstdint>
#include <span>
#include <variant>
#include <vector>

#include <array>
#include <cmath>
#include <cstddef>
#include <memory_resource>
#include <span>
#include <vector>

#include <glm/glm.hpp>

#include "shs/input/input.command.hpp"
#include "shs/input/input_state.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace input
    {
    // K5.1 (Run B): the legacy by-value runtime_state_gateway wrapper is
    // retired. One pod, one public Kleisli gateway: shs::input::input_gateway.

    inline void emit_human_commands(
        const InputState& in,
        std::vector<RuntimeCommand>& out,
        float base_speed,
        float boost_multiplier,
        float look_sensitivity)
    {
        const float speed = base_speed * (in.boost ? boost_multiplier : 1.0f);
        if (in.forward) out.push_back(make_move_local_intent(glm::vec3(0.0f, 0.0f, 1.0f), speed));
        if (in.backward) out.push_back(make_move_local_intent(glm::vec3(0.0f, 0.0f, -1.0f), speed));
        if (in.left) out.push_back(make_move_local_intent(glm::vec3(-1.0f, 0.0f, 0.0f), speed));
        if (in.right) out.push_back(make_move_local_intent(glm::vec3(1.0f, 0.0f, 0.0f), speed));
        if (in.ascend) out.push_back(make_move_local_intent(glm::vec3(0.0f, 1.0f, 0.0f), speed));
        if (in.descend) out.push_back(make_move_local_intent(glm::vec3(0.0f, -1.0f, 0.0f), speed));

        if (in.look_active && (in.look_dx != 0.0f || in.look_dy != 0.0f))
        {
            out.push_back(make_look_intent(in.look_dx, in.look_dy, look_sensitivity));
        }

        if (in.toggle_light_shafts) out.push_back(make_toggle_light_shafts_intent());
        if (in.toggle_bot) out.push_back(make_toggle_bot_intent());
        if (in.quit) out.push_back(make_quit_intent());
    }

    inline void emit_orbit_bot_commands(float time_s, std::vector<RuntimeCommand>& out)
    {
        const float sway = std::sin(time_s * 0.5f);
        out.push_back(make_look_intent(0.35f + 0.25f * sway, 0.0f, 0.01f));
        out.push_back(make_move_local_intent(glm::vec3(0.0f, 0.0f, 0.4f + 0.2f * std::sin(time_s * 0.8f)), 2.0f));
    }

    } // inline namespace input
}
