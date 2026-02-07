#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: human_controller.hpp
    МОДУЛЬ: input
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн input модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include "shs/input/camera_commands.hpp"
#include "shs/input/command_processor.hpp"
#include "shs/input/input_state.hpp"

namespace shs
{
    inline void emit_human_commands(
        const InputState& in,
        CommandProcessor& out,
        float base_speed,
        float boost_multiplier,
        float look_sensitivity
    )
    {
        const float speed = base_speed * (in.boost ? boost_multiplier : 1.0f);

        if (in.forward) out.emplace<MoveCommand>(glm::vec3(0.0f, 0.0f, 1.0f), speed);
        if (in.backward) out.emplace<MoveCommand>(glm::vec3(0.0f, 0.0f, -1.0f), speed);
        if (in.left) out.emplace<MoveCommand>(glm::vec3(-1.0f, 0.0f, 0.0f), speed);
        if (in.right) out.emplace<MoveCommand>(glm::vec3(1.0f, 0.0f, 0.0f), speed);
        if (in.ascend) out.emplace<MoveCommand>(glm::vec3(0.0f, 1.0f, 0.0f), speed);
        if (in.descend) out.emplace<MoveCommand>(glm::vec3(0.0f, -1.0f, 0.0f), speed);

        if (in.look_active && (in.look_dx != 0.0f || in.look_dy != 0.0f))
        {
            out.emplace<LookCommand>(in.look_dx, in.look_dy, look_sensitivity);
        }

        if (in.toggle_light_shafts) out.emplace<ToggleLightShaftsCommand>();
        if (in.toggle_bot) out.emplace<ToggleBotCommand>();
        if (in.quit) out.emplace<QuitCommand>();
    }
}

