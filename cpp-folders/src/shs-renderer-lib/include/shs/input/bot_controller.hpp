#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: bot_controller.hpp
    МОДУЛЬ: input
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн input модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <cmath>
#include <glm/glm.hpp>

#include "shs/input/camera_commands.hpp"
#include "shs/input/command_processor.hpp"

namespace shs
{
    inline void emit_orbit_bot_commands(float time_s, CommandProcessor& out)
    {
        const float sway = std::sin(time_s * 0.5f);
        out.emplace<LookCommand>(0.35f + 0.25f * sway, 0.0f, 0.01f);
        out.emplace<MoveCommand>(glm::vec3(0.0f, 0.0f, 0.4f + 0.2f * std::sin(time_s * 0.8f)), 2.0f);
    }
}
