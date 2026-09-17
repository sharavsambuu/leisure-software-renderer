#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: bot_controller.hpp
    МОДУЛЬ: input
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн input модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <cmath>

#include "shs/input/value_commands.hpp"

namespace shs
{
    inline void emit_orbit_bot_runtime_commands(float time_s, std::vector<RuntimeCommand>& out)
    {
        emit_orbit_bot_commands(time_s, out);
    }
}
