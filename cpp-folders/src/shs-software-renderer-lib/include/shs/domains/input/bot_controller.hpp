#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: bot_controller.hpp
    МОДУЛЬ: input
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн input модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <cmath>

#include "shs/input/value_actions.hpp"

namespace shs
{
    inline void emit_orbit_bot_runtime_actions(float time_s, std::vector<RuntimeAction>& out)
    {
        emit_orbit_bot_actions(time_s, out);
    }
}
