#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: human_controller.hpp
    МОДУЛЬ: input
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн input модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/

#include "shs/input/input_state.hpp"
#include "shs/input/value_actions.hpp"

namespace shs
{
    inline void emit_human_runtime_actions(
        const InputState& in,
        std::vector<RuntimeAction>& out,
        float base_speed,
        float boost_multiplier,
        float look_sensitivity
    )
    {
        emit_human_actions(in, out, base_speed, boost_multiplier, look_sensitivity);
    }
}
