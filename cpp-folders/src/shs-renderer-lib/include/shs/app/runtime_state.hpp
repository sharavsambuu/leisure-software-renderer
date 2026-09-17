#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: runtime_state.hpp
    МОДУЛЬ: app
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн app модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


// Edge-zone re-export: RuntimeState is input-domain state (P2.3).
#include "shs/input/input_state.hpp"

namespace shs
{
    using shs::RuntimeState;
}
