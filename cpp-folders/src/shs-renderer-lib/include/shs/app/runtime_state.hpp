#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: runtime_state.hpp
    МОДУЛЬ: app
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн app модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


// Edge-zone alias: the session aggregate (camera rig + render/session
// settings) is app-owned since step 4.1; the old root symbol stays valid.
#include "shs/app/session_orchestrator.gateway.hpp"

namespace shs
{
    using shs::RuntimeState;
}
