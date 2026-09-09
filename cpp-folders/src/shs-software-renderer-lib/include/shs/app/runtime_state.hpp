#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: runtime_state.hpp
    МОДУЛЬ: app
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн app модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include "shs/camera/camera_rig.hpp"

namespace shs
{
    struct RuntimeState
    {
        CameraRig camera{};
        bool enable_light_shafts = true;
        bool quit_requested = false;
        bool bot_enabled = false;
    };
}
