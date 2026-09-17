#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: input_state.hpp
    МОДУЛЬ: input
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн input модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include "shs/camera/camera_rig.hpp"

namespace shs
{
    /** @brief Input-domain runtime state (moved from execution/app P2.3). */
    struct RuntimeState
    {
        CameraRig camera{};
        bool enable_light_shafts = true;
        bool quit_requested = false;
        bool bot_enabled = false;

        bool operator==(const RuntimeState&) const = default;
    };

    struct InputState
    {
        bool forward = false;
        bool backward = false;
        bool left = false;
        bool right = false;
        bool ascend = false;
        bool descend = false;
        bool boost = false;

        bool look_active = false;
        float look_dx = 0.0f;
        float look_dy = 0.0f;

        bool toggle_light_shafts = false;
        bool toggle_bot = false;
        bool quit = false;
    };
}

