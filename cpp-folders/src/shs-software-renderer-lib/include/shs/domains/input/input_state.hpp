#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: input_state.hpp
    МОДУЛЬ: input
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн input модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


namespace shs
{
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

