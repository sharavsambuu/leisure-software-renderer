#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: platform_input.hpp
    МОДУЛЬ: platform
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн platform модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


namespace shs
{
    struct PlatformInputState
    {
        bool quit = false;
        bool toggle_light_shafts = false;
        bool toggle_bot = false;
        bool cycle_debug_view = false;
        bool cycle_cull_mode = false;
        bool toggle_front_face = false;
        bool toggle_shading_model = false;
        bool toggle_sky_mode = false;
        bool toggle_follow_camera = false;

        bool forward = false;
        bool backward = false;
        bool left = false;
        bool right = false;
        bool ascend = false;
        bool descend = false;
        bool boost = false;

        bool right_mouse_down = false;
        bool right_mouse_up = false;
        bool left_mouse_down = false;
        bool left_mouse_up = false;
        float mouse_dx = 0.0f;
        float mouse_dy = 0.0f;
    };
}
