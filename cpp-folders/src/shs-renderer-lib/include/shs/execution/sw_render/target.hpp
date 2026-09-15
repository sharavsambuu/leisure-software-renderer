#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: target.hpp
    МОДУЛЬ: render
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн render модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <cstdint>

namespace shs
{
    struct RenderTargetDesc
    {
        int width = 0;
        int height = 0;
        bool has_color = true;
        bool has_depth = false;
        bool has_motion = false;
    };

    struct RenderTargetHandle
    {
        uint32_t id = 0;
        bool valid() const { return id != 0; }
    };
}

