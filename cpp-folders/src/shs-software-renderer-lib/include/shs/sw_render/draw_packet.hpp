#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: draw_packet.hpp
    МОДУЛЬ: render
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн render модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <cstdint>

#include <glm/glm.hpp>

namespace shs
{
    struct DrawPacket
    {
        uint32_t mesh = 0;
        uint32_t material = 0;
        glm::mat4 model{1.0f};
        uint32_t sort_key = 0;
    };
}

