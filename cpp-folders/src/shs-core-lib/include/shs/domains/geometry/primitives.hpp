#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: primitives.hpp
    МОДУЛЬ: geometry
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн geometry модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <glm/glm.hpp>

namespace shs
{
    struct PlaneDesc
    {
        float width = 10.0f;
        float depth = 10.0f;
        int seg_x = 10;
        int seg_z = 10;
    };

    struct SphereDesc
    {
        float radius = 1.0f;
        int seg_u = 32;
        int seg_v = 16;
    };

    struct BoxDesc
    {
        glm::vec3 size{1.0f, 1.0f, 1.0f};
        int seg_x = 1;
        int seg_y = 1;
        int seg_z = 1;
    };

    struct ConeDesc
    {
        float radius = 1.0f;
        float height = 2.0f;
        int seg_radial = 24;
        int seg_height = 1;
        bool cap = true;
    };
}

