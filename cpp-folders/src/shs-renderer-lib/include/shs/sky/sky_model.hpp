#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: sky_model.hpp
    МОДУЛЬ: sky
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн sky модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <glm/glm.hpp>

namespace shs
{
    class ISkyModel
    {
    public:
        virtual ~ISkyModel() = default;
        virtual glm::vec3 sample(const glm::vec3& direction_ws) const = 0; // linear color
    };
}

