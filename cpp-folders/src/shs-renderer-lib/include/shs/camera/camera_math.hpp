#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: camera_math.hpp
    МОДУЛЬ: camera
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн camera модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

namespace shs
{
    inline glm::vec3 forward_from_yaw_pitch(float yaw, float pitch)
    {
        glm::vec3 f{};
        f.x = std::cos(pitch) * std::cos(yaw);
        f.y = std::sin(pitch);
        f.z = std::cos(pitch) * std::sin(yaw);
        return glm::normalize(f);
    }

    inline glm::vec3 right_from_forward(const glm::vec3& fwd, const glm::vec3& world_up = glm::vec3(0.0f, 1.0f, 0.0f))
    {
        return glm::normalize(glm::cross(fwd, world_up));
    }
}

