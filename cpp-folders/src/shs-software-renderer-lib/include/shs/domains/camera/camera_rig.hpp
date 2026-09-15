#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: camera_rig.hpp
    МОДУЛЬ: camera
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн camera модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include "shs/camera/camera_math.hpp"

namespace shs
{
    struct CameraRig
    {
        glm::vec3 pos{0.0f, 0.0f, -3.0f};
        float yaw = glm::half_pi<float>();
        float pitch = 0.0f;

        glm::vec3 forward() const
        {
            return forward_from_yaw_pitch(yaw, pitch);
        }

        glm::vec3 right() const
        {
            return right_from_forward(forward());
        }
    };
}

