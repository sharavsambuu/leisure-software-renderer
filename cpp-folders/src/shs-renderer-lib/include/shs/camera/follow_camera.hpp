#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: follow_camera.hpp
    МОДУЛЬ: camera
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн camera модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <cmath>
#include <algorithm>
#include <glm/glm.hpp>

#include "shs/camera/camera_rig.hpp"

namespace shs
{
    inline void follow_target(
        CameraRig& rig,
        const glm::vec3& target_pos,
        const glm::vec3& offset_ws,
        float smoothing,
        float dt
    )
    {
        const float k = std::clamp(smoothing, 0.0f, 1.0f);
        const float t = 1.0f - std::pow(1.0f - k, std::max(0.0f, dt) * 60.0f);
        const glm::vec3 desired = target_pos + offset_ws;
        rig.pos = glm::mix(rig.pos, desired, t);
    }
}
