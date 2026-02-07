#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: convention.hpp
    МОДУЛЬ: camera
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн camera модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace shs
{
    // Left-handed look-at matrix with NDC z in [-1, 1].
    inline glm::mat4 look_at_lh(const glm::vec3& eye, const glm::vec3& target, const glm::vec3& up)
    {
        return glm::lookAtLH(eye, target, up);
    }

    // Left-handed perspective projection, NDC z in [-1, 1].
    inline glm::mat4 perspective_lh_no(float fovy_radians, float aspect, float znear, float zfar)
    {
        return glm::perspectiveLH_NO(fovy_radians, aspect, znear, zfar);
    }

    // Left-handed orthographic projection, NDC z in [-1, 1].
    inline glm::mat4 ortho_lh_no(float left, float right, float bottom, float top, float znear, float zfar)
    {
        return glm::orthoLH_NO(left, right, bottom, top, znear, zfar);
    }
}
