#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: view_camera.hpp
    МОДУЛЬ: camera
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн camera модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <glm/glm.hpp>

#include "shs/camera/convention.hpp"

namespace shs
{
    struct ViewCamera
    {
        glm::vec3 pos{0.0f, 0.0f, -3.0f};
        glm::vec3 target{0.0f, 0.0f, 0.0f};
        glm::vec3 up{0.0f, 1.0f, 0.0f};

        float fov_y_radians = glm::radians(60.0f);
        float znear = 0.1f;
        float zfar = 200.0f;

        glm::mat4 view{1.0f};
        glm::mat4 proj{1.0f};
        glm::mat4 viewproj{1.0f};
        glm::mat4 prev_viewproj{1.0f};

        void update_matrices(float aspect)
        {
            view = look_at_lh(pos, target, up);
            proj = perspective_lh_no(fov_y_radians, aspect, znear, zfar);
            prev_viewproj = viewproj;
            viewproj = proj * view;
        }
    };
}
