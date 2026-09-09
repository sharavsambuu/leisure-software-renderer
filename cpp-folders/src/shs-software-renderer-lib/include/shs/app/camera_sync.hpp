#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: camera_sync.hpp
    МОДУЛЬ: app
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн app модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include "shs/camera/camera_rig.hpp"
#include "shs/camera/view_camera.hpp"
#include "shs/scene/scene_types.hpp"

namespace shs
{
    inline void sync_camera_to_scene(CameraRig& rig, Scene& scene, float aspect)
    {
        ViewCamera vc{};
        vc.pos = rig.pos;
        vc.target = rig.pos + rig.forward();
        vc.up = {0.0f, 1.0f, 0.0f};
        vc.fov_y_radians = scene.cam.fov_y_radians;
        vc.znear = scene.cam.znear;
        vc.zfar = scene.cam.zfar;
        vc.viewproj = scene.cam.viewproj;
        vc.update_matrices(aspect);

        scene.cam.pos = vc.pos;
        scene.cam.target = vc.target;
        scene.cam.up = vc.up;
        scene.cam.view = vc.view;
        scene.cam.proj = vc.proj;
        scene.cam.prev_viewproj = vc.prev_viewproj;
        scene.cam.viewproj = vc.viewproj;
    }
}
