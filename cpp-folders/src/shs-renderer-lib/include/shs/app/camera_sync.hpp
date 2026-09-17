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
    // Pre-4.2 compatibility path (step 4.2,
    // engine_domain_separation_migration.md): projection settings are read
    // from the scene copy. The canonical camera-settings funnel is
    // shs::app::sync_session_to_scene (shs/app/session_settings_sync.hpp),
    // which sources pose AND projection settings from the authoritative
    // app-owned SessionState. This overload is kept for existing scene-side
    // consumers; behavior is unchanged.
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
