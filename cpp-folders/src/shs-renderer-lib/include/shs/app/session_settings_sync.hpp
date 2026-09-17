#pragma once

/*
    SHS RENDERER SAN

    FILE: session_settings_sync.hpp
    MODULE: app
    PURPOSE: Step 4.2 (engine_domain_separation_migration.md): the ONE
             authoritative owner of session-scoped camera settings (rig
             pose + projection fov/znear/zfar) and session render settings
             (light-shafts toggle) is shs::app::SessionState. The scene
             camera and the per-frame FrameParams are renderer PROJECTIONS
             of that state; both are written only through the two canonical
             sync funnels below, so scattered copies cannot drift:

               - sync_session_to_scene: session camera settings ->
               shs::Scene (view/proj matrices via shs::ViewCamera, same
               math as the pre-4.2 shs::sync_camera_to_scene compat path,
               with projection settings sourced from the session instead
               of the scene copy).

               - apply_session_render_settings: session render settings ->
               shs::FrameParams. Apply AFTER
               apply_render_technique_recipe_to_frame_params: technique
               presets are planning-level defaults; the session toggle is
               the runtime owner and wins.

             Both funnels are pure projections of session state onto the
             per-frame render inputs; neither mutates the session.
*/

#include "shs/app/session_orchestrator.gateway.hpp"
#include "shs/camera/view_camera.hpp"
#include "shs/render/frame/frame_params.hpp"
#include "shs/scene/scene_types.hpp"

namespace shs::app
{
    // Canonical camera-settings funnel (step 4.2): rig pose + session
    // projection settings -> scene camera. Field assignment order and
    // matrix math are identical to shs::sync_camera_to_scene; only the
    // fov/znear/zfar source changes (session instead of scene copy).
    inline void sync_session_to_scene(
        const SessionState& session, Scene& scene, float aspect)
    {
        ViewCamera vc{};
        vc.pos = session.camera.pos;
        vc.target = session.camera.pos + session.camera.forward();
        vc.up = {0.0f, 1.0f, 0.0f};
        vc.fov_y_radians = session.fov_y_radians;
        vc.znear = session.znear;
        vc.zfar = session.zfar;
        vc.viewproj = scene.cam.viewproj;
        vc.update_matrices(aspect);

        // The scene camera is a pure projection of the session: settings
        // fields are overwritten from the session, never stale.
        scene.cam.fov_y_radians = session.fov_y_radians;
        scene.cam.znear = session.znear;
        scene.cam.zfar = session.zfar;
        scene.cam.pos = vc.pos;
        scene.cam.target = vc.target;
        scene.cam.up = vc.up;
        scene.cam.view = vc.view;
        scene.cam.proj = vc.proj;
        scene.cam.prev_viewproj = vc.prev_viewproj;
        scene.cam.viewproj = vc.viewproj;
    }

    // Canonical render-settings funnel (step 4.2): session render settings
    // -> per-frame FrameParams. Writes both the legacy flat toggle and the
    // pass-block field the light-shafts pass actually consumes; everything
    // else in FrameParams is untouched.
    inline void apply_session_render_settings(
        const SessionState& session, FrameParams& fp)
    {
        fp.enable_light_shafts = session.enable_light_shafts;
        fp.pass.light_shafts.enable = session.enable_light_shafts;
    }
} // namespace shs::app
