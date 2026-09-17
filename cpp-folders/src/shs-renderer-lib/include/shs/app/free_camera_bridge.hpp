#pragma once

/*
    SHS RENDERER SAN

    FILE: free_camera_bridge.hpp
    MODULE: execution/platform (edge bridge; R2 P2.1)
    PURPOSE: Maps the platform edge input state into the camera domain's
             plain FreeCameraInput, then drives FreeCamera purely.
             Domains never see PlatformInputState; edges never own camera math.
*/

#include "shs/camera/free_camera.hpp"
#include "shs/platform/platform_input.hpp"

namespace shs
{
// namespace-cutover: app compat wrapper (step 7; shs::app pre-exists, cannot be inline)
    namespace app
    {
    inline FreeCameraInput to_camera_input(const PlatformInputState& in)
    {
        FreeCameraInput out{};
        out.forward  = in.forward;
        out.backward = in.backward;
        out.left     = in.left;
        out.right    = in.right;
        out.ascend   = in.ascend;
        out.descend  = in.descend;
        out.boost    = in.boost;
        return out;
    }

    inline void update_free_camera_from_platform(FreeCamera& cam, const PlatformInputState& in, float dt)
    {
        cam.update(to_camera_input(in), dt);
    }

    } // namespace app

    // namespace-cutover compatibility (step 7): root spellings of app symbols
    using app::to_camera_input;
    using app::update_free_camera_from_platform;
} // namespace shs
