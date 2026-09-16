#pragma once

/*
    SHS RENDERER SAN

    FILE: camera.contract.hpp
    MODULE: domains/camera
    PURPOSE: CORE 1. TYPES — the camera pod's discoverability seam (R5a P3.3).
             Rigs, view/light cameras and their pure builders. All LH (+Z
             forward) per Constitution I; math lives in camera_math.hpp and is
             included for discoverability (function usings arrive with P5).
*/

#include "shs/domains/camera/camera_math.hpp"
#include "shs/domains/camera/camera_rig.hpp"
#include "shs/domains/camera/convention.hpp"
#include "shs/domains/camera/follow_camera.hpp"
#include "shs/domains/camera/free_camera.hpp"
#include "shs/domains/camera/light_camera.hpp"
#include "shs/domains/camera/view_camera.hpp"

namespace shs::camera
{
    using shs::CameraRig;
    using shs::FreeCamera;
    using shs::FreeCameraInput;
    using shs::ViewCamera;
    using shs::LightCamera;
    using shs::follow_target;
    using shs::build_dir_light_camera_aabb;
} // namespace shs::camera
