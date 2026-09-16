#pragma once

/*
    SHS RENDERER SAN

    FILE: camera.reducer.hpp
    MODULE: domains/camera
    PURPOSE: CORE 4. REDUCER — the identity transition (R5a, P4.1 house shape).
             Camera rigs are caller-owned values transformed by pure builders;
             the pod owns no reduced state yet (a rig-selection state, if any,
             lands with the first consumer per the rule of three).
*/

#include <memory_resource>
#include <span>

#include "shs/domains/camera/camera.action.hpp"
#include "shs/domains/camera/camera.contract.hpp"
#include "shs/domains/camera/camera.event.hpp"

namespace shs::camera
{
    struct CameraState
    {
        bool operator==(const CameraState&) const = default;
    };

    struct CameraReduceInputs
    {
    };

    inline void reduce_camera(
        CameraState&                   state,
        std::span<const CameraAction>  actions,
        const CameraReduceInputs&      inputs,
        std::pmr::vector<CameraEvent>& events)
    {
        (void)state;
        (void)actions;
        (void)inputs;
        (void)events;
    }
} // namespace shs::camera
