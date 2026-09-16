#pragma once

/*
    SHS RENDERER SAN

    FILE: camera.gateway.hpp
    MODULE: domains/camera
    PURPOSE: CORE 4. GATEWAY — the identity transition (R5a, P4.1 house shape).
             Camera rigs are caller-owned values transformed by pure builders;
             the pod owns no applied state yet (a rig-selection state, if any,
             lands with the first consumer per the rule of three).
*/

#include <memory_resource>
#include <span>

#include "shs/domains/camera/camera.command.hpp"
#include "shs/domains/camera/camera.contract.hpp"
#include "shs/domains/camera/camera.event.hpp"

namespace shs::camera
{
    struct CameraState
    {
        bool operator==(const CameraState&) const = default;
    };

    struct CameraContext
    {
    };

    inline void camera_gateway(
        CameraState&                   state,
        std::span<const CameraCommand>  commands,
        const CameraContext&      context,
        std::pmr::vector<CameraEvent>& events)
    {
        (void)state;
        (void)commands;
        (void)context;
        (void)events;
    }
} // namespace shs::camera
