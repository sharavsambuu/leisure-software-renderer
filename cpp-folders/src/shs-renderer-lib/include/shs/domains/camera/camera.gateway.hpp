#pragma once

/*
    SHS RENDERER SAN

    FILE: camera.gateway.hpp
    MODULE: domains/camera
    PURPOSE: CORE 4. GATEWAY — the pinned identity transition, Kleisli house
             shape (Run C, K1.5): (State, span<Commands>, Context, arena)
             -> CameraStep. Camera rigs are caller-owned values transformed by
             pure builders; the pod owns no applied state yet (K1.4 Run B
             verdict: rig state lives in the input aggregate until an
             orchestrator host exists). The command vocabulary is EMPTY by
             law (§6.1) and the static_assert makes the silence provable.
             Pinned by camera_tests (kit: replay + empty log).
*/

#include <cstdint>
#include <memory_resource>
#include <span>
#include <type_traits>
#include <variant>

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

    // Batch outcome summary (house shape per kdba_kleisli_migration_plan.md).
    // An identity pod applies nothing — it counts what passed the seam.
    struct CameraStep
    {
        uint32_t commands_observed = 0;

        bool operator==(const CameraStep&) const = default;
    };

    inline CameraStep camera_gateway(
        CameraState&                   state,
        std::span<const CameraCommand>  commands,
        const CameraContext&      context,
        std::pmr::vector<CameraEvent>& events)
    {
        (void)state;
        (void)context;
        (void)events;
        CameraStep step{};
        for (const CameraCommand& command : commands)
        {
            std::visit([&](const auto& cmd) {
                using T = std::decay_t<decltype(cmd)>;
                static_assert(std::is_same_v<T, std::monostate>,
                    "camera command vocabulary is empty by law (§6.1): "
                    "land a new intent as a named apply_* arrow first");
                (void)cmd;
            }, command);
            step.commands_observed += 1;
        }
        return step;
    }
} // namespace shs::camera
