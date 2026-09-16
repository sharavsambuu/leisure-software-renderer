#pragma once

/*
    SHS RENDERER SAN

    FILE: scene.gateway.hpp
    MODULE: domains/scene
    PURPOSE: CORE 4. GATEWAY — the identity transition (R5b, P4.1 house shape).
             Item projection (to_render_items) is already a pure const method;
             store lifecycles reduce through commands only after the R5b edge
             migration.
*/

#include <memory_resource>
#include <span>

#include "shs/domains/scene/scene.command.hpp"
#include "shs/domains/scene/scene.contract.hpp"
#include "shs/domains/scene/scene.event.hpp"

namespace shs::scene
{
    struct SceneState
    {
        bool operator==(const SceneState&) const = default;
    };

    struct SceneContext
    {
    };

    inline void scene_gateway(
        SceneState&                   state,
        std::span<const SceneCommand>  commands,
        const SceneContext&      context,
        std::pmr::vector<SceneEvent>& events)
    {
        (void)state;
        (void)commands;
        (void)context;
        (void)events;
    }
} // namespace shs::scene
