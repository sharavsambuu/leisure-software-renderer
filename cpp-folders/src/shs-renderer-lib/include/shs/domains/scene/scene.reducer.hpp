#pragma once

/*
    SHS RENDERER SAN

    FILE: scene.reducer.hpp
    MODULE: domains/scene
    PURPOSE: CORE 4. REDUCER — the identity transition (R5b, P4.1 house shape).
             Item projection (to_render_items) is already a pure const method;
             store lifecycles reduce through commands only after the R5b edge
             migration.
*/

#include <memory_resource>
#include <span>

#include "shs/domains/scene/scene.action.hpp"
#include "shs/domains/scene/scene.contract.hpp"
#include "shs/domains/scene/scene.event.hpp"

namespace shs::scene
{
    struct SceneState
    {
        bool operator==(const SceneState&) const = default;
    };

    struct SceneReduceInputs
    {
    };

    inline void reduce_scene(
        SceneState&                   state,
        std::span<const SceneAction>  actions,
        const SceneReduceInputs&      inputs,
        std::pmr::vector<SceneEvent>& events)
    {
        (void)state;
        (void)actions;
        (void)inputs;
        (void)events;
    }
} // namespace shs::scene
