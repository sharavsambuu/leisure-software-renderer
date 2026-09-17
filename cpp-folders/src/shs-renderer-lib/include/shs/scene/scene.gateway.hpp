#pragma once

/*
    SHS RENDERER SAN

    FILE: scene.gateway.hpp
    MODULE: domains/scene
    PURPOSE: CORE 4. GATEWAY — the pinned identity transition, Kleisli house
             shape (Run C, K1.5): (State, span<Commands>, Context, arena)
             -> SceneStep. Item projection (to_render_items) is already a
             pure const method; the empty-vocabulary static_assert makes the
             silence provable (§6.1). Pinned by scene_tests.
*/

#include <cstdint>
#include <memory_resource>
#include <span>
#include <type_traits>
#include <variant>

#include "shs/scene/scene.command.hpp"
#include "shs/scene/scene.contract.hpp"
#include "shs/scene/scene.event.hpp"

namespace shs::scene
{
    struct SceneState
    {
        bool operator==(const SceneState&) const = default;
    };

    struct SceneContext
    {
    };

    // Batch outcome summary (house shape per kdba_kleisli_migration_plan.md).
    // An identity pod applies nothing — it counts what passed the seam.
    struct SceneStep
    {
        uint32_t commands_observed = 0;

        bool operator==(const SceneStep&) const = default;
    };

    inline SceneStep scene_gateway(
        SceneState&                   state,
        std::span<const SceneCommand>  commands,
        const SceneContext&      context,
        std::pmr::vector<SceneEvent>& events)
    {
        (void)state;
        (void)context;
        (void)events;
        SceneStep step{};
        for (const SceneCommand& command : commands)
        {
            std::visit([&](const auto& cmd) {
                using T = std::decay_t<decltype(cmd)>;
                static_assert(std::is_same_v<T, std::monostate>,
                    "scene command vocabulary is empty by law (§6.1): "
                    "land a new intent as a named apply_* arrow first");
                (void)cmd;
            }, command);
            step.commands_observed += 1;
        }
        return step;
    }
} // namespace shs::scene
