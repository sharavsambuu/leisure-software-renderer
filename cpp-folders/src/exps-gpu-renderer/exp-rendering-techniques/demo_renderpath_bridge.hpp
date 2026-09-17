#pragma once

/*
    SHS RENDERER SAN

    FILE: demo_renderpath_bridge.hpp
    SCOPE: exp-rendering-techniques (Run 1 / P3 task 2: path config -> intents)
    PURPOSE: Pure mapping between the demo's closed input tokens and the
             renderpath pod's closed intent vocabulary
             (shs::renderpath::RenderPathCommand).

    Design (Domain Pod / Constitution I S7):
      - No SDL, no Vulkan, no GLM: GPU-free testable. The demo edge resolves
        recipe payloads (executor registry) and pushes intents; the pure
        reducer `reduce_render_path` decides accept/reject; the demo applies
        accepted events to its executor edge.
      - The demo's `TechniqueMode` axis maps 1:1 onto the pod's
        `RenderPathRenderingTechnique` axis.
*/

#include <optional>

#include "demo_input_actions.hpp"
#include "shs/renderpath/renderpath.command.hpp"

namespace shs::demo
{
    // Demo TechniqueMode -> pod rendering technique.
    inline shs::renderpath::RenderPathRenderingTechnique technique_for_mode(shs::TechniqueMode mode)
    {
        switch (mode)
        {
            case shs::TechniqueMode::ForwardPlus:
                return shs::renderpath::RenderPathRenderingTechnique::ForwardPlus;
            case shs::TechniqueMode::Deferred:
                return shs::renderpath::RenderPathRenderingTechnique::Deferred;
            case shs::TechniqueMode::Forward:
            default:
                return shs::renderpath::RenderPathRenderingTechnique::ForwardLit;
        }
    }

    // Pod rendering technique -> demo TechniqueMode.
    inline shs::TechniqueMode mode_for_technique(shs::renderpath::RenderPathRenderingTechnique technique)
    {
        switch (technique)
        {
            case shs::renderpath::RenderPathRenderingTechnique::ForwardPlus:
                return shs::TechniqueMode::ForwardPlus;
            case shs::renderpath::RenderPathRenderingTechnique::Deferred:
                return shs::TechniqueMode::Deferred;
            case shs::renderpath::RenderPathRenderingTechnique::ForwardLit:
            default:
                return shs::TechniqueMode::Forward;
        }
    }

    // Hot-swap cycle order for the technique axis: Forward -> Forward+ -> Deferred -> Forward.
    inline shs::TechniqueMode next_demo_technique_mode(shs::TechniqueMode mode)
    {
        switch (mode)
        {
            case shs::TechniqueMode::Forward: return shs::TechniqueMode::ForwardPlus;
            case shs::TechniqueMode::ForwardPlus: return shs::TechniqueMode::Deferred;
            case shs::TechniqueMode::Deferred:
            default: return shs::TechniqueMode::Forward;
        }
    }

    // Pure token -> intent mapping for the renderpath pod. Path-recipe cycling
    // returns nullopt here: the recipe payload must be resolved from the
    // demo-owned executor registry, which is edge-side (not pure).
    inline std::optional<shs::renderpath::RenderPathCommand> map_action_to_renderpath_command(
        DemoInputAction action,
        shs::TechniqueMode current_mode)
    {
        switch (action)
        {
            case DemoInputAction::CycleRenderingTechnique:
                return shs::renderpath::RenderPathCommand{
                    shs::renderpath::SetRenderingTechniqueIntent{
                        technique_for_mode(next_demo_technique_mode(current_mode))}};
            default:
                return std::nullopt; // not a renderpath intent (tuning/debug/path-recipe)
        }
    }
} // namespace shs::demo
