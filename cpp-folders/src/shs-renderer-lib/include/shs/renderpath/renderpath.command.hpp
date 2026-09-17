#pragma once

/*
    SHS RENDERER SAN

    FILE: renderpath.command.hpp
    MODULE: domains/renderpath
    PURPOSE: INTENT TOKENS for the renderpath pod — closed RenderPathCommand
             variant (Constitution §6.1). Pure value payloads; no platform
             types, no std::string.
*/

#include <cstdint>
#include <variant>

#include "shs/renderpath/renderpath.contract.hpp"

namespace shs::renderpath
{
    // Install a whole recipe (path preset) and compile it against capabilities.
    struct SelectPathPresetIntent
    {
        RenderPathRecipe recipe{};
    };

    // Hot-swap the rendering technique (ForwardLit / ForwardPlus / Deferred).
    struct SetRenderingTechniqueIntent
    {
        RenderPathRenderingTechnique technique = RenderPathRenderingTechnique::ForwardLit;
    };

    // Hot-swap view-chain culling mode.
    struct SetViewCullingModeIntent
    {
        RenderPathCullingMode mode = RenderPathCullingMode::Frustum;
    };

    // Hot-swap shadow-chain culling mode.
    struct SetShadowCullingModeIntent
    {
        RenderPathCullingMode mode = RenderPathCullingMode::FrustumAndOptionalOcclusion;
    };

    // Closed key set for runtime toggles carried by RenderPathRuntimeState.
    enum class RuntimeToggle : uint8_t
    {
        ViewOcclusion = 0,
        ShadowOcclusion = 1,
        DebugAabb = 2,
        LitMode = 3,
        Shadows = 4
    };

    struct SetRuntimeToggleIntent
    {
        RuntimeToggle toggle = RuntimeToggle::ViewOcclusion;
        bool enabled = false;
    };

    // Closed command vocabulary for the renderpath pod.
    using RenderPathCommand = std::variant<
        SelectPathPresetIntent,
        SetRenderingTechniqueIntent,
        SetViewCullingModeIntent,
        SetShadowCullingModeIntent,
        SetRuntimeToggleIntent
    >;
} // namespace shs::renderpath
