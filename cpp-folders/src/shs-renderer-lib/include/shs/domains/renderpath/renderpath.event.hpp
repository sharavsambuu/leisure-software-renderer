#pragma once

/*
    SHS RENDERER SAN

    FILE: renderpath.event.hpp
    MODULE: domains/renderpath
    PURPOSE: RAW FACTS emitted by the renderpath reducer — closed
             RenderPathEvent variant (Constitution §6.1). Plain values only
             (no std::string payloads); recipe identity stays in pod state,
             events carry counts/enums so they can live on the frame arena.
*/

#include <cstdint>
#include <variant>

#include "shs/domains/frame/technique_mode.hpp"
#include "shs/domains/renderpath/renderpath.contract.hpp"

namespace shs::renderpath
{
    // Why a candidate recipe failed to compile. Classified from the compiler
    // plan's error text; the plan itself stays the source of truth.
    enum class PathSwapRejectionReason : uint8_t
    {
        CompileInvalid = 0,
        EmptyPassChain = 1,
        BackendUnavailable = 2,
        MissingRequiredPass = 3,
        DepthUnsupported = 4,
        OcclusionUnsupported = 5
    };

    // PATH_COMPILED: a recipe compiled valid and is now the active plan.
    struct PathCompiledEvent
    {
        TechniqueMode technique_mode = TechniqueMode::Forward;
        RenderPathRenderingTechnique render_technique = RenderPathRenderingTechnique::ForwardLit;
        uint32_t pass_count = 0;
    };

    // PATH_SWAP_REJECTED: compile failed; the pod kept the previous plan.
    struct PathSwapRejectedEvent
    {
        PathSwapRejectionReason reason = PathSwapRejectionReason::CompileInvalid;
    };

    struct TechniqueSwitchedEvent
    {
        RenderPathRenderingTechnique previous = RenderPathRenderingTechnique::ForwardLit;
        RenderPathRenderingTechnique current = RenderPathRenderingTechnique::ForwardLit;
    };

    struct CullingModeChangedEvent
    {
        bool view_chain = true; // false => shadow chain
        RenderPathCullingMode previous = RenderPathCullingMode::Frustum;
        RenderPathCullingMode current = RenderPathCullingMode::Frustum;
    };

    struct RuntimeToggledEvent
    {
        RuntimeToggle toggle = RuntimeToggle::ViewOcclusion;
        bool enabled = false;
    };

    // Closed event vocabulary for the renderpath pod.
    using RenderPathEvent = std::variant<
        PathCompiledEvent,
        PathSwapRejectedEvent,
        TechniqueSwitchedEvent,
        CullingModeChangedEvent,
        RuntimeToggledEvent
    >;
} // namespace shs::renderpath
