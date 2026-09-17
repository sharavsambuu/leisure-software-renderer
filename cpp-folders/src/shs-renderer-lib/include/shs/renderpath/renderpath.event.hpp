#pragma once

/*
    SHS RENDERER SAN

    FILE: renderpath.event.hpp
    MODULE: domains/renderpath
    PURPOSE: RAW FACTS emitted by the renderpath gateway — closed
             RenderPathEvent variant (Constitution §6.1). Plain values only
             (no std::string payloads); recipe identity stays in pod state,
             events carry counts/enums so they can live on the frame arena.
             Zero-signal-loss (K3.2 house answer, Run A 2026-09-17): every
             consumed command emits at least one fact — accepted transitions
             emit their change fact, same-value commands emit *Unchanged
             facts, rejected swaps emit PATH_SWAP_REJECTED. Silence is never
             a valid gateway outcome.
*/

#include <compare>
#include <cstdint>
#include <variant>

#include "shs/render/frame/technique_mode.hpp"
#include "shs/renderpath/renderpath.command.hpp"
#include "shs/renderpath/renderpath.contract.hpp"

namespace shs::renderpath
{
    // Why a candidate recipe failed to compile. Produced natively by the
    // compiler (R4 P4.6); plan.errors strings are diagnostics only.
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

        bool operator==(const PathCompiledEvent&) const = default;
    };

    // PATH_SWAP_REJECTED: compile failed; the pod kept the previous plan.
    struct PathSwapRejectedEvent
    {
        PathSwapRejectionReason reason = PathSwapRejectionReason::CompileInvalid;

        bool operator==(const PathSwapRejectedEvent&) const = default;
    };

    struct TechniqueSwitchedEvent
    {
        RenderPathRenderingTechnique previous = RenderPathRenderingTechnique::ForwardLit;
        RenderPathRenderingTechnique current = RenderPathRenderingTechnique::ForwardLit;

        bool operator==(const TechniqueSwitchedEvent&) const = default;
    };

    // K4.2 (Run A): the positional-bool CullingModeChangedEvent{view_chain, ...}
    // is split into two named facts — one event, one fact.
    struct ViewCullingModeChangedEvent
    {
        RenderPathCullingMode previous = RenderPathCullingMode::Frustum;
        RenderPathCullingMode current = RenderPathCullingMode::Frustum;

        bool operator==(const ViewCullingModeChangedEvent&) const = default;
    };

    struct ShadowCullingModeChangedEvent
    {
        RenderPathCullingMode previous = RenderPathCullingMode::FrustumAndOptionalOcclusion;
        RenderPathCullingMode current = RenderPathCullingMode::FrustumAndOptionalOcclusion;

        bool operator==(const ShadowCullingModeChangedEvent&) const = default;
    };

    struct RuntimeToggledEvent
    {
        RuntimeToggle toggle = RuntimeToggle::ViewOcclusion;
        bool enabled = false;

        bool operator==(const RuntimeToggledEvent&) const = default;
    };

    // K3.2 unchanged facts: a consumed command whose value is already active
    // mutates nothing — it still emits its fact so the event log tells the
    // whole story (no silent no-op sites in the gateway).
    struct TechniqueUnchangedEvent
    {
        RenderPathRenderingTechnique current = RenderPathRenderingTechnique::ForwardLit;

        bool operator==(const TechniqueUnchangedEvent&) const = default;
    };

    struct ViewCullingUnchangedEvent
    {
        RenderPathCullingMode current = RenderPathCullingMode::Frustum;

        bool operator==(const ViewCullingUnchangedEvent&) const = default;
    };

    struct ShadowCullingUnchangedEvent
    {
        RenderPathCullingMode current = RenderPathCullingMode::FrustumAndOptionalOcclusion;

        bool operator==(const ShadowCullingUnchangedEvent&) const = default;
    };

    // Closed event vocabulary for the renderpath pod.
    using RenderPathEvent = std::variant<
        PathCompiledEvent,
        PathSwapRejectedEvent,
        TechniqueSwitchedEvent,
        ViewCullingModeChangedEvent,
        ShadowCullingModeChangedEvent,
        RuntimeToggledEvent,
        TechniqueUnchangedEvent,
        ViewCullingUnchangedEvent,
        ShadowCullingUnchangedEvent
    >;

    inline const char* renderpath_event_name(const RenderPathEvent& ev)
    {
        if (std::holds_alternative<PathCompiledEvent>(ev))            return "path_compiled";
        if (std::holds_alternative<PathSwapRejectedEvent>(ev))        return "path_swap_rejected";
        if (std::holds_alternative<TechniqueSwitchedEvent>(ev))       return "technique_switched";
        if (std::holds_alternative<ViewCullingModeChangedEvent>(ev))  return "view_culling_mode_changed";
        if (std::holds_alternative<ShadowCullingModeChangedEvent>(ev)) return "shadow_culling_mode_changed";
        if (std::holds_alternative<RuntimeToggledEvent>(ev))          return "runtime_toggled";
        if (std::holds_alternative<TechniqueUnchangedEvent>(ev))      return "technique_unchanged";
        if (std::holds_alternative<ViewCullingUnchangedEvent>(ev))    return "view_culling_unchanged";
        return "shadow_culling_unchanged";
    }

    static_assert(std::variant_size_v<RenderPathEvent> == 9,
        "renderpath event vocabulary changed: update name table + EVENT_FLOW.md");
} // namespace shs::renderpath
