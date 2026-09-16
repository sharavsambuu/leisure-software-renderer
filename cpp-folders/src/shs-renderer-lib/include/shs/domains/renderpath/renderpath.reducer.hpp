#pragma once

/*
    SHS RENDERER SAN

    FILE: renderpath.reducer.hpp
    MODULE: domains/renderpath
    PURPOSE: PURE TRANSITION for the renderpath pod.
             State_{t+1}, Events = f(State_t, Commands, Compiler, Caps).
             Wraps RenderPathCompiler value-fully: the compiler is an input,
             never pod state. Reducer invariant: an invalid compile keeps the
             previous plan and emits PATH_SWAP_REJECTED. Events append to a
             caller-provided pmr vector (frame arena). No platform headers.
*/

#include <cstdint>
#include <expected>
#include <memory_resource>
#include <span>
#include <string>
#include <type_traits>
#include <variant>

#include "shs/domains/renderpath/renderpath.action.hpp"
#include "shs/domains/renderpath/renderpath.contract.hpp"
#include "shs/domains/renderpath/renderpath.event.hpp"

namespace shs::renderpath
{
    // Pod state: current recipe + the plan it compiled to.
    struct RenderPathPodState
    {
        RenderPathRecipe recipe{};
        RenderPathExecutionPlan plan{};
        bool has_plan = false;
    };

    // --- technique mapping -------------------------------------------------

    inline TechniqueMode technique_mode_for(RenderPathRenderingTechnique technique)
    {
        switch (technique)
        {
            case RenderPathRenderingTechnique::ForwardLit: return TechniqueMode::Forward;
            case RenderPathRenderingTechnique::ForwardPlus: return TechniqueMode::ForwardPlus;
            case RenderPathRenderingTechnique::Deferred: return TechniqueMode::Deferred;
        }
        return TechniqueMode::Forward;
    }

    // --- rejection mapping -------------------------------------------------
    // Compiler-native enum -> pod event enum, 1:1 (R4 P4.6). The string ladder
    // this replaces (classify_plan_rejection over plan.errors text) is deleted;
    // error strings remain as diagnostics, never as decision inputs.
    inline PathSwapRejectionReason map_rejection(RenderPathCompileRejection reason)
    {
        switch (reason)
        {
            case RenderPathCompileRejection::EmptyPassChain:     return PathSwapRejectionReason::EmptyPassChain;
            case RenderPathCompileRejection::BackendUnavailable: return PathSwapRejectionReason::BackendUnavailable;
            case RenderPathCompileRejection::MissingRequiredPass: return PathSwapRejectionReason::MissingRequiredPass;
            case RenderPathCompileRejection::DepthUnsupported:   return PathSwapRejectionReason::DepthUnsupported;
            case RenderPathCompileRejection::OcclusionUnsupported: return PathSwapRejectionReason::OcclusionUnsupported;
            case RenderPathCompileRejection::CompileInvalid:     return PathSwapRejectionReason::CompileInvalid;
        }
        return PathSwapRejectionReason::CompileInvalid;
    }

    // --- runtime toggles ---------------------------------------------------

    inline bool apply_runtime_toggle(RenderPathRuntimeState& state, RuntimeToggle toggle, bool enabled)
    {
        switch (toggle)
        {
            case RuntimeToggle::ViewOcclusion: state.view_occlusion_enabled = enabled; return true;
            case RuntimeToggle::ShadowOcclusion: state.shadow_occlusion_enabled = enabled; return true;
            case RuntimeToggle::DebugAabb: state.debug_aabb = enabled; return true;
            case RuntimeToggle::LitMode: state.lit_mode = enabled; return true;
            case RuntimeToggle::Shadows: state.enable_shadows = enabled; return true;
        }
        return false;
    }

    namespace detail
    {
        // Honest fallible channel (VOP spec §8, R4 P4.6): the compiler returns
        // expected natively; the pod maps the native reason to its event enum.
        // No string classification exists anywhere on this path anymore.
        inline std::expected<RenderPathExecutionPlan, PathSwapRejectionReason>
        compile_render_path_plan(
            const RenderPathRecipe& candidate,
            const RenderPathCompiler& compiler,
            const RenderPathCapabilitySet& caps)
        {
            return compiler.try_compile(candidate, caps)
                .transform_error([](RenderPathCompileRejection reason)
                {
                    return map_rejection(reason);
                });
        }

        // Compile a candidate recipe against capabilities; on success the pod
        // state adopts recipe + plan, on failure the previous plan stays
        // untouched and PATH_SWAP_REJECTED is emitted (reducer invariant).
        // Monadic chain per VOP spec §8: the events side-effects stay in the
        // transform/or_else continuations; the value/error channel carries
        // only the plan or its closed-enum rejection reason.
        inline void try_swap_plan(
            RenderPathPodState& state,
            const RenderPathRecipe& candidate,
            const RenderPathCompiler& compiler,
            const RenderPathCapabilitySet& caps,
            std::pmr::vector<RenderPathEvent>& events)
        {
            compile_render_path_plan(candidate, compiler, caps)
                .transform([&](RenderPathExecutionPlan&& candidate_plan) {
                    state.recipe = candidate;
                    state.plan = std::move(candidate_plan);
                    state.has_plan = true;
                    events.push_back(PathCompiledEvent{
                        state.plan.technique_mode,
                        state.plan.render_technique,
                        static_cast<uint32_t>(state.plan.pass_chain.size())
                    });
                })
                .or_else([&](const PathSwapRejectionReason reason)
                             -> std::expected<void, PathSwapRejectionReason> {
                    events.push_back(PathSwapRejectedEvent{ reason });
                    return {};
                });
        }
    } // namespace detail

    // Reduce a frame's renderpath commands against the current pod state.
    // compiler + caps are pure value inputs; events land on the caller's
    // frame arena via the pmr event vector.
    inline void reduce_render_path(
        RenderPathPodState& state,
        std::span<const RenderPathCommand> commands,
        const RenderPathCompiler& compiler,
        const RenderPathCapabilitySet& caps,
        std::pmr::vector<RenderPathEvent>& events)
    {
        for (const RenderPathCommand& command : commands)
        {
            std::visit([&](const auto& cmd) {
                using T = std::decay_t<decltype(cmd)>;

                if constexpr (std::is_same_v<T, SelectPathPresetIntent>)
                {
                    detail::try_swap_plan(state, cmd.recipe, compiler, caps, events);
                }
                else if constexpr (std::is_same_v<T, SetRenderingTechniqueIntent>)
                {
                    if (cmd.technique == state.recipe.render_technique) return;
                    const RenderPathRenderingTechnique previous = state.recipe.render_technique;
                    RenderPathRecipe candidate = state.recipe;
                    candidate.render_technique = cmd.technique;
                    candidate.technique_mode = technique_mode_for(cmd.technique);
                    detail::try_swap_plan(state, candidate, compiler, caps, events);
                    events.push_back(TechniqueSwitchedEvent{ previous, cmd.technique });
                }
                else if constexpr (std::is_same_v<T, SetViewCullingModeIntent>)
                {
                    const RenderPathCullingMode previous = state.recipe.view_culling;
                    if (cmd.mode == previous) return;
                    RenderPathRecipe candidate = state.recipe;
                    candidate.view_culling = cmd.mode;
                    detail::try_swap_plan(state, candidate, compiler, caps, events);
                    events.push_back(CullingModeChangedEvent{ true, previous, cmd.mode });
                }
                else if constexpr (std::is_same_v<T, SetShadowCullingModeIntent>)
                {
                    const RenderPathCullingMode previous = state.recipe.shadow_culling;
                    if (cmd.mode == previous) return;
                    RenderPathRecipe candidate = state.recipe;
                    candidate.shadow_culling = cmd.mode;
                    detail::try_swap_plan(state, candidate, compiler, caps, events);
                    events.push_back(CullingModeChangedEvent{ false, previous, cmd.mode });
                }
                else if constexpr (std::is_same_v<T, SetRuntimeToggleIntent>)
                {
                    apply_runtime_toggle(state.recipe.runtime_defaults, cmd.toggle, cmd.enabled);
                    if (state.has_plan)
                    {
                        apply_runtime_toggle(state.plan.runtime_state, cmd.toggle, cmd.enabled);
                    }
                    events.push_back(RuntimeToggledEvent{ cmd.toggle, cmd.enabled });
                }
            }, command);
        }
    }
} // namespace shs::renderpath
