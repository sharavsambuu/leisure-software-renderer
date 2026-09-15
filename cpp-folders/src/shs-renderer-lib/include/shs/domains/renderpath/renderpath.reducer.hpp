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

    // --- rejection classification -----------------------------------------

    inline PathSwapRejectionReason classify_plan_rejection(const RenderPathExecutionPlan& plan)
    {
        for (const auto& error : plan.errors)
        {
            if (error.find("pass chain is empty") != std::string::npos ||
                error.find("no executable passes") != std::string::npos)
            {
                return PathSwapRejectionReason::EmptyPassChain;
            }
            if (error.find("not registered in context") != std::string::npos)
            {
                return PathSwapRejectionReason::BackendUnavailable;
            }
            if (error.find("no 'shadow_map' pass") != std::string::npos ||
                error.find("no 'depth_prepass' pass") != std::string::npos)
            {
                return PathSwapRejectionReason::MissingRequiredPass;
            }
            if (error.find("no depth attachment support") != std::string::npos)
            {
                return PathSwapRejectionReason::DepthUnsupported;
            }
            if (error.find("requires occlusion culling, but backend does not support") != std::string::npos)
            {
                return PathSwapRejectionReason::OcclusionUnsupported;
            }
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
        // First std::expected adoption slot (VOP spec §8): compile → classify
        // as an honest expected value instead of a (plan, valid, errors[])
        // pod followed by a stringly-typed classification ladder. The closed
        // enum error payload keeps the channel constitution-compatible.
        inline std::expected<RenderPathExecutionPlan, PathSwapRejectionReason>
        compile_render_path_plan(
            const RenderPathRecipe& candidate,
            const RenderPathCompiler& compiler,
            const RenderPathCapabilitySet& caps)
        {
            RenderPathExecutionPlan plan = compiler.compile(candidate, caps);
            if (!plan.valid)
            {
                return std::unexpected(classify_plan_rejection(plan));
            }
            return plan;
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
