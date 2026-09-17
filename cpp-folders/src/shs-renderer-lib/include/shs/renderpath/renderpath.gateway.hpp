#pragma once

/*
    SHS RENDERER SAN

    FILE: renderpath.gateway.hpp
    MODULE: domains/renderpath
    PURPOSE: PURE TRANSITION for the renderpath pod — Kleisli house shape
             (Run A, K1.2): (State, span<Commands>, Compiler, Caps, arena)
             -> RenderPathStep. Wraps RenderPathCompiler value-fully: the
             compiler is an input, never pod state. Gateway invariant: an
             invalid compile keeps the previous plan and emits
             PATH_SWAP_REJECTED. Events append to a caller-provided pmr
             vector (frame arena, A.7 divergence — the Writer log is never
             bundled into the value payload). No platform headers.
*/

#include <cstdint>
#include <expected>
#include <memory_resource>
#include <span>
#include <string>
#include <type_traits>
#include <variant>

#include "shs/renderpath/renderpath.command.hpp"
#include "shs/renderpath/renderpath.contract.hpp"
#include "shs/renderpath/renderpath.event.hpp"

namespace shs::renderpath
{
    // Pod state: current recipe + the plan it compiled to.
    // K4.1: plan_generation replaces the phantom validity bit —
    // 0 = no plan ever installed; bumped on every successful plan install.
    // Value-honest (state can no longer lie: a default plan with generation
    // 0 is "no plan", not "plan that lies"), replay-friendly, and feeds
    // P6.1's plan-hash executor rebuild keys.
    struct RenderPathPodState
    {
        RenderPathRecipe recipe{};
        RenderPathExecutionPlan plan{};
        uint32_t plan_generation = 0;

        bool operator==(const RenderPathPodState&) const = default;
    };

    // Batch outcome summary (K1.1 spike decision, recorded in
    // docs/backlog/kdba_kleisli_migration_plan.md): the batch rim is
    // infallible — every real failure (compile rejection) is absorbed by the
    // per-command expected rail inside try_swap_plan and materialized as a
    // PATH_SWAP_REJECTED fact — so the value rail is a plain Step, not an
    // invented batch-level expected (ERROR_FLOW non-vacuity law).
    struct RenderPathStep
    {
        uint32_t commands_applied = 0;  // commands that mutated pod state
        uint32_t noops_observed   = 0;  // same-value commands (unchanged facts)
        uint32_t swaps_rejected   = 0;  // PATH_SWAP_REJECTED facts emitted
        uint32_t plan_generation  = 0;  // resulting pod generation (0 = none)

        bool operator==(const RenderPathStep&) const = default;
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
    // error strings remain as diagnostics, never as decision context.
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
        // state adopts recipe + plan (+ generation bump) and PATH_COMPILED is
        // emitted; on failure the previous plan stays untouched and
        // PATH_SWAP_REJECTED is emitted (gateway invariant).
        // Monadic chain per VOP spec §8: the events side-effects stay in the
        // transform/or_else continuations; the value/error channel carries
        // only the plan or its closed-enum rejection reason. Returns the
        // outcome for the Step tally (K1.2).
        inline bool try_swap_plan(
            RenderPathPodState& state,
            const RenderPathRecipe& candidate,
            const RenderPathCompiler& compiler,
            const RenderPathCapabilitySet& caps,
            std::pmr::vector<RenderPathEvent>& events)
        {
            bool accepted = false;
            compile_render_path_plan(candidate, compiler, caps)
                .transform([&](RenderPathExecutionPlan&& candidate_plan) {
                    state.recipe = candidate;
                    state.plan = std::move(candidate_plan);
                    state.plan_generation += 1;
                    accepted = true;
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
            return accepted;
        }
    } // namespace detail

    namespace detail
    {
        // --- named per-intent arrows (K2.2: transition bodies live here; the
        // public gateway below is only the assembly point) ----------------

        // K3.2 house answer (Run A): a consumed command never emits nothing.
        // Same-value commands emit an *Unchanged fact (a no-op is not a
        // failure — it never touches the error rail); rejected swaps emit
        // PATH_SWAP_REJECTED; accepted transitions emit their change fact.
        // Decision recorded in docs/backlog/kdba_kleisli_migration_plan.md.

        inline void apply_select_path_preset(
            RenderPathPodState& state,
            const SelectPathPresetIntent& cmd,
            const RenderPathCompiler& compiler,
            const RenderPathCapabilitySet& caps,
            std::pmr::vector<RenderPathEvent>& events,
            RenderPathStep& step)
        {
            if (try_swap_plan(state, cmd.recipe, compiler, caps, events))
            {
                step.commands_applied += 1;
            }
            else
            {
                step.swaps_rejected += 1;
            }
        }

        inline void apply_set_technique(
            RenderPathPodState& state,
            const SetRenderingTechniqueIntent& cmd,
            const RenderPathCompiler& compiler,
            const RenderPathCapabilitySet& caps,
            std::pmr::vector<RenderPathEvent>& events,
            RenderPathStep& step)
        {
            if (cmd.technique == state.recipe.render_technique)
            {
                step.noops_observed += 1;
                events.push_back(TechniqueUnchangedEvent{ cmd.technique });
                return;
            }
            const RenderPathRenderingTechnique previous = state.recipe.render_technique;
            RenderPathRecipe candidate = state.recipe;
            candidate.render_technique = cmd.technique;
            candidate.technique_mode = technique_mode_for(cmd.technique);
            if (try_swap_plan(state, candidate, compiler, caps, events))
            {
                step.commands_applied += 1;
                events.push_back(TechniqueSwitchedEvent{ previous, cmd.technique });
            }
            else
            {
                step.swaps_rejected += 1;
            }
        }

        inline void apply_set_view_culling(
            RenderPathPodState& state,
            const SetViewCullingModeIntent& cmd,
            const RenderPathCompiler& compiler,
            const RenderPathCapabilitySet& caps,
            std::pmr::vector<RenderPathEvent>& events,
            RenderPathStep& step)
        {
            if (cmd.mode == state.recipe.view_culling)
            {
                step.noops_observed += 1;
                events.push_back(ViewCullingUnchangedEvent{ cmd.mode });
                return;
            }
            const RenderPathCullingMode previous = state.recipe.view_culling;
            RenderPathRecipe candidate = state.recipe;
            candidate.view_culling = cmd.mode;
            if (try_swap_plan(state, candidate, compiler, caps, events))
            {
                step.commands_applied += 1;
                events.push_back(ViewCullingModeChangedEvent{ previous, cmd.mode });
            }
            else
            {
                step.swaps_rejected += 1;
            }
        }

        inline void apply_set_shadow_culling(
            RenderPathPodState& state,
            const SetShadowCullingModeIntent& cmd,
            const RenderPathCompiler& compiler,
            const RenderPathCapabilitySet& caps,
            std::pmr::vector<RenderPathEvent>& events,
            RenderPathStep& step)
        {
            if (cmd.mode == state.recipe.shadow_culling)
            {
                step.noops_observed += 1;
                events.push_back(ShadowCullingUnchangedEvent{ cmd.mode });
                return;
            }
            const RenderPathCullingMode previous = state.recipe.shadow_culling;
            RenderPathRecipe candidate = state.recipe;
            candidate.shadow_culling = cmd.mode;
            if (try_swap_plan(state, candidate, compiler, caps, events))
            {
                step.commands_applied += 1;
                events.push_back(ShadowCullingModeChangedEvent{ previous, cmd.mode });
            }
            else
            {
                step.swaps_rejected += 1;
            }
        }

        inline void apply_set_runtime_toggle(
            RenderPathPodState& state,
            const SetRuntimeToggleIntent& cmd,
            std::pmr::vector<RenderPathEvent>& events,
            RenderPathStep& step)
        {
            apply_runtime_toggle(state.recipe.runtime_defaults, cmd.toggle, cmd.enabled);
            if (state.plan_generation != 0)
            {
                apply_runtime_toggle(state.plan.runtime_state, cmd.toggle, cmd.enabled);
            }
            events.push_back(RuntimeToggledEvent{ cmd.toggle, cmd.enabled });
            step.commands_applied += 1;
        }
    } // namespace detail

    // Apply one batch of renderpath commands against the current pod state
    // (assembly point only — transition bodies live in the named per-intent
    // arrows above, Rule 2 as amended). compiler + caps are pure value
    // context; events land on the caller's frame arena via the pmr event
    // vector. Returns the batch outcome summary (RenderPathStep, K1.2).
    inline RenderPathStep renderpath_gateway(
        RenderPathPodState& state,
        std::span<const RenderPathCommand> commands,
        const RenderPathCompiler& compiler,
        const RenderPathCapabilitySet& caps,
        std::pmr::vector<RenderPathEvent>& events)
    {
        RenderPathStep step{};
        for (const RenderPathCommand& command : commands)
        {
            std::visit([&](const auto& cmd) {
                using T = std::decay_t<decltype(cmd)>;

                if constexpr (std::is_same_v<T, SelectPathPresetIntent>)
                {
                    detail::apply_select_path_preset(state, cmd, compiler, caps, events, step);
                }
                else if constexpr (std::is_same_v<T, SetRenderingTechniqueIntent>)
                {
                    detail::apply_set_technique(state, cmd, compiler, caps, events, step);
                }
                else if constexpr (std::is_same_v<T, SetViewCullingModeIntent>)
                {
                    detail::apply_set_view_culling(state, cmd, compiler, caps, events, step);
                }
                else if constexpr (std::is_same_v<T, SetShadowCullingModeIntent>)
                {
                    detail::apply_set_shadow_culling(state, cmd, compiler, caps, events, step);
                }
                else if constexpr (std::is_same_v<T, SetRuntimeToggleIntent>)
                {
                    detail::apply_set_runtime_toggle(state, cmd, events, step);
                }
            }, command);
        }
        step.plan_generation = state.plan_generation;
        return step;
    }
} // namespace shs::renderpath
