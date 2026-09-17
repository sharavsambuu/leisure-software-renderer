#pragma once

/*
    SHS RENDERER SAN

    FILE: demo_frame_planner.hpp
    PURPOSE: Pure per-frame demo planner (Run 1 / P3 task 3, arch doc §5.4).
            spatial_fx-style: plain-value inputs in, ordered DemoFrameCommand
            span out, built on the caller's frame arena. GPU-free, no
            SDL/Vulkan — the executor edge translates commands into recording.
*/

#include <cstdint>
#include <memory_resource>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <shs/renderpath/planning/pass_id.hpp>
#include <shs/renderpath/planning/render_path_compiler.hpp>
#include <shs/renderpath/planning/technique_profile.hpp>
#include <shs/frame/technique_mode.hpp>

namespace shs::demo
{
    // Plain-value inputs gathered from the edges each frame; no app references.
    struct DemoFramePlanInputs
    {
        // Path state (reducer-driven; executor holds the active plan).
        shs::renderpath::RenderPathExecutionPlan active_plan{};
        bool active_plan_valid = false;
        shs::render::TechniqueMode technique_mode = shs::TechniqueMode::Deferred;

        // Feature toggles.
        bool depth_prepass_enabled = false;
        bool scene_pass_enabled = false;
        bool light_culling_enabled = false;
        bool gpu_light_culler_enabled = false;
        bool multithread_recording_enabled = false;

        // Composition-gated debug passes.
        bool ssao_enabled = false;
        bool motion_blur_enabled = false;
        bool depth_of_field_enabled = false;
        bool taa_enabled = false;
    };

    // Resolved per-pass gates the executor edge copies into its context.
    struct DemoFramePassGates
    {
        bool depth_prepass = false;
        bool scene = false;
        bool light_culling = false;
        bool gpu_light_culler = false;
        bool has_motion_blur_pass = false;
        bool has_depth_of_field_pass = false;
    };

    // Closed per-frame command vocabulary (value stream; edge translates).
    struct DemoCmdRecordSecondaries
    {
        bool depth = false;
        bool scene = false;
    };
    struct DemoCmdExecutePassChain
    {
        uint8_t reserved = 0;
    };
    struct DemoCmdSceneClearOnly
    {
        uint8_t reserved = 0;
    };
    struct DemoCmdHistoryColorCopy
    {
        uint8_t reserved = 0;
    };
    struct DemoCmdPhaseFSnapshotCopy
    {
        uint8_t reserved = 0;
    };

    using DemoFrameCommand = std::variant<
        DemoCmdRecordSecondaries,
        DemoCmdExecutePassChain,
        DemoCmdSceneClearOnly,
        DemoCmdHistoryColorCopy,
        DemoCmdPhaseFSnapshotCopy>;

    struct DemoFramePlan
    {
        shs::renderpath::RenderPathExecutionPlan resolved_plan{};
        DemoFramePassGates gates{};
        std::pmr::vector<DemoFrameCommand> commands{};

        explicit DemoFramePlan(std::pmr::memory_resource* arena) : commands(arena) {}
    };

    // PURE: resolve the plan (active reducer-approved plan, or technique-profile
    // fallback), compute the per-pass gates, and emit the ordered command span.
    // Deterministic: identical inputs produce identical plans.
    inline DemoFramePlan plan_demo_frame(const DemoFramePlanInputs& in, std::pmr::memory_resource* arena)
    {
        DemoFramePlan plan(arena);

        if (in.active_plan_valid && !in.active_plan.pass_chain.empty())
        {
            plan.resolved_plan = in.active_plan;
        }
        else
        {
            // Fallback plan from the technique profile (pre-pod behavior).
            shs::renderpath::RenderPathExecutionPlan fallback{};
            fallback.recipe_name = std::string("fallback_") + shs::render::technique_mode_name(in.technique_mode);
            fallback.backend = shs::RenderBackendType::Vulkan;
            fallback.technique_mode = in.technique_mode;
            fallback.valid = true;
            const shs::renderpath::TechniqueProfile profile = shs::renderpath::make_default_technique_profile(in.technique_mode);
            fallback.pass_chain.reserve(profile.passes.size());
            for (const auto& p : profile.passes)
            {
                fallback.pass_chain.push_back(shs::renderpath::RenderPathCompiledPass{p.id, p.pass_id, p.required});
            }
            plan.resolved_plan = std::move(fallback);
        }

        const auto plan_has_pass = [&plan](shs::renderpath::PassId pass_id) -> bool {
            for (const auto& p : plan.resolved_plan.pass_chain)
            {
                if (p.pass_id == pass_id) return true;
                if (shs::renderpath::parse_pass_id(p.id) == pass_id) return true;
            }
            return false;
        };

        plan.gates.depth_prepass = in.depth_prepass_enabled;
        plan.gates.scene = in.scene_pass_enabled;
        plan.gates.light_culling = in.light_culling_enabled;
        plan.gates.gpu_light_culler = in.gpu_light_culler_enabled;
        plan.gates.has_motion_blur_pass =
            plan_has_pass(shs::PassId::MotionBlur) && in.motion_blur_enabled;
        plan.gates.has_depth_of_field_pass =
            plan_has_pass(shs::PassId::DepthOfField) && in.depth_of_field_enabled;
        (void)in.ssao_enabled;
        (void)in.taa_enabled;

        const bool want_secondaries =
            in.multithread_recording_enabled && (plan.gates.depth_prepass || plan.gates.scene);
        if (want_secondaries)
        {
            plan.commands.push_back(
                DemoCmdRecordSecondaries{plan.gates.depth_prepass, plan.gates.scene});
        }
        plan.commands.push_back(DemoCmdExecutePassChain{});
        return plan;
    }

    // What the edge reports after executing the pass chain.
    struct DemoFrameDispatchSummary
    {
        bool ok = true;
        bool scene_pass_executed = false;
    };

    // PURE: followup decisions after the pass chain ran; appends to the same
    // command span so the edge keeps consuming one ordered stream.
    inline void plan_demo_frame_followups(
        const DemoFrameDispatchSummary& summary,
        std::pmr::vector<DemoFrameCommand>& out)
    {
        if (!summary.scene_pass_executed)
        {
            out.push_back(DemoCmdSceneClearOnly{});
        }
        out.push_back(DemoCmdHistoryColorCopy{});
        out.push_back(DemoCmdPhaseFSnapshotCopy{});
    }
} // namespace shs::demo

