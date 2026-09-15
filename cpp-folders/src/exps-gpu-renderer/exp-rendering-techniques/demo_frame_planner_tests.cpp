// GPU-free tests for the pure per-frame demo planner (demo_frame_planner.hpp).
// No SDL/Vulkan/GLM dependency — always configured with the tree.
#include <cstdio>
#include <memory_resource>
#include <vector>

#include "demo_frame_planner.hpp"

namespace
{
    bool test_fallback_plan_from_technique_mode()
    {
        std::pmr::unsynchronized_pool_resource arena{};
        shs::demo::DemoFramePlanInputs in{};
        in.active_plan_valid = false; // no reducer-approved plan yet
        in.technique_mode = shs::TechniqueMode::ForwardPlus;

        const shs::demo::DemoFramePlan plan = shs::demo::plan_demo_frame(in, &arena);
        if (!plan.resolved_plan.valid) return false;
        if (plan.resolved_plan.recipe_name != "fallback_forward_plus") return false;
        if (plan.resolved_plan.pass_chain.empty()) return false;
        // Fallback chain comes from the default technique profile.
        bool has_forward_plus = false;
        for (const auto& p : plan.resolved_plan.pass_chain)
        {
            if (p.pass_id == shs::PassId::PBRForwardPlus) has_forward_plus = true;
        }
        if (!has_forward_plus) return false;
        // Single-threaded: no secondary recording command, just the chain.
        if (plan.commands.size() != 1) return false;
        if (!std::holds_alternative<shs::demo::DemoCmdExecutePassChain>(plan.commands[0])) return false;
        return true;
    }

    bool test_active_plan_preferred_when_valid()
    {
        std::pmr::unsynchronized_pool_resource arena{};
        shs::demo::DemoFramePlanInputs in{};
        in.active_plan_valid = true;
        in.active_plan.recipe_name = "demo_forward_plus";
        in.active_plan.valid = true;
        in.active_plan.pass_chain = {
            shs::RenderPathCompiledPass{"shadow_map", shs::PassId::ShadowMap, false},
            shs::RenderPathCompiledPass{"pbr_forward_plus", shs::PassId::PBRForwardPlus, true}};

        const shs::demo::DemoFramePlan plan = shs::demo::plan_demo_frame(in, &arena);
        if (plan.resolved_plan.recipe_name != "demo_forward_plus") return false;
        if (plan.resolved_plan.pass_chain.size() != 2) return false;
        if (plan.resolved_plan.pass_chain[0].pass_id != shs::PassId::ShadowMap) return false;
        return true;
    }

    bool test_gates_and_secondary_decisions()
    {
        std::pmr::unsynchronized_pool_resource arena{};
        shs::demo::DemoFramePlanInputs in{};
        in.active_plan_valid = true;
        in.active_plan.valid = true;
        in.active_plan.pass_chain = {
            shs::RenderPathCompiledPass{"depth_prepass", shs::PassId::DepthPrepass, true},
            shs::RenderPathCompiledPass{"pbr_forward_plus", shs::PassId::PBRForwardPlus, true},
            shs::RenderPathCompiledPass{"motion_blur", shs::PassId::MotionBlur, false},
            shs::RenderPathCompiledPass{"dof", shs::PassId::DepthOfField, false}};
        in.depth_prepass_enabled = true;
        in.scene_pass_enabled = false;
        in.multithread_recording_enabled = true;
        in.motion_blur_enabled = true;
        in.depth_of_field_enabled = false; // disabled by composition even though in plan

        const shs::demo::DemoFramePlan plan = shs::demo::plan_demo_frame(in, &arena);
        if (!plan.gates.depth_prepass || plan.gates.scene) return false;
        if (!plan.gates.has_motion_blur_pass) return false;
        if (plan.gates.has_depth_of_field_pass) return false; // in plan but gated off

        // Multithreaded + depth enabled -> secondaries command first.
        if (plan.commands.size() != 2) return false;
        const auto* sec = std::get_if<shs::demo::DemoCmdRecordSecondaries>(&plan.commands[0]);
        if (!sec || !sec->depth || sec->scene) return false;
        if (!std::holds_alternative<shs::demo::DemoCmdExecutePassChain>(plan.commands[1])) return false;
        return true;
    }

    bool test_followup_decisions()
    {
        std::pmr::unsynchronized_pool_resource arena{};
        shs::demo::DemoFramePlan plan(&arena);

        // Scene pass executed -> no clear-only, history + snapshot copies.
        shs::demo::DemoFrameDispatchSummary ok_summary{};
        ok_summary.ok = true;
        ok_summary.scene_pass_executed = true;
        shs::demo::plan_demo_frame_followups(ok_summary, plan.commands);
        if (plan.commands.size() != 2) return false;
        if (!std::holds_alternative<shs::demo::DemoCmdHistoryColorCopy>(plan.commands[0])) return false;
        if (!std::holds_alternative<shs::demo::DemoCmdPhaseFSnapshotCopy>(plan.commands[1])) return false;

        // Scene pass skipped -> clear-only first.
        shs::demo::DemoFrameDispatchSummary no_scene{};
        no_scene.ok = true;
        no_scene.scene_pass_executed = false;
        plan.commands.clear();
        shs::demo::plan_demo_frame_followups(no_scene, plan.commands);
        if (plan.commands.size() != 3) return false;
        if (!std::holds_alternative<shs::demo::DemoCmdSceneClearOnly>(plan.commands[0])) return false;
        return true;
    }

    bool test_plan_is_deterministic()
    {
        std::pmr::unsynchronized_pool_resource arena{};
        shs::demo::DemoFramePlanInputs in{};
        in.active_plan_valid = true;
        in.active_plan.valid = true;
        in.active_plan.pass_chain = {
            shs::RenderPathCompiledPass{"tonemap", shs::PassId::Tonemap, true}};
        in.motion_blur_enabled = true;

        const shs::demo::DemoFramePlan a = shs::demo::plan_demo_frame(in, &arena);
        const shs::demo::DemoFramePlan b = shs::demo::plan_demo_frame(in, &arena);
        if (a.resolved_plan.pass_chain.size() != b.resolved_plan.pass_chain.size()) return false;
        if (a.gates.has_motion_blur_pass != b.gates.has_motion_blur_pass) return false;
        if (a.commands.size() != b.commands.size()) return false;
        for (size_t i = 0; i < a.commands.size(); ++i)
        {
            if (a.commands[i].index() != b.commands[i].index()) return false;
        }
        return true;
    }
} // namespace

int main()
{
    struct NamedCheck
    {
        const char* name;
        bool (*fn)();
    };
    const NamedCheck checks[] = {
        {"fallback_plan_from_technique_mode", test_fallback_plan_from_technique_mode},
        {"active_plan_preferred_when_valid", test_active_plan_preferred_when_valid},
        {"gates_and_secondary_decisions", test_gates_and_secondary_decisions},
        {"followup_decisions", test_followup_decisions},
        {"plan_is_deterministic", test_plan_is_deterministic},
    };
    bool all_pass = true;
    for (const NamedCheck& check : checks)
    {
        const bool ok = check.fn();
        std::fprintf(stderr, "[%s] %s\n", ok ? "PASS" : "FAIL", check.name);
        all_pass = all_pass && ok;
    }
    if (!all_pass)
    {
        std::fprintf(stderr, "shs_demo_frame_planner_tests: FAILED\n");
        return 1;
    }
    std::fprintf(
        stderr,
        "shs_demo_frame_planner_tests: all %zu checks passed\n",
        sizeof(checks) / sizeof(checks[0]));
    return 0;
}

