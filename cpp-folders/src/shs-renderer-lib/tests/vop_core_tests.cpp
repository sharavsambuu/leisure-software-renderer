#include <cmath>
#include <cstdio>
#include <memory>
#include <string>

#include "shs/core/context.hpp"
#include "shs/frame/frame_params.hpp"
#include "shs/input/camera_commands.hpp"
#include "shs/input/command_processor.hpp"
#include "shs/input/value_actions.hpp"
#include "shs/input/value_input_latch.hpp"
#include "shs/pipeline/pluggable_pipeline.hpp"

namespace
{
    bool approx_eq(float a, float b, float eps = 1e-4f)
    {
        return std::abs(a - b) <= eps;
    }

    struct DummyBackend final : shs::IRenderBackend
    {
        explicit DummyBackend(shs::RenderBackendType t)
            : type_(t)
        {}

        shs::RenderBackendType type() const override { return type_; }
        void begin_frame(shs::Context&, const shs::RenderBackendFrameInfo&) override { ++begin_count; }
        void end_frame(shs::Context&, const shs::RenderBackendFrameInfo&) override { ++end_count; }

        shs::RenderBackendType type_ = shs::RenderBackendType::Software;
        int begin_count = 0;
        int end_count = 0;
    };

    struct DummyPass final : shs::IRenderPass
    {
        DummyPass(std::string id, shs::RenderBackendType preferred, shs::RHIQueueClass queue)
            : id_(std::move(id)), preferred_(preferred), queue_(queue)
        {}

        const char* id() const override { return id_.c_str(); }
        shs::RenderBackendType preferred_backend() const override { return preferred_; }
        shs::RHIQueueClass preferred_queue() const override { return queue_; }
        bool supports_backend(shs::RenderBackendType) const override { return true; }
        shs::TechniquePassContract describe_contract() const override
        {
            shs::TechniquePassContract c{};
            c.role = shs::TechniquePassRole::Visibility;
            return c;
        }
        shs::PassExecutionResult execute_resolved(shs::Context&, const shs::PassExecutionRequest& request) override
        {
            if (!request.valid) return shs::PassExecutionResult::not_executed();
            return shs::PassExecutionResult::executed_no_outputs();
        }

        std::string id_{};
        shs::RenderBackendType preferred_ = shs::RenderBackendType::Software;
        shs::RHIQueueClass queue_ = shs::RHIQueueClass::Graphics;
    };

    struct RejectingRequestPass final : shs::IRenderPass
    {
        RejectingRequestPass(int* build_count, int* execute_count)
            : build_count_(build_count), execute_count_(execute_count)
        {}

        const char* id() const override { return "rejecting_request"; }
        shs::RenderBackendType preferred_backend() const override { return shs::RenderBackendType::Software; }
        bool supports_backend(shs::RenderBackendType) const override { return true; }
        shs::TechniquePassContract describe_contract() const override
        {
            shs::TechniquePassContract c{};
            c.role = shs::TechniquePassRole::Visibility;
            return c;
        }
        shs::PassExecutionRequest build_execution_request(
            const shs::Context& ctx,
            const shs::Scene& scene,
            const shs::FrameParams& fp,
            shs::RTRegistry& rtr) const override
        {
            if (build_count_) ++(*build_count_);
            shs::PassExecutionRequest out = shs::IRenderPass::build_execution_request(ctx, scene, fp, rtr);
            out.valid = false;
            return out;
        }
        shs::PassExecutionResult execute_resolved(shs::Context&, const shs::PassExecutionRequest& request) override
        {
            if (!request.valid) return shs::PassExecutionResult::not_executed();
            if (execute_count_) ++(*execute_count_);
            return shs::PassExecutionResult::executed_no_outputs();
        }

        int* build_count_ = nullptr;
        int* execute_count_ = nullptr;
    };

    struct ContractPass final : shs::IRenderPass
    {
        ContractPass(std::string id, shs::TechniquePassContract contract)
            : id_(std::move(id)), contract_(contract)
        {}

        const char* id() const override { return id_.c_str(); }
        shs::RenderBackendType preferred_backend() const override { return shs::RenderBackendType::Software; }
        bool supports_backend(shs::RenderBackendType) const override { return true; }
        shs::TechniquePassContract describe_contract() const override { return contract_; }
        shs::PassExecutionResult execute_resolved(shs::Context&, const shs::PassExecutionRequest& request) override
        {
            if (!request.valid) return shs::PassExecutionResult::not_executed();
            return shs::PassExecutionResult::executed_no_outputs();
        }

        std::string id_{};
        shs::TechniquePassContract contract_{};
    };

    struct ResolvedOnlyPass final : shs::IRenderPass
    {
        ResolvedOnlyPass(int* execute_count, int* resolved_count)
            : execute_count_(execute_count), resolved_count_(resolved_count)
        {}

        const char* id() const override { return "resolved_only"; }
        shs::RenderBackendType preferred_backend() const override { return shs::RenderBackendType::Software; }
        bool supports_backend(shs::RenderBackendType) const override { return true; }
        shs::TechniquePassContract describe_contract() const override
        {
            shs::TechniquePassContract c{};
            c.role = shs::TechniquePassRole::Visibility;
            return c;
        }
        void execute(shs::Context&, const shs::Scene&, const shs::FrameParams&, shs::RTRegistry&)
        {
            if (execute_count_) ++(*execute_count_);
        }
        shs::PassExecutionResult execute_resolved(shs::Context&, const shs::PassExecutionRequest& request) override
        {
            if (!request.valid) return shs::PassExecutionResult::not_executed();
            if (resolved_count_) ++(*resolved_count_);
            return shs::PassExecutionResult::executed_no_outputs();
        }

        int* execute_count_ = nullptr;
        int* resolved_count_ = nullptr;
    };

    bool test_runtime_action_reducer()
    {
        shs::RuntimeState s{};
        s.camera.pos = glm::vec3(0.0f, 0.0f, 0.0f);
        s.camera.yaw = glm::half_pi<float>();
        s.camera.pitch = 0.0f;
        s.enable_light_shafts = true;
        s.bot_enabled = false;
        s.quit_requested = false;

        std::vector<shs::RuntimeAction> actions{};
        actions.push_back(shs::make_move_local_action(glm::vec3(0.0f, 0.0f, 1.0f), 4.0f));
        actions.push_back(shs::make_look_action(10.0f, -5.0f, 0.01f));
        actions.push_back(shs::make_toggle_light_shafts_action());
        actions.push_back(shs::make_toggle_bot_action());
        actions.push_back(shs::make_quit_action());

        const shs::RuntimeState out = shs::reduce_runtime_state(s, actions, 0.5f);
        if (!approx_eq(out.camera.pos.z, 2.0f)) return false;
        if (!approx_eq(out.camera.yaw, glm::half_pi<float>() + 0.1f)) return false;
        if (!approx_eq(out.camera.pitch, 0.05f)) return false;
        if (out.enable_light_shafts) return false;
        if (!out.bot_enabled) return false;
        if (!out.quit_requested) return false;
        return true;
    }

    bool test_runtime_input_latch_reducer()
    {
        shs::RuntimeInputLatch s{};
        std::vector<shs::RuntimeInputEvent> ev{};
        ev.push_back(shs::make_bool_input_event(shs::RuntimeInputEventType::SetForward, true));
        ev.push_back(shs::make_bool_input_event(shs::RuntimeInputEventType::SetRightMouseDown, true));
        ev.push_back(shs::make_mouse_delta_input_event(3.0f, -2.0f));
        ev.push_back(shs::make_mouse_delta_input_event(1.0f, 5.0f));
        ev.push_back(shs::make_quit_input_event());

        shs::RuntimeInputLatch out = shs::reduce_runtime_input_latch(s, ev);
        if (!out.forward) return false;
        if (!out.right_mouse_down) return false;
        if (!approx_eq(out.mouse_dx_accum, 4.0f)) return false;
        if (!approx_eq(out.mouse_dy_accum, 3.0f)) return false;
        if (!out.quit_requested) return false;

        out = shs::clear_runtime_input_frame_deltas(out);
        if (!approx_eq(out.mouse_dx_accum, 0.0f)) return false;
        if (!approx_eq(out.mouse_dy_accum, 0.0f)) return false;
        if (!out.forward) return false;
        return true;
    }

    bool test_pipeline_execution_plan()
    {
        shs::Context ctx{};
        DummyBackend sw(shs::RenderBackendType::Software);
        DummyBackend vk(shs::RenderBackendType::Vulkan);
        ctx.register_backend(&sw);
        ctx.register_backend(&vk);
        ctx.set_primary_backend(&sw);

        shs::PluggablePipeline pipeline{};
        pipeline.add_pass_instance(std::make_unique<DummyPass>(
            "cpu_setup",
            shs::RenderBackendType::Software,
            shs::RHIQueueClass::Graphics));
        pipeline.add_pass_instance(std::make_unique<DummyPass>(
            "gpu_light",
            shs::RenderBackendType::Vulkan,
            shs::RHIQueueClass::Compute));

        shs::FrameParams fp{};
        fp.technique.mode = shs::TechniqueMode::Forward;
        fp.technique.active_modes_mask = shs::technique_mode_mask_all();
        fp.hybrid.allow_cross_backend_passes = false;
        fp.hybrid.strict_backend_availability = false;

        shs::RTRegistry rtr{};
        const shs::PipelineExecutionPlan plan = pipeline.build_execution_plan(ctx, fp, rtr);
        if (plan.passes.empty()) return false;
        if (plan.passes[0].label != "cpu_setup") return false;
        if (plan.passes[0].backend_type != shs::RenderBackendType::Software) return false;
        if (plan.passes[0].queue != shs::RHIQueueClass::Graphics) return false;
        if (plan.passes.size() != 1) return false;
        if (plan.report.warnings.empty()) return false;
        return true;
    }

    bool test_command_processor_value_reduce()
    {
        shs::RuntimeState s{};
        s.camera.pos = glm::vec3(0.0f, 0.0f, 0.0f);
        s.camera.yaw = glm::half_pi<float>();
        s.camera.pitch = 0.0f;
        s.enable_light_shafts = true;

        shs::CommandProcessor proc{};
        proc.emplace<shs::MoveCommand>(glm::vec3(0.0f, 0.0f, 1.0f), 4.0f);
        proc.emplace<shs::LookCommand>(10.0f, -5.0f, 0.01f);
        proc.emplace<shs::ToggleLightShaftsCommand>();

        const shs::RuntimeState out = proc.reduce_all(s, 0.5f);
        if (!approx_eq(out.camera.pos.z, 2.0f)) return false;
        if (!approx_eq(out.camera.yaw, glm::half_pi<float>() + 0.1f)) return false;
        if (!approx_eq(out.camera.pitch, 0.05f)) return false;
        if (out.enable_light_shafts) return false;
        return true;
    }

    bool test_pipeline_uses_execution_request_gate()
    {
        int build_count = 0;
        int execute_count = 0;

        shs::Context ctx{};
        DummyBackend sw(shs::RenderBackendType::Software);
        ctx.register_backend(&sw);
        ctx.set_primary_backend(&sw);

        shs::PluggablePipeline pipeline{};
        pipeline.add_pass_instance(std::make_unique<RejectingRequestPass>(&build_count, &execute_count));

        shs::Scene scene{};
        shs::FrameParams fp{};
        fp.w = 8;
        fp.h = 8;
        fp.hybrid.emulate_vulkan_runtime = false;
        shs::RTRegistry rtr{};

        pipeline.execute(ctx, scene, fp, rtr);
        if (build_count <= 0) return false;
        if (execute_count != 0) return false;
        return true;
    }

    bool test_profile_config_uses_mode_hints_before_instantiation()
    {
        int create_count = 0;

        shs::PassFactoryRegistry registry{};
        (void)registry.register_factory("hint_only_mode_check", [&create_count]() -> std::unique_ptr<shs::IRenderPass>
        {
            ++create_count;
            return std::make_unique<DummyPass>(
                "hint_only_mode_check",
                shs::RenderBackendType::Software,
                shs::RHIQueueClass::Graphics);
        });

        shs::TechniquePassContract descriptor_contract{};
        descriptor_contract.supported_modes_mask = shs::technique_mode_bit(shs::TechniqueMode::Deferred);
        (void)registry.register_descriptor("hint_only_mode_check", descriptor_contract);

        shs::TechniqueProfile profile{};
        profile.mode = shs::TechniqueMode::Forward;
        profile.passes.push_back(shs::TechniquePassEntry{
            "hint_only_mode_check",
            shs::PassId::Unknown,
            true
        });

        shs::PluggablePipeline pipeline{};
        std::vector<std::string> missing{};
        const bool ok = pipeline.configure_from_profile(registry, profile, &missing);
        if (ok) return false;
        if (create_count != 0) return false;
        if (missing.size() != 1) return false;
        if (missing[0] != "hint_only_mode_check") return false;
        return true;
    }

    bool test_execution_plan_ignores_context_runtime_flags()
    {
        shs::Context ctx{};
        DummyBackend sw(shs::RenderBackendType::Software);
        ctx.register_backend(&sw);
        ctx.set_primary_backend(&sw);

        shs::TechniquePassContract depth_writer{};
        depth_writer.supported_modes_mask = shs::technique_mode_bit(shs::TechniqueMode::ForwardPlus);
        depth_writer.semantics = {
            shs::write_semantic(shs::PassSemantic::Depth, shs::ContractDomain::Software, "depth")
        };

        shs::TechniquePassContract depth_reader{};
        depth_reader.supported_modes_mask = shs::technique_mode_bit(shs::TechniqueMode::ForwardPlus);
        depth_reader.requires_depth_prepass = true;
        depth_reader.semantics = {
            shs::read_semantic(shs::PassSemantic::Depth, shs::ContractDomain::Software, "depth")
        };

        shs::PluggablePipeline pipeline{};
        pipeline.add_pass_instance(std::make_unique<ContractPass>("depth_writer", depth_writer));
        pipeline.add_pass_instance(std::make_unique<ContractPass>("depth_reader", depth_reader));

        shs::FrameParams fp{};
        fp.technique.mode = shs::TechniqueMode::ForwardPlus;
        fp.technique.active_modes_mask = shs::technique_mode_mask_all();
        fp.technique.depth_prepass = true;

        shs::RTRegistry rtr{};
        const shs::PipelineExecutionPlan plan = pipeline.build_execution_plan(ctx, fp, rtr);
        if (plan.passes.size() != 2) return false;
        for (const std::string& w : plan.report.warnings)
        {
            if (w.find("depth_prepass_valid") != std::string::npos) return false;
            if (w.find("light_culling_valid") != std::string::npos) return false;
        }
        for (const std::string& e : plan.report.errors)
        {
            if (e.find("depth_prepass_valid") != std::string::npos) return false;
            if (e.find("light_culling_valid") != std::string::npos) return false;
        }
        return true;
    }

    bool test_pipeline_runtime_uses_execute_resolved()
    {
        int execute_count = 0;
        int resolved_count = 0;

        shs::Context ctx{};
        DummyBackend sw(shs::RenderBackendType::Software);
        ctx.register_backend(&sw);
        ctx.set_primary_backend(&sw);

        shs::PluggablePipeline pipeline{};
        pipeline.add_pass_instance(std::make_unique<ResolvedOnlyPass>(&execute_count, &resolved_count));

        shs::Scene scene{};
        shs::FrameParams fp{};
        fp.w = 8;
        fp.h = 8;
        fp.hybrid.emulate_vulkan_runtime = false;
        shs::RTRegistry rtr{};

        pipeline.execute(ctx, scene, fp, rtr);
        if (execute_count != 0) return false;
        if (resolved_count != 1) return false;
        return true;
    }

}

int main()
{
    const bool ok_actions = test_runtime_action_reducer();
    const bool ok_latch = test_runtime_input_latch_reducer();
    const bool ok_plan = test_pipeline_execution_plan();
    const bool ok_cmds = test_command_processor_value_reduce();
    const bool ok_request_gate = test_pipeline_uses_execution_request_gate();
    const bool ok_profile_hint = test_profile_config_uses_mode_hints_before_instantiation();
    const bool ok_context_flags = test_execution_plan_ignores_context_runtime_flags();
    const bool ok_resolved_only = test_pipeline_runtime_uses_execute_resolved();

    if (!ok_actions) std::fprintf(stderr, "[vop-tests] runtime action reducer failed\n");
    if (!ok_latch) std::fprintf(stderr, "[vop-tests] runtime input latch reducer failed\n");
    if (!ok_plan) std::fprintf(stderr, "[vop-tests] pipeline execution plan failed\n");
    if (!ok_cmds) std::fprintf(stderr, "[vop-tests] command processor value reduce failed\n");
    if (!ok_request_gate) std::fprintf(stderr, "[vop-tests] pipeline request-gate path failed\n");
    if (!ok_profile_hint) std::fprintf(stderr, "[vop-tests] profile mode-hint precheck failed\n");
    if (!ok_context_flags) std::fprintf(stderr, "[vop-tests] context runtime-flag coupling check failed\n");
    if (!ok_resolved_only) std::fprintf(stderr, "[vop-tests] runtime did not use execute_resolved path\n");

    if (!(ok_actions && ok_latch && ok_plan && ok_cmds && ok_request_gate && ok_profile_hint && ok_context_flags && ok_resolved_only)) return 1;
    std::fprintf(stderr, "[vop-tests] all tests passed\n");
    return 0;
}
