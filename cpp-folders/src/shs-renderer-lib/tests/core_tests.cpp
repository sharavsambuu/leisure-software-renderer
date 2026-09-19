#include <cmath>
#include <cstdio>
#include <memory>
#include <string>

#include "shs/app/context.hpp"
#include "shs/app/session_orchestrator.gateway.hpp"
#include "shs/render/frame/frame_params.hpp"
#include "shs/input/storage/camera_commands.hpp"
#include "shs/input/storage/command_processor.hpp"
#include "shs/input/value_commands.hpp"
#include "shs/input/value_input_latch.hpp"
#include "shs/renderpath/execution/pluggable_pipeline.hpp"

namespace
{
    bool approx_eq(float a, float b, float eps = 1e-4f)
    {
        return std::abs(a - b) <= eps;
    }

    struct DummyBackend final : shs::rhi::IRenderBackend
    {
        explicit DummyBackend(shs::render::RenderBackendType t)
            : type_(t)
        {}

        shs::render::RenderBackendType type() const override { return type_; }
        void begin_frame(shs::app::Context&, const shs::rhi::RenderBackendFrameInfo&) override { ++begin_count; }
        void end_frame(shs::app::Context&, const shs::rhi::RenderBackendFrameInfo&) override { ++end_count; }

        shs::render::RenderBackendType type_ = shs::RenderBackendType::Software;
        int begin_count = 0;
        int end_count = 0;
    };

    struct DummyPass final : shs::renderpath::IRenderPass
    {
        DummyPass(std::string id, shs::render::RenderBackendType preferred, shs::rhi::RHIQueueClass queue)
            : id_(std::move(id)), preferred_(preferred), queue_(queue)
        {}

        const char* id() const override { return id_.c_str(); }
        shs::render::RenderBackendType preferred_backend() const override { return preferred_; }
        shs::rhi::RHIQueueClass preferred_queue() const override { return queue_; }
        bool supports_backend(shs::render::RenderBackendType) const override { return true; }
        shs::renderpath::TechniquePassContract describe_contract() const override
        {
            shs::renderpath::TechniquePassContract c{};
            c.role = shs::TechniquePassRole::Visibility;
            return c;
        }
        shs::renderpath::PassExecutionResult execute_resolved(shs::app::Context&, const shs::renderpath::PassExecutionRequest& request) override
        {
            if (!request.valid) return shs::PassExecutionResult::invalid_request();
            return shs::PassExecutionResult::executed_no_outputs();
        }

        std::string id_{};
        shs::render::RenderBackendType preferred_ = shs::RenderBackendType::Software;
        shs::rhi::RHIQueueClass queue_ = shs::RHIQueueClass::Graphics;
    };

    struct RejectingRequestPass final : shs::renderpath::IRenderPass
    {
        RejectingRequestPass(int* build_count, int* execute_count)
            : build_count_(build_count), execute_count_(execute_count)
        {}

        const char* id() const override { return "rejecting_request"; }
        shs::render::RenderBackendType preferred_backend() const override { return shs::RenderBackendType::Software; }
        bool supports_backend(shs::render::RenderBackendType) const override { return true; }
        shs::renderpath::TechniquePassContract describe_contract() const override
        {
            shs::renderpath::TechniquePassContract c{};
            c.role = shs::TechniquePassRole::Visibility;
            return c;
        }
        shs::renderpath::PassExecutionRequest build_execution_request(
            const shs::app::Context& ctx,
            const shs::scene::Scene& scene,
            const shs::render::FrameParams& fp,
            shs::render::RTRegistry& rtr) const override
        {
            if (build_count_) ++(*build_count_);
            shs::renderpath::PassExecutionRequest out = shs::IRenderPass::build_execution_request(ctx, scene, fp, rtr);
            out.valid = false;
            return out;
        }
        shs::renderpath::PassExecutionResult execute_resolved(shs::app::Context&, const shs::renderpath::PassExecutionRequest& request) override
        {
            if (!request.valid) return shs::PassExecutionResult::invalid_request();
            if (execute_count_) ++(*execute_count_);
            return shs::PassExecutionResult::executed_no_outputs();
        }

        int* build_count_ = nullptr;
        int* execute_count_ = nullptr;
    };

    struct ContractPass final : shs::renderpath::IRenderPass
    {
        ContractPass(std::string id, shs::renderpath::TechniquePassContract contract)
            : id_(std::move(id)), contract_(contract)
        {}

        const char* id() const override { return id_.c_str(); }
        shs::render::RenderBackendType preferred_backend() const override { return shs::RenderBackendType::Software; }
        bool supports_backend(shs::render::RenderBackendType) const override { return true; }
        shs::renderpath::TechniquePassContract describe_contract() const override { return contract_; }
        shs::renderpath::PassExecutionResult execute_resolved(shs::app::Context&, const shs::renderpath::PassExecutionRequest& request) override
        {
            if (!request.valid) return shs::PassExecutionResult::invalid_request();
            return shs::PassExecutionResult::executed_no_outputs();
        }

        std::string id_{};
        shs::renderpath::TechniquePassContract contract_{};
    };

    struct ResolvedOnlyPass final : shs::renderpath::IRenderPass
    {
        ResolvedOnlyPass(int* execute_count, int* resolved_count)
            : execute_count_(execute_count), resolved_count_(resolved_count)
        {}

        const char* id() const override { return "resolved_only"; }
        shs::render::RenderBackendType preferred_backend() const override { return shs::RenderBackendType::Software; }
        bool supports_backend(shs::render::RenderBackendType) const override { return true; }
        shs::renderpath::TechniquePassContract describe_contract() const override
        {
            shs::renderpath::TechniquePassContract c{};
            c.role = shs::TechniquePassRole::Visibility;
            return c;
        }
        void execute(shs::app::Context&, const shs::scene::Scene&, const shs::render::FrameParams&, shs::render::RTRegistry&)
        {
            if (execute_count_) ++(*execute_count_);
        }
        shs::renderpath::PassExecutionResult execute_resolved(shs::app::Context&, const shs::renderpath::PassExecutionRequest& request) override
        {
            if (!request.valid) return shs::PassExecutionResult::invalid_request();
            if (resolved_count_) ++(*resolved_count_);
            return shs::PassExecutionResult::executed_no_outputs();
        }

        int* execute_count_ = nullptr;
        int* resolved_count_ = nullptr;
    };

    bool test_runtime_command_gateway()
    {
        shs::app::SessionState s{};
        s.camera.pos = glm::vec3(0.0f, 0.0f, 0.0f);
        s.camera.yaw = glm::half_pi<float>();
        s.camera.pitch = 0.0f;
        s.enable_light_shafts = true;
        s.bot_enabled = false;
        s.quit_requested = false;

        std::vector<shs::input::RuntimeCommand> commands{};
        commands.push_back(shs::input::make_move_local_intent(glm::vec3(0.0f, 0.0f, 1.0f), 4.0f));
        commands.push_back(shs::input::make_look_intent(10.0f, -5.0f, 0.01f));
        commands.push_back(shs::input::make_toggle_light_shafts_intent());
        commands.push_back(shs::input::make_toggle_bot_intent());
        commands.push_back(shs::input::make_quit_intent());

        std::pmr::monotonic_buffer_resource arena{4096};
        std::pmr::vector<shs::input::InputEvent> events{&arena};
        shs::app::session_orchestrate(s, std::span<const shs::input::RuntimeCommand>{commands.data(), commands.size()},
            shs::input::InputContext{0.5f}, events);
        const shs::app::SessionState& out = s;
        if (!approx_eq(out.camera.pos.z, 2.0f)) return false;
        if (!approx_eq(out.camera.yaw, glm::half_pi<float>() + 0.1f)) return false;
        if (!approx_eq(out.camera.pitch, 0.05f)) return false;
        if (out.enable_light_shafts) return false;
        if (!out.bot_enabled) return false;
        if (!out.quit_requested) return false;
        return true;
    }

    bool test_input_latch_gateway()
    {
        shs::input::RuntimeInputLatch s{};
        std::vector<shs::input::RuntimeInputEvent> ev{};
        ev.push_back(shs::input::make_bool_input_event(shs::RuntimeInputEventType::SetForward, true));
        ev.push_back(shs::input::make_bool_input_event(shs::RuntimeInputEventType::SetRightMouseDown, true));
        ev.push_back(shs::input::make_mouse_delta_input_event(3.0f, -2.0f));
        ev.push_back(shs::input::make_mouse_delta_input_event(1.0f, 5.0f));
        ev.push_back(shs::input::make_quit_input_event());

        shs::input::RuntimeInputLatch out = shs::input::input_latch_gateway(s, ev);
        if (!out.forward) return false;
        if (!out.right_mouse_down) return false;
        if (!approx_eq(out.mouse_dx_accum, 4.0f)) return false;
        if (!approx_eq(out.mouse_dy_accum, 3.0f)) return false;
        if (!out.quit_requested) return false;

        out = shs::input::clear_runtime_input_frame_deltas(out);
        if (!approx_eq(out.mouse_dx_accum, 0.0f)) return false;
        if (!approx_eq(out.mouse_dy_accum, 0.0f)) return false;
        if (!out.forward) return false;
        return true;
    }

    bool test_pipeline_execution_plan()
    {
        shs::app::Context ctx{};
        DummyBackend sw(shs::RenderBackendType::Software);
        DummyBackend vk(shs::RenderBackendType::Vulkan);
        ctx.register_backend(&sw);
        ctx.register_backend(&vk);
        ctx.set_primary_backend(&sw);

        shs::renderpath::PluggablePipeline pipeline{};
        pipeline.add_pass_instance(std::make_unique<DummyPass>(
            "cpu_setup",
            shs::RenderBackendType::Software,
            shs::RHIQueueClass::Graphics));
        pipeline.add_pass_instance(std::make_unique<DummyPass>(
            "gpu_light",
            shs::RenderBackendType::Vulkan,
            shs::RHIQueueClass::Compute));

        shs::render::FrameParams fp{};
        fp.technique.mode = shs::TechniqueMode::Forward;
        fp.technique.active_modes_mask = shs::render::technique_mode_mask_all();
        fp.hybrid.allow_cross_backend_passes = false;
        fp.hybrid.strict_backend_availability = false;

        shs::render::RTRegistry rtr{};
        const shs::renderpath::PipelineExecutionPlan plan = pipeline.build_execution_plan(ctx, fp, rtr);
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
        shs::app::SessionState s{};
        s.camera.pos = glm::vec3(0.0f, 0.0f, 0.0f);
        s.camera.yaw = glm::half_pi<float>();
        s.camera.pitch = 0.0f;
        s.enable_light_shafts = true;

        shs::input::CommandProcessor proc{};
        proc.emplace<shs::input::MoveCommand>(glm::vec3(0.0f, 0.0f, 1.0f), 4.0f);
        proc.emplace<shs::input::LookCommand>(10.0f, -5.0f, 0.01f);
        proc.emplace<shs::input::ToggleLightShaftsCommand>();

        // Step 4.1: the processor is translation-only now; the host applies
        // the collected batch through the explicit app orchestrator.
        const std::vector<shs::input::RuntimeCommand> commands = proc.collect_runtime_commands();
        std::pmr::monotonic_buffer_resource arena{1024};
        std::pmr::vector<shs::input::InputEvent> events{&arena};
        shs::app::session_orchestrate(s,
            std::span<const shs::input::RuntimeCommand>{commands.data(), commands.size()},
            shs::input::InputContext{0.5f}, events);
        const shs::app::SessionState& out = s;
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

        shs::app::Context ctx{};
        DummyBackend sw(shs::RenderBackendType::Software);
        ctx.register_backend(&sw);
        ctx.set_primary_backend(&sw);

        shs::renderpath::PluggablePipeline pipeline{};
        pipeline.add_pass_instance(std::make_unique<RejectingRequestPass>(&build_count, &execute_count));

        shs::scene::Scene scene{};
        shs::render::FrameParams fp{};
        fp.w = 8;
        fp.h = 8;
        fp.hybrid.emulate_vulkan_runtime = false;
        shs::render::RTRegistry rtr{};

        pipeline.execute(ctx, scene, fp, rtr);
        if (build_count <= 0) return false;
        if (execute_count != 0) return false;
        return true;
    }

    bool test_profile_config_uses_mode_hints_before_instantiation()
    {
        int create_count = 0;

        shs::renderpath::PassFactoryRegistry registry{};
        (void)registry.register_factory("hint_only_mode_check", [&create_count]() -> std::unique_ptr<shs::renderpath::IRenderPass>
        {
            ++create_count;
            return std::make_unique<DummyPass>(
                "hint_only_mode_check",
                shs::RenderBackendType::Software,
                shs::RHIQueueClass::Graphics);
        });

        shs::renderpath::TechniquePassContract descriptor_contract{};
        descriptor_contract.supported_modes_mask = shs::render::technique_mode_bit(shs::TechniqueMode::Deferred);
        (void)registry.register_descriptor("hint_only_mode_check", descriptor_contract);

        shs::renderpath::TechniqueProfile profile{};
        profile.mode = shs::TechniqueMode::Forward;
        profile.passes.push_back(shs::renderpath::TechniquePassEntry{
            "hint_only_mode_check",
            shs::PassId::Unknown,
            true
        });

        shs::renderpath::PluggablePipeline pipeline{};
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
        shs::app::Context ctx{};
        DummyBackend sw(shs::RenderBackendType::Software);
        ctx.register_backend(&sw);
        ctx.set_primary_backend(&sw);

        shs::renderpath::TechniquePassContract depth_writer{};
        depth_writer.supported_modes_mask = shs::render::technique_mode_bit(shs::TechniqueMode::ForwardPlus);
        depth_writer.semantics = {
            shs::renderpath::write_semantic(shs::PassSemantic::Depth, shs::render_domain_host(), "depth")
        };

        shs::renderpath::TechniquePassContract depth_reader{};
        depth_reader.supported_modes_mask = shs::render::technique_mode_bit(shs::TechniqueMode::ForwardPlus);
        depth_reader.requires_depth_prepass = true;
        depth_reader.semantics = {
            shs::renderpath::read_semantic(shs::PassSemantic::Depth, shs::render_domain_host(), "depth")
        };

        shs::renderpath::PluggablePipeline pipeline{};
        pipeline.add_pass_instance(std::make_unique<ContractPass>("depth_writer", depth_writer));
        pipeline.add_pass_instance(std::make_unique<ContractPass>("depth_reader", depth_reader));

        shs::render::FrameParams fp{};
        fp.technique.mode = shs::TechniqueMode::ForwardPlus;
        fp.technique.active_modes_mask = shs::render::technique_mode_mask_all();
        fp.technique.depth_prepass = true;

        shs::render::RTRegistry rtr{};
        const shs::renderpath::PipelineExecutionPlan plan = pipeline.build_execution_plan(ctx, fp, rtr);
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

        shs::app::Context ctx{};
        DummyBackend sw(shs::RenderBackendType::Software);
        ctx.register_backend(&sw);
        ctx.set_primary_backend(&sw);

        shs::renderpath::PluggablePipeline pipeline{};
        pipeline.add_pass_instance(std::make_unique<ResolvedOnlyPass>(&execute_count, &resolved_count));

        shs::scene::Scene scene{};
        shs::render::FrameParams fp{};
        fp.w = 8;
        fp.h = 8;
        fp.hybrid.emulate_vulkan_runtime = false;
        shs::render::RTRegistry rtr{};

        pipeline.execute(ctx, scene, fp, rtr);
        if (execute_count != 0) return false;
        if (resolved_count != 1) return false;
        return true;
    }

    // ROP-2.2 (owner ruling 2026-09-18): the pass rim's outcome is a closed
    // vocabulary. The three refusals the old `bool executed` collapsed into one
    // value must be distinguishable facts, and none of them may read as
    // executed — otherwise the §8 `(payload, bool valid)` collapse is back.
    bool test_pass_outcomes_are_distinguishable()
    {
        using shs::renderpath::PassExecutionResult;
        using shs::renderpath::PassOutcome;

        const PassExecutionResult invalid = PassExecutionResult::invalid_request();
        const PassExecutionResult unmet = PassExecutionResult::prerequisites_unmet();
        const PassExecutionResult declined = PassExecutionResult::declined();
        const PassExecutionResult executed = PassExecutionResult::executed_no_outputs();

        if (invalid.outcome != PassOutcome::InvalidRequest) return false;
        if (unmet.outcome != PassOutcome::PrerequisitesUnmet) return false;
        if (declined.outcome != PassOutcome::Declined) return false;
        if (executed.outcome != PassOutcome::Executed) return false;

        // Pairwise distinct: the collapse the old shape permitted is now
        // unrepresentable, not merely discouraged.
        if (invalid == unmet || invalid == declined || unmet == declined) return false;

        // Only `Executed` reports executed; no refusal claims an output bit.
        if (invalid.executed() || unmet.executed() || declined.executed()) return false;
        if (!executed.executed()) return false;
        if (invalid.produced_depth || unmet.produced_light_grid
            || declined.produced_light_index_list) return false;

        // Same reason builds the same value (the value-equality the kit relies on).
        if (PassExecutionResult::declined() != declined) return false;
        return true;
    }

}

int main()
{
    const bool ok_commands = test_runtime_command_gateway();
    const bool ok_latch = test_input_latch_gateway();
    const bool ok_plan = test_pipeline_execution_plan();
    const bool ok_cmds = test_command_processor_value_reduce();
    const bool ok_request_gate = test_pipeline_uses_execution_request_gate();
    const bool ok_profile_hint = test_profile_config_uses_mode_hints_before_instantiation();
    const bool ok_context_flags = test_execution_plan_ignores_context_runtime_flags();
    const bool ok_resolved_only = test_pipeline_runtime_uses_execute_resolved();
    const bool ok_outcomes = test_pass_outcomes_are_distinguishable();

    if (!ok_commands) std::fprintf(stderr, "[kdba-tests] runtime command gateway failed\n");
    if (!ok_latch) std::fprintf(stderr, "[kdba-tests] runtime input latch gateway failed\n");
    if (!ok_plan) std::fprintf(stderr, "[kdba-tests] pipeline execution plan failed\n");
    if (!ok_cmds) std::fprintf(stderr, "[kdba-tests] command processor value reduce failed\n");
    if (!ok_request_gate) std::fprintf(stderr, "[kdba-tests] pipeline request-gate path failed\n");
    if (!ok_profile_hint) std::fprintf(stderr, "[kdba-tests] profile mode-hint precheck failed\n");
    if (!ok_context_flags) std::fprintf(stderr, "[kdba-tests] context runtime-flag coupling check failed\n");
    if (!ok_resolved_only) std::fprintf(stderr, "[kdba-tests] runtime did not use execute_resolved path\n");
    if (!ok_outcomes) std::fprintf(stderr, "[kdba-tests] pass outcome refusals are not distinguishable\n");

    if (!(ok_commands && ok_latch && ok_plan && ok_cmds && ok_request_gate && ok_profile_hint && ok_context_flags && ok_resolved_only && ok_outcomes)) return 1;
    std::fprintf(stderr, "[kdba-tests] all tests passed\n");
    return 0;
}
