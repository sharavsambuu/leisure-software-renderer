#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: pluggable_pipeline.hpp
    МОДУЛЬ: pipeline
    ЗОРИЛГО: Рендерлэх дамжлагыг уян хатан залгаж, салгаж болохуйцаар (pluggable) тодорхойлох суурь бүтэц.
*/


#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <array>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <sstream>
#include <optional>

#include "shs/pipeline/frame_graph.hpp"
#include "shs/pipeline/pass_id.hpp"
#include "shs/pipeline/pass_registry.hpp"
#include "shs/pipeline/render_path_compiler.hpp"
#include "shs/pipeline/render_pass.hpp"
#include "shs/pipeline/technique_profile.hpp"
#include "shs/rhi/sync/vk_runtime.hpp"

namespace shs
{
    struct PipelineExecutionPass
    {
        IRenderPass* pass = nullptr;
        IRenderBackend* backend = nullptr;
        RenderBackendType backend_type = RenderBackendType::Software;
        RHIQueueClass queue = RHIQueueClass::Graphics;
        std::string label{};
    };

    struct PipelineExecutionBackendGroup
    {
        IRenderBackend* backend = nullptr;
        RenderBackendType backend_type = RenderBackendType::Software;
        std::vector<PipelineExecutionPass> passes{};
    };

    struct PipelineExecutionPlan
    {
        std::vector<IRenderPass*> order{};
        std::vector<PipelineExecutionPass> passes{};
        std::vector<PipelineExecutionBackendGroup> backend_groups{};
        FrameGraphReport report{};
        bool valid = true;
    };

    class PipelineRuntimeExecutor
    {
    public:
        void execute(
            Context& ctx,
            const Scene& scene,
            const FrameParams& fp,
            RTRegistry& rtr,
            const PipelineExecutionPlan& plan,
            VulkanLikeRuntime& vk_like_runtime) const
        {
            reset_debug_stats(ctx);

            const RenderBackendFrameInfo backend_frame{
                ++ctx.frame_index,
                fp.w,
                fp.h
            };

            const bool emulate_vk = fp.hybrid.emulate_vulkan_runtime;
            VulkanLikeRuntime& vk_like = vk_like_runtime;
            if (emulate_vk)
            {
                VulkanLikeRuntimeConfig cfg{};
                cfg.frames_in_flight = fp.hybrid.emulated_frames_in_flight;
                cfg.allow_parallel_tasks = fp.hybrid.emulate_parallel_recording;
                vk_like.configure(cfg);
                vk_like.set_job_system(ctx.job_system);
                vk_like.begin_frame(backend_frame.frame_index);
            }

            std::array<uint64_t, 4> queue_timeline_sem{0, 0, 0, 0};
            std::array<uint64_t, 4> queue_timeline_val{0, 0, 0, 0};
            RuntimeCapabilities runtime_caps = initial_runtime_capabilities(fp);
            LightCullingRuntimePayload light_culling_payload{};
            if (emulate_vk)
            {
                for (size_t qi = 0; qi < queue_timeline_sem.size(); ++qi)
                {
                    const auto q = (RHIQueueClass)qi;
                    queue_timeline_sem[qi] = vk_like.queue_timeline_semaphore(q);
                    queue_timeline_val[qi] = vk_like.timeline_value(queue_timeline_sem[qi]);
                }
            }

            IRenderBackend* current_backend = nullptr;
            for (const PipelineExecutionBackendGroup& group : plan.backend_groups)
            {
                IRenderBackend* run_backend = group.backend;
                if (!run_backend) continue;
                if (run_backend != current_backend)
                {
                    if (emulate_vk)
                    {
                        vk_like.execute_all();
                    }
                    if (current_backend) current_backend->end_frame(ctx, backend_frame);
                    run_backend->begin_frame(ctx, backend_frame);
                    current_backend = run_backend;
                }

                for (const PipelineExecutionPass& planned : group.passes)
                {
                    IRenderPass* p = planned.pass;
                    if (!p) continue;
                    const char* id = p->id();
                    auto run_pass = [&ctx, &scene, &fp, &rtr, p, id, &runtime_caps, &light_culling_payload]() {
                        PassExecutionRequest request = p->build_execution_request(ctx, scene, fp, rtr);
                        if (!request.valid) return;
                        request.depth_prepass_ready = runtime_caps.depth_prepass_ready;
                        request.light_culling_ready = runtime_caps.light_culling_ready;
                        request.inputs.light_culling = &light_culling_payload;
                        const auto t0 = std::chrono::steady_clock::now();
                        const PassExecutionResult result = p->execute_resolved(ctx, request);
                        const float ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
                        record_pass_timing(ctx, id, ms);
                        update_runtime_capabilities(result, runtime_caps);
                    };

                    if (emulate_vk)
                    {
                        VulkanLikeSubmission sub{};
                        sub.queue = planned.queue;
                        sub.allow_parallel_tasks = fp.hybrid.emulate_parallel_recording;
                        sub.label = planned.label;
                        sub.tasks.push_back(VulkanLikeTask{sub.label, run_pass});

                        const size_t qi = (size_t)sub.queue;
                        const uint64_t sem = queue_timeline_sem[qi];
                        const uint64_t cur = queue_timeline_val[qi];
                        if (sem != 0 && cur > 0)
                        {
                            sub.waits.push_back(RHISemaphoreWaitDesc{sem, cur, RHIPipelineStage::Top});
                        }
                        if (sem != 0)
                        {
                            sub.signals.push_back(RHISemaphoreSignalDesc{sem, cur + 1, RHIPipelineStage::Bottom});
                        }
                        queue_timeline_val[qi] = cur + 1;
                        vk_like.submit(std::move(sub));
                    }
                    else
                    {
                        run_pass();
                    }
                }
            }

            if (emulate_vk)
            {
                vk_like.execute_all();
                vk_like.end_frame();
                const auto& vks = vk_like.stats();
                ctx.debug.vk_like_submissions = vks.submissions;
                ctx.debug.vk_like_tasks = vks.tasks_executed;
                ctx.debug.vk_like_stalls = vks.stalled_submissions;
            }
            if (current_backend) current_backend->end_frame(ctx, backend_frame);
        }

    private:
        struct RuntimeCapabilities
        {
            bool depth_prepass_ready = false;
            bool light_culling_ready = false;
        };

        static bool technique_uses_light_culling(const FrameParams& fp)
        {
            return
                fp.technique.light_culling ||
                fp.technique.mode == TechniqueMode::ForwardPlus ||
                fp.technique.mode == TechniqueMode::TiledDeferred ||
                fp.technique.mode == TechniqueMode::ClusteredForward;
        }

        static RuntimeCapabilities initial_runtime_capabilities(const FrameParams& fp)
        {
            RuntimeCapabilities out{};
            out.depth_prepass_ready = !fp.technique.depth_prepass;
            out.light_culling_ready = !technique_uses_light_culling(fp);
            return out;
        }

        static void update_runtime_capabilities(const PassExecutionResult& result, RuntimeCapabilities& caps)
        {
            if (!result.executed) return;
            if (result.produced_depth)
            {
                caps.depth_prepass_ready = true;
            }
            if (result.produced_light_grid && result.produced_light_index_list)
            {
                caps.light_culling_ready = true;
            }
        }

        static void reset_debug_stats(Context& ctx)
        {
            ctx.debug.ms_shadow = 0.0f;
            ctx.debug.ms_pbr = 0.0f;
            ctx.debug.ms_tonemap = 0.0f;
            ctx.debug.ms_shafts = 0.0f;
            ctx.debug.ms_motion_blur = 0.0f;
            ctx.debug.vk_like_submissions = 0;
            ctx.debug.vk_like_tasks = 0;
            ctx.debug.vk_like_stalls = 0;
        }

        static void record_pass_timing(Context& ctx, const char* id, float ms)
        {
            const PassId pid = parse_pass_id(id ? id : "");
            if (pid == PassId::ShadowMap) ctx.debug.ms_shadow = ms;
            else if (pid == PassId::PBRForward || pid == PassId::PBRForwardPlus || pid == PassId::PBRForwardClustered) ctx.debug.ms_pbr = ms;
            else if (pid == PassId::Tonemap) ctx.debug.ms_tonemap = ms;
            else if (id && std::string_view(id) == "light_shafts") ctx.debug.ms_shafts = ms;
            else if (pid == PassId::MotionBlur) ctx.debug.ms_motion_blur = ms;
        }
    };

    class PipelineExecutionPlanner
    {
    public:
        PipelineExecutionPlan build(
            Context& ctx,
            const FrameGraph& frame_graph,
            const FrameGraphReport& base_report,
            const std::vector<IRenderPass*>& fallback_order,
            const FrameParams& fp,
            const RTRegistry& rtr,
            bool strict_graph_validation) const
        {
            PipelineExecutionPlan out{};
            out.report = base_report;
            validate_resources(frame_graph, rtr, out.report);
            const bool graph_ok = out.report.valid;
            if (!graph_ok && strict_graph_validation)
            {
                out.valid = out.report.valid;
                return out;
            }

            out.order = out.report.valid ? frame_graph.ordered_passes() : fallback_order;
            validate_pass_contracts(out.order, fp, out.report, strict_graph_validation);
            if (!out.report.valid && strict_graph_validation)
            {
                out.valid = out.report.valid;
                return out;
            }

            IRenderBackend* planned_backend = nullptr;
            for (IRenderPass* p : out.order)
            {
                if (!p || !p->enabled()) continue;
                if (!p->supports_technique_mode(fp.technique.mode))
                {
                    std::ostringstream oss;
                    oss << "Pass '" << (p->id() ? p->id() : "unnamed")
                        << "' does not support technique mode '" << technique_mode_name(fp.technique.mode) << "'.";
                    out.report.warnings.push_back(oss.str());
                    continue;
                }

                const TechniquePassContract contract = p->describe_contract();
                if (is_compatibility_lane_pass(*p, contract))
                {
                    std::ostringstream oss;
                    oss << "Pass '" << (p->id() ? p->id() : "unnamed")
                        << "' is running in compatibility lane: missing explicit contract metadata."
                        << " Provide semantic contract/descriptor registration for planner-visible participation.";
                    (void)push_contract_issue(out.report, oss.str(), true, strict_graph_validation);
                    if (strict_graph_validation) break;
                }
                if (!runtime_contract_requirements_satisfied(*p, contract, out.report, strict_graph_validation))
                {
                    if (strict_graph_validation) break;
                    continue;
                }

                RenderBackendType run_backend_type = RenderBackendType::Software;
                IRenderBackend* run_backend = select_backend_for_pass(ctx, *p, run_backend_type);
                if (!run_backend)
                {
                    std::ostringstream oss;
                    oss << "No available backend for pass '" << (p->id() ? p->id() : "unnamed") << "'.";
                    if (fp.hybrid.strict_backend_availability)
                    {
                        out.report.valid = false;
                        out.report.errors.push_back(oss.str());
                        if (strict_graph_validation) break;
                    }
                    else
                    {
                        out.report.warnings.push_back(oss.str());
                    }
                    continue;
                }

                if (run_backend != planned_backend)
                {
                    if (!fp.hybrid.allow_cross_backend_passes && planned_backend != nullptr)
                    {
                        std::ostringstream oss;
                        oss << "Cross-backend pass switch blocked for pass '" << (p->id() ? p->id() : "unnamed") << "'.";
                        if (fp.hybrid.strict_backend_availability)
                        {
                            out.report.valid = false;
                            out.report.errors.push_back(oss.str());
                            if (strict_graph_validation) break;
                        }
                        else
                        {
                            out.report.warnings.push_back(oss.str());
                        }
                        continue;
                    }
                }

                out.passes.push_back(PipelineExecutionPass{
                    p,
                    run_backend,
                    run_backend_type,
                    p->preferred_queue(),
                    pass_id_is_standard(parse_pass_id(p->id() ? p->id() : ""))
                        ? pass_id_name(parse_pass_id(p->id() ? p->id() : ""))
                        : (p->id() ? p->id() : "unnamed")
                });
                planned_backend = run_backend;
            }

            for (const PipelineExecutionPass& p : out.passes)
            {
                if (out.backend_groups.empty() || out.backend_groups.back().backend != p.backend)
                {
                    PipelineExecutionBackendGroup g{};
                    g.backend = p.backend;
                    g.backend_type = p.backend_type;
                    out.backend_groups.push_back(std::move(g));
                }
                out.backend_groups.back().passes.push_back(p);
            }

            out.valid = out.report.valid;
            return out;
        }

    private:
        static bool resource_type_matches(PassResourceType expected, RTKind actual)
        {
            switch (expected)
            {
                case PassResourceType::Unknown: return true;
                case PassResourceType::Temp: return true;
                case PassResourceType::Shadow: return actual == RTKind::Shadow;
                case PassResourceType::ColorHDR: return actual == RTKind::ColorHDR;
                case PassResourceType::ColorLDR: return actual == RTKind::ColorLDR;
                case PassResourceType::Motion: return actual == RTKind::Motion;
            }
            return false;
        }

        static void validate_resources(const FrameGraph& frame_graph, const RTRegistry& rtr, FrameGraphReport& report)
        {
            std::unordered_map<uint64_t, std::vector<std::string>> writers{};
            std::unordered_map<uint64_t, RTRegistry::Extent> first_extent{};
            for (const auto& node : frame_graph.nodes())
            {
                RTRegistry::Extent pass_extent{};
                bool pass_extent_set = false;
                for (const auto& res : node.io.resources)
                {
                    if (res.key == 0) continue;
                    if (node.pass && !pass_resource_domain_matches_backend(res.domain, node.pass->preferred_backend()))
                    {
                        std::ostringstream oss;
                        oss << "Resource domain '" << pass_resource_domain_name(res.domain) << "' may not match pass backend '"
                            << render_backend_type_name(node.pass->preferred_backend()) << "' in pass '" << node.pass_id << "'.";
                        report.warnings.push_back(oss.str());
                    }
                    if (pass_resource_key_is_named(res.key))
                    {
                        continue;
                    }

                    const RTHandle h{pass_rt_id_from_key(res.key)};
                    if (!h.valid() || !rtr.has(h))
                    {
                        report.valid = false;
                        std::ostringstream oss;
                        oss << "Missing RT for pass '" << node.pass_id << "' resource '" << (res.name.empty() ? "unnamed" : res.name) << "'.";
                        report.errors.push_back(oss.str());
                        continue;
                    }

                    const RTKind k = rtr.kind(h);
                    if (!resource_type_matches(res.type, k))
                    {
                        report.valid = false;
                        std::ostringstream oss;
                        oss << "RT type mismatch in pass '" << node.pass_id << "' resource '" << (res.name.empty() ? "unnamed" : res.name) << "'.";
                        report.errors.push_back(oss.str());
                    }

                    if (pass_access_has_write(res.access))
                    {
                        writers[res.key].push_back(node.pass_id);
                    }

                    const RTRegistry::Extent ex = rtr.extent(h);
                    if (ex.valid() && res.type != PassResourceType::Shadow)
                    {
                        if (!pass_extent_set)
                        {
                            pass_extent = ex;
                            pass_extent_set = true;
                        }
                        else if (pass_extent.w != ex.w || pass_extent.h != ex.h)
                        {
                            std::ostringstream oss;
                            oss << "Extent mismatch in pass '" << node.pass_id << "' near resource '" << (res.name.empty() ? "unnamed" : res.name) << "'.";
                            report.warnings.push_back(oss.str());
                        }
                    }

                    auto it_first = first_extent.find(res.key);
                    if (it_first == first_extent.end())
                    {
                        first_extent.emplace(res.key, ex);
                    }
                    else if (ex.valid() && it_first->second.valid() && (it_first->second.w != ex.w || it_first->second.h != ex.h))
                    {
                        std::ostringstream oss;
                        oss << "Global extent mismatch for resource '" << (res.name.empty() ? "unnamed" : res.name) << "'.";
                        report.warnings.push_back(oss.str());
                    }
                }
            }

            for (const auto& kv : writers)
            {
                if (kv.second.size() <= 1) continue;
                std::ostringstream oss;
                oss << "Multiple writers detected for one resource: ";
                for (size_t i = 0; i < kv.second.size(); ++i)
                {
                    if (i) oss << ", ";
                    oss << "'" << kv.second[i] << "'";
                }
                report.warnings.push_back(oss.str());
            }
        }

        struct PassSemanticHash
        {
            size_t operator()(PassSemantic s) const
            {
                return (size_t)s;
            }
        };

        static bool push_contract_issue(FrameGraphReport& report, const std::string& msg, bool severe, bool strict_graph_validation)
        {
            if (severe && strict_graph_validation)
            {
                report.valid = false;
                report.errors.push_back(msg);
                return false;
            }
            report.warnings.push_back(msg);
            return true;
        }

        static bool contract_reads_semantic(const PassSemanticRef& ref)
        {
            return ref.semantic != PassSemantic::Unknown && contract_access_has_read(ref.access);
        }

        static bool contract_writes_semantic(const PassSemanticRef& ref)
        {
            return ref.semantic != PassSemantic::Unknown && contract_access_has_write(ref.access);
        }

        static bool technique_mode_enabled_in_active_mask(const FrameParams& fp)
        {
            return technique_mode_in_mask(fp.technique.active_modes_mask, fp.technique.mode);
        }

        static void validate_pass_contracts(
            const std::vector<IRenderPass*>& order,
            const FrameParams& fp,
            FrameGraphReport& report,
            bool strict_graph_validation)
        {
            const bool mode_allowed = technique_mode_enabled_in_active_mask(fp);
            if (!mode_allowed)
            {
                std::ostringstream oss;
                oss << "Active technique mask excludes current mode '" << technique_mode_name(fp.technique.mode)
                    << "' (mask=0x" << std::hex << fp.technique.active_modes_mask << std::dec << ").";
                (void)push_contract_issue(report, oss.str(), true, strict_graph_validation);
            }

            std::unordered_map<PassSemantic, PassSemanticRef, PassSemanticHash> produced_semantics{};
            const bool light_culling_enabled =
                fp.technique.light_culling ||
                fp.technique.mode == TechniqueMode::ForwardPlus ||
                fp.technique.mode == TechniqueMode::TiledDeferred ||
                fp.technique.mode == TechniqueMode::ClusteredForward;

            for (const IRenderPass* p : order)
            {
                if (!p || !p->enabled()) continue;
                const char* id = p->id() ? p->id() : "unnamed";
                const TechniquePassContract contract = p->describe_contract();

                if (!technique_mode_in_mask(contract.supported_modes_mask, fp.technique.mode))
                {
                    std::ostringstream oss;
                    oss << "Pass '" << id << "' contract excludes current mode '" << technique_mode_name(fp.technique.mode) << "'.";
                    (void)push_contract_issue(report, oss.str(), true, strict_graph_validation);
                }

                if (contract.requires_depth_prepass && !fp.technique.depth_prepass)
                {
                    std::ostringstream oss;
                    oss << "Pass '" << id << "' requires depth prepass but technique.depth_prepass is disabled.";
                    (void)push_contract_issue(report, oss.str(), true, strict_graph_validation);
                }

                if (contract.requires_light_culling && !light_culling_enabled)
                {
                    std::ostringstream oss;
                    oss << "Pass '" << id << "' requires light culling but technique.light_culling is disabled for this mode.";
                    (void)push_contract_issue(report, oss.str(), true, strict_graph_validation);
                }

                if (contract.requires_depth_prepass && produced_semantics.find(PassSemantic::Depth) == produced_semantics.end())
                {
                    std::ostringstream oss;
                    oss << "Pass '" << id << "' requires depth prepass semantics but no earlier pass produced '"
                        << pass_semantic_name(PassSemantic::Depth) << "'.";
                    (void)push_contract_issue(report, oss.str(), true, strict_graph_validation);
                }

                if (contract.requires_light_culling)
                {
                    const bool has_grid = produced_semantics.find(PassSemantic::LightGrid) != produced_semantics.end();
                    const bool has_list = produced_semantics.find(PassSemantic::LightIndexList) != produced_semantics.end();
                    if (!has_grid || !has_list)
                    {
                        std::ostringstream oss;
                        oss << "Pass '" << id << "' requires light-culling semantics (light_grid + light_index_list), but prior producers are missing.";
                        (void)push_contract_issue(report, oss.str(), true, strict_graph_validation);
                    }
                }

                if (contract.prefer_async_compute && p->preferred_queue() != RHIQueueClass::Compute)
                {
                    std::ostringstream oss;
                    oss << "Pass '" << id << "' prefers async compute but preferred queue is not compute.";
                    (void)push_contract_issue(report, oss.str(), false, strict_graph_validation);
                }

                for (const PassSemanticRef& sref : contract.semantics)
                {
                    if (!contract_reads_semantic(sref)) continue;

                    const auto it_produced = produced_semantics.find(sref.semantic);
                    if (it_produced == produced_semantics.end())
                    {
                        std::ostringstream oss;
                        oss << "Pass '" << id << "' reads semantic '" << pass_semantic_name(sref.semantic)
                            << "' without an earlier producer in this pipeline.";
                        if (!sref.alias.empty()) oss << " Alias='" << sref.alias << "'.";
                        (void)push_contract_issue(report, oss.str(), false, strict_graph_validation);
                        continue;
                    }

                    const PassSemanticRef& prod = it_produced->second;
                    if (prod.space != sref.space || prod.encoding != sref.encoding)
                    {
                        std::ostringstream oss;
                        oss << "Pass '" << id << "' reads semantic '" << pass_semantic_name(sref.semantic)
                            << "' with representation mismatch. Produced("
                            << pass_semantic_space_name(prod.space) << ", "
                            << pass_semantic_encoding_name(prod.encoding) << ") vs Read("
                            << pass_semantic_space_name(sref.space) << ", "
                            << pass_semantic_encoding_name(sref.encoding) << ").";
                        (void)push_contract_issue(report, oss.str(), true, strict_graph_validation);
                    }
                    if (prod.lifetime != sref.lifetime)
                    {
                        std::ostringstream oss;
                        oss << "Pass '" << id << "' reads semantic '" << pass_semantic_name(sref.semantic)
                            << "' with lifetime mismatch. Produced("
                            << pass_semantic_lifetime_name(prod.lifetime) << ") vs Read("
                            << pass_semantic_lifetime_name(sref.lifetime) << ").";
                        (void)push_contract_issue(report, oss.str(), true, strict_graph_validation);
                    }
                }

                for (const PassSemanticRef& sref : contract.semantics)
                {
                    if (contract_writes_semantic(sref))
                    {
                        produced_semantics[sref.semantic] = sref;
                    }
                }
            }
        }

        static bool runtime_contract_requirements_satisfied(
            const IRenderPass& pass,
            const TechniquePassContract& contract,
            FrameGraphReport& report,
            bool strict_graph_validation
        )
        {
            if (contract.prefer_async_compute && pass.preferred_queue() != RHIQueueClass::Compute)
            {
                const char* id = pass.id() ? pass.id() : "unnamed";
                std::ostringstream oss;
                oss << "Pass '" << id << "' prefers async compute but does not target compute queue.";
                (void)push_contract_issue(report, oss.str(), false, strict_graph_validation);
            }

            return true;
        }

        static bool contract_has_explicit_metadata(const TechniquePassContract& contract)
        {
            return
                contract.role != TechniquePassRole::Custom ||
                !contract.semantics.empty() ||
                contract.requires_depth_prepass ||
                contract.requires_light_culling ||
                contract.prefer_async_compute;
        }

        static bool is_compatibility_lane_pass(const IRenderPass& pass, const TechniquePassContract& contract)
        {
            const PassId pass_id = parse_pass_id(pass.id() ? pass.id() : "");
            if (pass_id_is_standard(pass_id)) return false;
            return !contract_has_explicit_metadata(contract);
        }

        static constexpr std::array<RenderBackendType, 3> all_backend_types()
        {
            return {
                RenderBackendType::Software,
                RenderBackendType::OpenGL,
                RenderBackendType::Vulkan
            };
        }

        static IRenderBackend* try_backend(Context& ctx, const IRenderPass& pass, RenderBackendType t)
        {
            if (!pass.supports_backend(t)) return nullptr;
            return ctx.backend(t);
        }

        static IRenderBackend* select_backend_for_pass(Context& ctx, const IRenderPass& pass, RenderBackendType& out_type)
        {
            const RenderBackendType preferred = pass.preferred_backend();
            if (IRenderBackend* b = try_backend(ctx, pass, preferred))
            {
                out_type = preferred;
                return b;
            }

            const RenderBackendType active = ctx.active_backend_type();
            if (IRenderBackend* b = try_backend(ctx, pass, active))
            {
                out_type = active;
                return b;
            }

            for (const RenderBackendType t : all_backend_types())
            {
                if (IRenderBackend* b = try_backend(ctx, pass, t))
                {
                    out_type = t;
                    return b;
                }
            }
            return nullptr;
        }
    };

    class PipelineResizeCoordinator
    {
    public:
        void dispatch_if_needed(Context& ctx, RTRegistry& rtr, const std::vector<std::unique_ptr<IRenderPass>>& passes, int w, int h)
        {
            if (w <= 0 || h <= 0) return;
            if (last_w_ == w && last_h_ == h && has_size_) return;
            for (const RenderBackendType t : all_backend_types())
            {
                if (auto* b = ctx.backend(t)) b->on_resize(ctx, w, h);
            }
            for (const auto& p : passes)
            {
                if (p) p->on_resize(ctx, rtr, w, h);
            }
            last_w_ = w;
            last_h_ = h;
            has_size_ = true;
        }

    private:
        static constexpr std::array<RenderBackendType, 3> all_backend_types()
        {
            return {
                RenderBackendType::Software,
                RenderBackendType::OpenGL,
                RenderBackendType::Vulkan
            };
        }

        int last_w_ = 0;
        int last_h_ = 0;
        bool has_size_ = false;
    };

    class PluggablePipeline
    {
    public:
        template<typename TPass, typename... Args>
        TPass& add_pass(Args&&... args)
        {
            auto p = std::make_unique<TPass>(std::forward<Args>(args)...);
            TPass& ref = *p;
            passes_.push_back(std::move(p));
            graph_dirty_ = true;
            return ref;
        }

        IRenderPass* add_pass_instance(std::unique_ptr<IRenderPass> pass)
        {
            if (!pass) return nullptr;
            IRenderPass* ref = pass.get();
            passes_.push_back(std::move(pass));
            graph_dirty_ = true;
            return ref;
        }

        bool add_pass_from_registry(const PassFactoryRegistry& registry, const std::string& id)
        {
            auto pass = registry.create(id);
            if (!pass) return false;
            add_pass_instance(std::move(pass));
            return true;
        }

        bool add_pass_from_registry(const PassFactoryRegistry& registry, PassId id)
        {
            if (!pass_id_is_standard(id)) return false;
            auto pass = registry.create(id);
            if (!pass) return false;
            add_pass_instance(std::move(pass));
            return true;
        }

        bool configure_for_technique(const PassFactoryRegistry& registry, TechniqueMode mode, std::vector<std::string>* out_missing_ids = nullptr)
        {
            const TechniqueProfile profile = make_default_technique_profile(mode);
            return configure_from_profile(registry, profile, out_missing_ids);
        }

        bool configure_from_profile(const PassFactoryRegistry& registry, const TechniqueProfile& profile, std::vector<std::string>* out_missing_ids = nullptr)
        {
            passes_.clear();
            graph_dirty_ = true;
            if (out_missing_ids) out_missing_ids->clear();

            bool ok = true;
            for (const auto& e : profile.passes)
            {
                const PassId pass_id = pass_id_is_standard(e.pass_id) ? e.pass_id : parse_pass_id(e.id);
                const std::string missing_id = pass_id_is_standard(pass_id) ? pass_id_string(pass_id) : e.id;
                const std::optional<bool> mode_hint = pass_id_is_standard(pass_id)
                    ? registry.supports_technique_mode_hint(pass_id, profile.mode)
                    : registry.supports_technique_mode_hint(e.id, profile.mode);
                if (mode_hint.has_value() && !mode_hint.value())
                {
                    if (e.required) ok = false;
                    if (out_missing_ids) out_missing_ids->push_back(missing_id);
                    continue;
                }
                auto p = pass_id_is_standard(pass_id) ? registry.create(pass_id) : registry.create(e.id);
                if (!p)
                {
                    if (e.required) ok = false;
                    if (out_missing_ids) out_missing_ids->push_back(missing_id);
                    continue;
                }
                if (!p->supports_technique_mode(profile.mode))
                {
                    if (e.required) ok = false;
                    if (out_missing_ids) out_missing_ids->push_back(missing_id);
                    continue;
                }
                add_pass_instance(std::move(p));
            }
            return ok;
        }

        bool configure_from_render_path_plan(
            const PassFactoryRegistry& registry,
            const RenderPathExecutionPlan& plan,
            std::vector<std::string>* out_missing_ids = nullptr)
        {
            passes_.clear();
            graph_dirty_ = true;
            if (out_missing_ids) out_missing_ids->clear();

            bool ok = plan.valid;
            for (const auto& entry : plan.pass_chain)
            {
                const PassId pass_id = pass_id_is_standard(entry.pass_id) ? entry.pass_id : parse_pass_id(entry.id);
                const std::string missing_id = pass_id_is_standard(pass_id) ? pass_id_string(pass_id) : entry.id;
                const std::optional<bool> mode_hint = pass_id_is_standard(pass_id)
                    ? registry.supports_technique_mode_hint(pass_id, plan.technique_mode)
                    : registry.supports_technique_mode_hint(entry.id, plan.technique_mode);
                if (mode_hint.has_value() && !mode_hint.value())
                {
                    if (entry.required) ok = false;
                    if (out_missing_ids) out_missing_ids->push_back(missing_id);
                    continue;
                }
                auto p = pass_id_is_standard(pass_id) ? registry.create(pass_id) : registry.create(entry.id);
                if (!p)
                {
                    if (entry.required) ok = false;
                    if (out_missing_ids) out_missing_ids->push_back(missing_id);
                    continue;
                }
                if (!p->supports_technique_mode(plan.technique_mode))
                {
                    if (entry.required) ok = false;
                    if (out_missing_ids) out_missing_ids->push_back(missing_id);
                    continue;
                }
                add_pass_instance(std::move(p));
            }
            return ok;
        }

        bool configure_from_render_path_recipe(
            const PassFactoryRegistry& registry,
            const RenderPathCompiler& compiler,
            const RenderPathRecipe& recipe,
            const RenderPathCapabilitySet& capabilities,
            RenderPathExecutionPlan* out_plan = nullptr,
            std::vector<std::string>* out_missing_ids = nullptr)
        {
            const RenderPathExecutionPlan plan = compiler.compile(recipe, capabilities, &registry);
            if (out_plan) *out_plan = plan;
            const bool configured = configure_from_render_path_plan(registry, plan, out_missing_ids);
            return configured && plan.valid;
        }

        IRenderPass* find(const std::string& pass_id)
        {
            for (auto& p : passes_)
            {
                if (pass_id == p->id()) return p.get();
            }
            return nullptr;
        }

        const IRenderPass* find(const std::string& pass_id) const
        {
            for (const auto& p : passes_)
            {
                if (pass_id == p->id()) return p.get();
            }
            return nullptr;
        }

        IRenderPass* find(PassId pass_id)
        {
            if (!pass_id_is_standard(pass_id)) return nullptr;
            for (auto& p : passes_)
            {
                if (!p || !p->id()) continue;
                if (parse_pass_id(p->id()) == pass_id) return p.get();
            }
            return nullptr;
        }

        const IRenderPass* find(PassId pass_id) const
        {
            if (!pass_id_is_standard(pass_id)) return nullptr;
            for (const auto& p : passes_)
            {
                if (!p || !p->id()) continue;
                if (parse_pass_id(p->id()) == pass_id) return p.get();
            }
            return nullptr;
        }

        bool set_enabled(const std::string& pass_id, bool enabled)
        {
            if (auto* p = find(pass_id))
            {
                p->set_enabled(enabled);
                graph_dirty_ = true;
                return true;
            }
            return false;
        }

        bool set_enabled(PassId pass_id, bool enabled)
        {
            if (auto* p = find(pass_id))
            {
                p->set_enabled(enabled);
                graph_dirty_ = true;
                return true;
            }
            return false;
        }

        const FrameGraphReport& graph_report() const { return graph_report_; }
        const FrameGraphReport& execution_report() const { return execution_report_; }
        VulkanLikeRuntime& vulkan_like_runtime() { return vk_like_runtime_; }
        const VulkanLikeRuntime& vulkan_like_runtime() const { return vk_like_runtime_; }
        void set_strict_graph_validation(bool v) { strict_graph_validation_ = v; }
        PipelineExecutionPlan build_execution_plan(Context& ctx, const FrameParams& fp, RTRegistry& rtr)
        {
            rebuild_graph_if_needed();
            return planner_.build(
                ctx,
                frame_graph_,
                graph_report_,
                linear_enabled_passes(),
                fp,
                rtr,
                strict_graph_validation_);
        }
        void on_scene_reset(Context& ctx, RTRegistry& rtr)
        {
            for (auto& p : passes_)
            {
                if (p) p->on_scene_reset(ctx, rtr);
            }
            ctx.shadow.reset_caches();
            reset_history(ctx, rtr);
        }

        void reset_history(Context& ctx, RTRegistry& rtr)
        {
            ctx.history.reset();
            ctx.temporal_aa.reset();
            for (auto& p : passes_)
            {
                if (p) p->reset_history(ctx, rtr);
            }
        }

        void execute(Context& ctx, const Scene& scene, const FrameParams& fp, RTRegistry& rtr)
        {
            const FrameParams& fp_eval = fp;

            const PipelineExecutionPlan plan = build_execution_plan(ctx, fp_eval, rtr);
            execution_report_ = plan.report;
            if (!execution_report_.valid && strict_graph_validation_) return;
            resize_coordinator_.dispatch_if_needed(ctx, rtr, passes_, fp_eval.w, fp_eval.h);
            runtime_executor_.execute(ctx, scene, fp_eval, rtr, plan, vk_like_runtime_);
        }

    private:
        void rebuild_graph_if_needed()
        {
            if (!graph_dirty_) return;
            frame_graph_.clear();

            for (size_t i = 0; i < passes_.size(); ++i)
            {
                IRenderPass* p = passes_[i].get();
                if (!p || !p->enabled()) continue;
                FrameGraphNode n{};
                n.pass = p;
                n.pass_id = p->id() ? p->id() : "";
                n.io = p->describe_io();
                n.original_index = i;
                frame_graph_.add_node(std::move(n));
            }

            frame_graph_.compile();
            graph_report_ = frame_graph_.report();
            graph_dirty_ = false;
        }

        std::vector<IRenderPass*> linear_enabled_passes() const
        {
            std::vector<IRenderPass*> out{};
            out.reserve(passes_.size());
            for (const auto& p : passes_)
            {
                if (p && p->enabled()) out.push_back(p.get());
            }
            return out;
        }

        std::vector<std::unique_ptr<IRenderPass>> passes_{};
        FrameGraph frame_graph_{};
        FrameGraphReport graph_report_{};
        FrameGraphReport execution_report_{};
        bool graph_dirty_ = true;
        bool strict_graph_validation_ = true;
        PipelineExecutionPlanner planner_{};
        PipelineResizeCoordinator resize_coordinator_{};
        PipelineRuntimeExecutor runtime_executor_{};
        VulkanLikeRuntime vk_like_runtime_{};
    };
}
