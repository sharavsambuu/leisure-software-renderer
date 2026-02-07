#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: pluggable_pipeline.hpp
    МОДУЛЬ: pipeline
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн pipeline модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <array>
#include <unordered_map>
#include <chrono>
#include <cstring>
#include <sstream>

#include "shs/pipeline/frame_graph.hpp"
#include "shs/pipeline/pass_registry.hpp"
#include "shs/pipeline/render_pass.hpp"
#include "shs/pipeline/technique_profile.hpp"

namespace shs
{
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
                auto p = registry.create(e.id);
                if (!p)
                {
                    if (e.required) ok = false;
                    if (out_missing_ids) out_missing_ids->push_back(e.id);
                    continue;
                }
                if (!p->supports_technique_mode(profile.mode))
                {
                    if (e.required) ok = false;
                    if (out_missing_ids) out_missing_ids->push_back(e.id);
                    continue;
                }
                add_pass_instance(std::move(p));
            }
            return ok;
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

        const FrameGraphReport& graph_report() const { return graph_report_; }
        const FrameGraphReport& execution_report() const { return execution_report_; }
        void set_strict_graph_validation(bool v) { strict_graph_validation_ = v; }
        void on_scene_reset(Context& ctx, RTRegistry& rtr)
        {
            for (auto& p : passes_)
            {
                if (p) p->on_scene_reset(ctx, rtr);
            }
            reset_history(ctx, rtr);
        }

        void reset_history(Context& ctx, RTRegistry& rtr)
        {
            ctx.history.reset();
            for (auto& p : passes_)
            {
                if (p) p->reset_history(ctx, rtr);
            }
        }

        void execute(Context& ctx, const Scene& scene, const FrameParams& fp, RTRegistry& rtr)
        {
            const FrameParams& fp_eval = fp;

            ctx.debug.ms_shadow = 0.0f;
            ctx.debug.ms_pbr = 0.0f;
            ctx.debug.ms_tonemap = 0.0f;
            ctx.debug.ms_shafts = 0.0f;
            ctx.debug.ms_motion_blur = 0.0f;

            rebuild_graph_if_needed();
            execution_report_ = graph_report_;
            validate_resources(rtr, execution_report_);
            const bool graph_ok = execution_report_.valid;
            if (!graph_ok && strict_graph_validation_) return;

            const std::vector<IRenderPass*> order = execution_report_.valid ? frame_graph_.ordered_passes() : linear_enabled_passes();
            dispatch_resize_if_needed(ctx, rtr, fp_eval.w, fp_eval.h);
            const RenderBackendFrameInfo backend_frame{
                ++ctx.frame_index,
                fp_eval.w,
                fp_eval.h
            };
            const bool emulate_vk = fp_eval.hybrid.emulate_vulkan_runtime;
            if (emulate_vk)
            {
                VulkanLikeRuntimeConfig cfg{};
                cfg.frames_in_flight = fp_eval.hybrid.emulated_frames_in_flight;
                cfg.allow_parallel_tasks = fp_eval.hybrid.emulate_parallel_recording;
                ctx.vk_like.configure(cfg);
                ctx.vk_like.set_job_system(ctx.job_system);
                ctx.vk_like.begin_frame(backend_frame.frame_index);
            }

            std::array<uint64_t, 4> queue_timeline_sem{0, 0, 0, 0};
            std::array<uint64_t, 4> queue_timeline_val{0, 0, 0, 0};
            if (emulate_vk)
            {
                for (size_t qi = 0; qi < queue_timeline_sem.size(); ++qi)
                {
                    const auto q = (RHIQueueClass)qi;
                    queue_timeline_sem[qi] = ctx.vk_like.queue_timeline_semaphore(q);
                    queue_timeline_val[qi] = ctx.vk_like.timeline_value(queue_timeline_sem[qi]);
                }
            }

            IRenderBackend* current_backend = nullptr;
            for (IRenderPass* p : order)
            {
                if (!p || !p->enabled()) continue;
                if (!p->supports_technique_mode(fp_eval.technique.mode))
                {
                    std::ostringstream oss;
                    oss << "Pass '" << (p->id() ? p->id() : "unnamed")
                        << "' does not support technique mode '" << technique_mode_name(fp_eval.technique.mode) << "'.";
                    execution_report_.warnings.push_back(oss.str());
                    continue;
                }
                RenderBackendType run_backend_type = RenderBackendType::Software;
                IRenderBackend* run_backend = select_backend_for_pass(ctx, *p, run_backend_type);
                if (!run_backend)
                {
                    std::ostringstream oss;
                    oss << "No available backend for pass '" << (p->id() ? p->id() : "unnamed") << "'.";
                    if (fp_eval.hybrid.strict_backend_availability)
                    {
                        execution_report_.valid = false;
                        execution_report_.errors.push_back(oss.str());
                        if (strict_graph_validation_) break;
                    }
                    else
                    {
                        execution_report_.warnings.push_back(oss.str());
                    }
                    continue;
                }

                if (run_backend != current_backend)
                {
                    if (!fp_eval.hybrid.allow_cross_backend_passes && current_backend != nullptr)
                    {
                        std::ostringstream oss;
                        oss << "Cross-backend pass switch blocked for pass '" << (p->id() ? p->id() : "unnamed") << "'.";
                        if (fp_eval.hybrid.strict_backend_availability)
                        {
                            execution_report_.valid = false;
                            execution_report_.errors.push_back(oss.str());
                            if (strict_graph_validation_) break;
                        }
                        else
                        {
                            execution_report_.warnings.push_back(oss.str());
                        }
                        continue;
                    }

                    if (emulate_vk)
                    {
                        ctx.vk_like.execute_all();
                    }
                    if (current_backend) current_backend->end_frame(ctx, backend_frame);
                    run_backend->begin_frame(ctx, backend_frame);
                    current_backend = run_backend;
                }

                const char* id = p->id();
                auto run_pass = [&ctx, &scene, &fp_eval, &rtr, p, id]() {
                    const auto t0 = std::chrono::steady_clock::now();
                    p->execute(ctx, scene, fp_eval, rtr);
                    const float ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
                    if (!id) return;
                    if (std::strcmp(id, "shadow_map") == 0) ctx.debug.ms_shadow = ms;
                    else if (std::strcmp(id, "pbr_forward") == 0 || std::strcmp(id, "pbr_forward_plus") == 0) ctx.debug.ms_pbr = ms;
                    else if (std::strcmp(id, "tonemap") == 0) ctx.debug.ms_tonemap = ms;
                    else if (std::strcmp(id, "light_shafts") == 0) ctx.debug.ms_shafts = ms;
                    else if (std::strcmp(id, "motion_blur") == 0) ctx.debug.ms_motion_blur = ms;
                };

                if (emulate_vk)
                {
                    VulkanLikeSubmission sub{};
                    sub.queue = p->preferred_queue();
                    sub.allow_parallel_tasks = fp_eval.hybrid.emulate_parallel_recording;
                    sub.label = id ? id : "unnamed";
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
                    ctx.vk_like.submit(std::move(sub));
                }
                else
                {
                    run_pass();
                }
                (void)run_backend_type;
            }

            if (emulate_vk)
            {
                ctx.vk_like.execute_all();
                ctx.vk_like.end_frame();
                const auto& vks = ctx.vk_like.stats();
                ctx.debug.vk_like_submissions = vks.submissions;
                ctx.debug.vk_like_tasks = vks.tasks_executed;
                ctx.debug.vk_like_stalls = vks.stalled_submissions;
            }
            if (current_backend) current_backend->end_frame(ctx, backend_frame);
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

        void validate_resources(const RTRegistry& rtr, FrameGraphReport& report) const
        {
            std::unordered_map<uint64_t, std::vector<std::string>> writers{};
            std::unordered_map<uint64_t, RTRegistry::Extent> first_extent{};
            for (const auto& node : frame_graph_.nodes())
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
                        // Named transient нь runtime allocator-аар шийдэгдэнэ.
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

        void dispatch_resize_if_needed(Context& ctx, RTRegistry& rtr, int w, int h)
        {
            if (w <= 0 || h <= 0) return;
            if (last_w_ == w && last_h_ == h && has_size_) return;
            for (const RenderBackendType t : all_backend_types())
            {
                if (auto* b = ctx.backend(t)) b->on_resize(ctx, w, h);
            }
            for (auto& p : passes_)
            {
                if (p) p->on_resize(ctx, rtr, w, h);
            }
            last_w_ = w;
            last_h_ = h;
            has_size_ = true;
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

        std::vector<std::unique_ptr<IRenderPass>> passes_{};
        FrameGraph frame_graph_{};
        FrameGraphReport graph_report_{};
        FrameGraphReport execution_report_{};
        bool graph_dirty_ = true;
        bool strict_graph_validation_ = false;
        int last_w_ = 0;
        int last_h_ = 0;
        bool has_size_ = false;
    };
}
