#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: scene_culling.hpp
    МОДУЛЬ: scene
    ЗОРИЛГО: SceneElementSet дээр frustum/occlusion culling гүйцэтгэх
            өндөр төвшний context abstraction.
*/

#if defined(SHS_HAS_JOLT) && ((SHS_HAS_JOLT + 0) == 1)

#include <cstdint>
#include <span>
#include <vector>

#include <glm/glm.hpp>

#include "shs/geometry/culling_runtime.hpp"
#include "shs/geometry/culling_software.hpp"
#include "shs/geometry/culling_visibility.hpp"
#include "shs/geometry/frustum_culling.hpp"
#include "shs/scene/scene_elements.hpp"

namespace shs
{
    class SceneCullingContext
    {
    public:
        explicit SceneCullingContext(
            VisibilityHistoryPolicy history_policy = {})
            : visibility_history_(history_policy)
        {}

        void clear()
        {
            frustum_result_ = CullingResultEx{};
            frustum_visible_indices_.clear();
            visible_indices_.clear();
            stats_ = CullingStats{};
            visibility_history_.clear();
        }

        void set_visibility_history_policy(VisibilityHistoryPolicy policy)
        {
            visibility_history_.set_policy(policy);
        }

        VisibilityHistoryPolicy visibility_history_policy() const
        {
            return visibility_history_.policy();
        }

        const CullingResultEx& frustum_result() const noexcept
        {
            return frustum_result_;
        }

        const CullingStats& stats() const noexcept
        {
            return stats_;
        }

        const std::vector<uint32_t>& frustum_visible_indices() const noexcept
        {
            return frustum_visible_indices_;
        }

        const std::vector<uint32_t>& visible_indices() const noexcept
        {
            return visible_indices_;
        }

        void run_frustum(
            SceneElementSet& scene,
            const Frustum& frustum,
            const CullingRequest& request = {})
        {
            const auto elems_const = scene.elements();
            frustum_result_ = run_frustum_culling(
                std::span<const SceneElement>(elems_const.data(), elems_const.size()),
                frustum,
                [](const SceneElement& e) -> const SceneShape& { return e.geometry; },
                request);
            frustum_visible_indices_.clear();
            frustum_visible_indices_.reserve(elems_const.size());

            std::vector<uint32_t> active_stable_ids{};
            active_stable_ids.reserve(scene.size());

            auto elems = scene.elements();
            for (size_t i = 0; i < elems.size(); ++i)
            {
                SceneElement& e = elems[i];
                active_stable_ids.push_back(e.geometry.stable_id);

                bool frustum_visible =
                    (i < frustum_result_.frustum_classes.size()) &&
                    cull_class_is_visible(
                        frustum_result_.frustum_classes[i],
                        frustum_result_.request.include_intersecting);
                if (!e.enabled) frustum_visible = false;
                e.frustum_visible = frustum_visible;
                e.visible = false;
                if (!frustum_visible)
                {
                    e.occluded = false;
                    visibility_history_.reset(e.geometry.stable_id);
                }
                else
                {
                    frustum_visible_indices_.push_back(static_cast<uint32_t>(i));
                }
            }

            frustum_result_.frustum_visible_indices = frustum_visible_indices_;
            frustum_result_.visible_indices = frustum_visible_indices_;
            stats_ = make_frustum_only_stats(
                static_cast<uint32_t>(scene.size()),
                static_cast<uint32_t>(frustum_visible_indices_.size()));
            frustum_result_.stats = stats_;

            visibility_history_.prune_to_ids(active_stable_ids);
        }

        void apply_occlusion_query_samples(
            SceneElementSet& scene,
            std::span<const uint32_t> query_object_indices,
            std::span<const uint64_t> passed_samples,
            uint64_t min_visible_samples = 1u)
        {
            apply_query_visibility_samples(
                scene.elements(),
                query_object_indices,
                passed_samples,
                min_visible_samples,
                visibility_history_,
                [](const SceneElement& e) -> uint32_t { return e.geometry.stable_id; },
                [](SceneElement& e, bool occluded) { e.occluded = occluded; });
        }

        void finalize_visibility(
            SceneElementSet& scene,
            bool apply_occlusion)
        {
            stats_ = build_visibility_from_frustum(
                scene.elements(),
                std::span<const uint32_t>(frustum_visible_indices_.data(), frustum_visible_indices_.size()),
                apply_occlusion,
                [](const SceneElement& e) -> bool { return e.occluded; },
                [](SceneElement& e, bool visible) { e.visible = visible; },
                visible_indices_);
        }

        bool apply_frustum_fallback_if_needed(
            SceneElementSet& scene,
            bool enable_occlusion,
            bool has_depth_attachment,
            uint32_t query_count)
        {
            const bool fallback = should_use_frustum_visibility_fallback(
                enable_occlusion,
                has_depth_attachment,
                query_count,
                stats_);
            if (!fallback) return false;

            visible_indices_ = frustum_visible_indices_;
            stats_ = make_culling_stats(
                static_cast<uint32_t>(scene.size()),
                static_cast<uint32_t>(frustum_visible_indices_.size()),
                static_cast<uint32_t>(visible_indices_.size()));

            auto elems = scene.elements();
            for (SceneElement& e : elems) e.visible = false;
            for (const uint32_t idx : visible_indices_)
            {
                if (idx >= elems.size()) continue;
                elems[idx].visible = true;
            }

            return true;
        }

        template<typename RasterizeOccluderFn>
        void run_software_occlusion(
            SceneElementSet& scene,
            bool enable_occlusion,
            std::span<float> occlusion_depth,
            int occlusion_width,
            int occlusion_height,
            const glm::mat4& view,
            const glm::mat4& view_proj,
            const RasterizeOccluderFn& rasterize_occluder,
            float depth_epsilon = 1e-4f)
        {
            stats_ = culling_sw::run_software_occlusion_pass(
                scene.elements(),
                std::span<const uint32_t>(frustum_visible_indices_.data(), frustum_visible_indices_.size()),
                enable_occlusion,
                occlusion_depth,
                occlusion_width,
                occlusion_height,
                view,
                view_proj,
                [](const SceneElement& e) -> AABB {
                    return e.geometry.world_aabb();
                },
                [](const SceneElement& e, const glm::mat4& view_mtx) -> float {
                    return culling_sw::view_depth_of_aabb_center(e.geometry.world_aabb(), view_mtx);
                },
                [](SceneElement& e, bool occluded) { e.occluded = occluded; },
                [](SceneElement& e, bool visible) { e.visible = visible; },
                rasterize_occluder,
                visible_indices_,
                depth_epsilon);
        }

    private:
        CullingResultEx frustum_result_{};
        std::vector<uint32_t> frustum_visible_indices_{};
        std::vector<uint32_t> visible_indices_{};
        CullingStats stats_{};
        VisibilityHistory visibility_history_{};
    };
}

#endif // SHS_HAS_JOLT
