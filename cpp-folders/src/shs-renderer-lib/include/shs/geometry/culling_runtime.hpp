#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: culling_runtime.hpp
    МОДУЛЬ: geometry
    ЗОРИЛГО: Frustum/Occlusion culling-д зориулсан runtime abstraction layer.
            Нэгэн жигд request/result/stats төрөл, мөн stable_id дээр
            суурилсан visibility history (hysteresis) хадгална.
*/

#if defined(SHS_HAS_JOLT) && ((SHS_HAS_JOLT + 0) == 1)

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "shs/geometry/jolt_culling.hpp"

namespace shs
{
    enum class CullingPassKind : uint8_t
    {
        Frustum = 0,
        Occlusion = 1,
        FrustumAndOcclusion = 2
    };

    struct CullingRequest
    {
        CullTolerance tolerance{};
        bool include_intersecting = true;
    };

    struct CullingStats
    {
        uint32_t scene_count = 0;
        uint32_t frustum_visible_count = 0;
        uint32_t occluded_count = 0;
        uint32_t visible_count = 0;
        uint32_t culled_count = 0;
    };

    struct CullingResultEx
    {
        CullingPassKind pass = CullingPassKind::Frustum;
        CullingRequest request{};
        std::vector<CullClass> frustum_classes{};
        std::vector<uint32_t> frustum_visible_indices{};
        std::vector<uint32_t> visible_indices{};
        CullingStats stats{};
    };

    inline void normalize_culling_stats(CullingStats& stats) noexcept
    {
        if (stats.frustum_visible_count < stats.visible_count)
        {
            stats.frustum_visible_count = stats.visible_count;
        }

        stats.occluded_count =
            (stats.frustum_visible_count >= stats.visible_count)
                ? (stats.frustum_visible_count - stats.visible_count)
                : 0u;

        stats.culled_count =
            (stats.scene_count >= stats.visible_count)
                ? (stats.scene_count - stats.visible_count)
                : 0u;
    }

    inline CullingStats make_frustum_only_stats(
        uint32_t scene_count,
        uint32_t visible_count) noexcept
    {
        CullingStats stats{};
        stats.scene_count = scene_count;
        stats.frustum_visible_count = visible_count;
        stats.visible_count = visible_count;
        normalize_culling_stats(stats);
        return stats;
    }

    inline CullingStats make_culling_stats(
        uint32_t scene_count,
        uint32_t frustum_visible_count,
        uint32_t visible_count) noexcept
    {
        CullingStats stats{};
        stats.scene_count = scene_count;
        stats.frustum_visible_count = frustum_visible_count;
        stats.visible_count = visible_count;
        normalize_culling_stats(stats);
        return stats;
    }

    template<typename TObject, typename GetCullableFn>
    requires requires(
        const TObject& object,
        const GetCullableFn& get_cullable,
        const Frustum& frustum,
        const CullTolerance& tol)
    {
        { classify_vs_frustum(get_cullable(object), frustum, tol) } -> std::same_as<CullClass>;
    }
    inline CullingResultEx run_frustum_culling(
        std::span<const TObject> objects,
        const Frustum& frustum,
        const GetCullableFn& get_cullable,
        const CullingRequest& request = {})
    {
        CullingResultEx out{};
        out.pass = CullingPassKind::Frustum;
        out.request = request;

        const size_t n = objects.size();
        out.frustum_classes.resize(n, CullClass::Outside);
        out.frustum_visible_indices.reserve(n);
        out.visible_indices.reserve(n);

        for (size_t i = 0; i < n; ++i)
        {
            const CullClass cls = classify_vs_frustum(get_cullable(objects[i]), frustum, request.tolerance);
            out.frustum_classes[i] = cls;
            if (cull_class_is_visible(cls, request.include_intersecting))
            {
                const uint32_t idx = static_cast<uint32_t>(i);
                out.frustum_visible_indices.push_back(idx);
                out.visible_indices.push_back(idx);
            }
        }

        out.stats = make_frustum_only_stats(
            static_cast<uint32_t>(n),
            static_cast<uint32_t>(out.visible_indices.size()));
        return out;
    }

    struct VisibilityHistoryPolicy
    {
        uint8_t hide_confirm_frames = 2;
        uint8_t show_confirm_frames = 1;
    };

    class VisibilityHistory
    {
    public:
        explicit VisibilityHistory(VisibilityHistoryPolicy policy = {})
            : policy_(policy)
        {}

        void set_policy(VisibilityHistoryPolicy policy) noexcept
        {
            policy_ = policy;
        }

        VisibilityHistoryPolicy policy() const noexcept
        {
            return policy_;
        }

        void clear() noexcept
        {
            entries_.clear();
        }

        void reset(uint32_t stable_id) noexcept
        {
            entries_.erase(stable_id);
        }

        bool is_occluded(uint32_t stable_id) const noexcept
        {
            const auto it = entries_.find(stable_id);
            if (it == entries_.end()) return false;
            return it->second.occluded;
        }

        bool update_from_visibility(uint32_t stable_id, bool query_visible) noexcept
        {
            Entry& e = entries_[stable_id];
            if (query_visible)
            {
                e.occluded_streak = 0;
                increment_sat(e.visible_streak);
                if (policy_.show_confirm_frames == 0u ||
                    e.visible_streak >= policy_.show_confirm_frames)
                {
                    e.occluded = false;
                }
            }
            else
            {
                e.visible_streak = 0;
                increment_sat(e.occluded_streak);
                if (policy_.hide_confirm_frames == 0u ||
                    e.occluded_streak >= policy_.hide_confirm_frames)
                {
                    e.occluded = true;
                }
            }

            return e.occluded;
        }

        void prune_to_ids(std::span<const uint32_t> stable_ids)
        {
            std::unordered_set<uint32_t> keep_ids(stable_ids.begin(), stable_ids.end());
            for (auto it = entries_.begin(); it != entries_.end();)
            {
                if (keep_ids.find(it->first) == keep_ids.end()) it = entries_.erase(it);
                else ++it;
            }
        }

    private:
        struct Entry
        {
            uint8_t occluded_streak = 0;
            uint8_t visible_streak = 0;
            bool occluded = false;
        };

        static void increment_sat(uint8_t& v) noexcept
        {
            if (v < 255u) ++v;
        }

        VisibilityHistoryPolicy policy_{};
        std::unordered_map<uint32_t, Entry> entries_{};
    };
}

#endif // SHS_HAS_JOLT
