#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: culling_visibility.hpp
    МОДУЛЬ: geometry
    ЗОРИЛГО: Frustum/Occlusion visibility state-ийг нэгтгэсэн helper API.
            Query result -> VisibilityHistory update, мөн frustum list-ээс
            render-visible list үүсгэх ерөнхий utility функцууд.
*/

#if defined(SHS_HAS_JOLT) && ((SHS_HAS_JOLT + 0) == 1)

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <span>
#include <vector>

#include "shs/geometry/culling_runtime.hpp"

namespace shs
{
    template<typename TObject, typename GetStableIdFn, typename SetOccludedFn>
    requires requires(
        TObject& object,
        const GetStableIdFn& get_stable_id,
        const SetOccludedFn& set_occluded,
        bool occluded)
    {
        { static_cast<uint32_t>(get_stable_id(object)) } -> std::same_as<uint32_t>;
        { set_occluded(object, occluded) } -> std::same_as<void>;
    }
    inline void apply_query_visibility_samples(
        std::span<TObject> objects,
        std::span<const uint32_t> object_indices,
        std::span<const uint64_t> passed_samples,
        uint64_t min_visible_samples,
        VisibilityHistory& history,
        const GetStableIdFn& get_stable_id,
        const SetOccludedFn& set_occluded)
    {
        const size_t mapped = std::min(object_indices.size(), passed_samples.size());
        for (size_t i = 0; i < mapped; ++i)
        {
            const uint32_t object_index = object_indices[i];
            if (object_index >= objects.size()) continue;

            TObject& object = objects[object_index];
            const bool query_visible = passed_samples[i] >= min_visible_samples;
            const bool occluded =
                history.update_from_visibility(get_stable_id(object), query_visible);
            set_occluded(object, occluded);
        }
    }

    template<typename TObject, typename GetOccludedFn, typename SetVisibleFn>
    requires requires(
        TObject& object,
        const GetOccludedFn& get_occluded,
        const SetVisibleFn& set_visible,
        bool visible)
    {
        { static_cast<bool>(get_occluded(object)) } -> std::same_as<bool>;
        { set_visible(object, visible) } -> std::same_as<void>;
    }
    inline CullingStats build_visibility_from_frustum(
        std::span<TObject> objects,
        std::span<const uint32_t> frustum_visible_indices,
        bool apply_occlusion,
        const GetOccludedFn& get_occluded,
        const SetVisibleFn& set_visible,
        std::vector<uint32_t>& visible_indices_out)
    {
        visible_indices_out.clear();
        visible_indices_out.reserve(frustum_visible_indices.size());

        uint32_t occluded_count = 0;
        for (const uint32_t idx : frustum_visible_indices)
        {
            if (idx >= objects.size()) continue;
            TObject& object = objects[idx];
            const bool occluded = apply_occlusion && get_occluded(object);
            const bool visible = !occluded;
            set_visible(object, visible);

            if (visible) visible_indices_out.push_back(idx);
            else ++occluded_count;
        }

        CullingStats stats = make_culling_stats(
            static_cast<uint32_t>(objects.size()),
            static_cast<uint32_t>(frustum_visible_indices.size()),
            static_cast<uint32_t>(visible_indices_out.size()));
        stats.occluded_count = occluded_count;
        normalize_culling_stats(stats);
        return stats;
    }

    inline bool should_use_frustum_visibility_fallback(
        bool enable_occlusion,
        bool has_depth_attachment,
        uint32_t query_count,
        const CullingStats& stats) noexcept
    {
        return
            enable_occlusion &&
            has_depth_attachment &&
            stats.frustum_visible_count > 0u &&
            query_count == 0u &&
            stats.visible_count == 0u;
    }
}

#endif // SHS_HAS_JOLT
