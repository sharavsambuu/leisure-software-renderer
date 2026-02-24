#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: light_set.hpp
    МОДУЛЬ: lighting
    ЗОРИЛГО: Төрөлжсөн гэрлүүдийг нэгтгээд culling-д бэлэн GPU payload руу
            flatten хийх extendable container семантик.
*/

#include <algorithm>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

#include "shs/lighting/light_types.hpp"

namespace shs
{
    struct LightSet
    {
        std::vector<PointLight> points{};
        std::vector<SpotLight> spots{};
        std::vector<RectAreaLight> rect_areas{};
        std::vector<TubeAreaLight> tube_areas{};

        void clear_local_lights()
        {
            points.clear();
            spots.clear();
            rect_areas.clear();
            tube_areas.clear();
        }

        size_t local_light_count() const
        {
            return points.size() + spots.size() + rect_areas.size() + tube_areas.size();
        }

        std::vector<CullingLightGPU> to_cullable_gpu(
            size_t max_count = std::numeric_limits<size_t>::max()) const
        {
            std::vector<CullingLightGPU> out{};
            flatten_cullable_gpu(out, max_count);
            return out;
        }

        void flatten_cullable_gpu(
            std::vector<CullingLightGPU>& out,
            size_t max_count = std::numeric_limits<size_t>::max()) const
        {
            out.clear();
            out.reserve(std::min(local_light_count(), max_count));

            const auto append = [&](const auto& src, auto&& builder) {
                for (const auto& l : src)
                {
                    if (out.size() >= max_count) return;
                    out.push_back(builder(l));
                }
            };

            append(points, [](const PointLight& l) { return make_point_culling_light(l); });
            append(spots, [](const SpotLight& l) { return make_spot_culling_light(l); });
            append(rect_areas, [](const RectAreaLight& l) { return make_rect_area_culling_light(l); });
            append(tube_areas, [](const TubeAreaLight& l) { return make_tube_area_culling_light(l); });
        }

        // Overlay Jolt-backed (or any FastCullable-like) bounds onto the packed lights.
        // Source ordering must match flatten_cullable_gpu order:
        // points -> spots -> rect_areas -> tube_areas.
        template<LightCullSphereSource T>
        void flatten_cullable_gpu(
            std::vector<CullingLightGPU>& out,
            std::span<const T> cull_sources,
            size_t max_count,
            LightCullingShape source_shape = LightCullingShape::GenericJoltBounds) const
        {
            flatten_cullable_gpu(out, max_count);
            const size_t n = std::min(out.size(), cull_sources.size());
            for (size_t i = 0; i < n; ++i)
            {
                apply_light_cull_bounds_from_source(out[i], cull_sources[i], source_shape);
            }
        }

        template<LightCullSphereSource T>
        std::vector<CullingLightGPU> to_cullable_gpu(
            std::span<const T> cull_sources,
            size_t max_count,
            LightCullingShape source_shape = LightCullingShape::GenericJoltBounds) const
        {
            std::vector<CullingLightGPU> out{};
            flatten_cullable_gpu(out, cull_sources, max_count, source_shape);
            return out;
        }

        template<LightCullSphereSource T>
        void flatten_cullable_gpu(
            std::vector<CullingLightGPU>& out,
            std::span<const T> cull_sources,
            LightCullingShape source_shape = LightCullingShape::GenericJoltBounds) const
        {
            flatten_cullable_gpu(out, cull_sources, std::numeric_limits<size_t>::max(), source_shape);
        }

        template<LightCullSphereSource T>
        std::vector<CullingLightGPU> to_cullable_gpu(
            std::span<const T> cull_sources,
            LightCullingShape source_shape = LightCullingShape::GenericJoltBounds) const
        {
            return to_cullable_gpu(cull_sources, std::numeric_limits<size_t>::max(), source_shape);
        }
    };
}
