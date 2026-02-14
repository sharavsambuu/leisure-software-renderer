#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: culling_software.hpp
    МОДУЛЬ: geometry
    ЗОРИЛГО: Software occlusion culling-д зориулсан нийтлэг utility болон pipeline.
            Depth-only raster, AABB screen rect projection, rect occlusion test,
            мөн frustum-visible list дээр software occlusion pass гүйцэтгэнэ.
*/

#if defined(SHS_HAS_JOLT) && ((SHS_HAS_JOLT + 0) == 1)

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <span>
#include <vector>

#include <glm/glm.hpp>

#include "shs/geometry/aabb.hpp"
#include "shs/geometry/culling_runtime.hpp"
#include "shs/geometry/jolt_debug_draw.hpp"

namespace shs::culling_sw
{
    struct ScreenRectDepth
    {
        int x_min = 0;
        int y_min = 0;
        int x_max = -1;
        int y_max = -1;
        float z_near = 1.0f;
        bool valid = false;
    };

    inline float edge_function(const glm::vec2& a, const glm::vec2& b, const glm::vec2& p) noexcept
    {
        return (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);
    }

    inline bool project_world_to_screen(
        const glm::vec3& world,
        const glm::mat4& view_proj,
        int width,
        int height,
        glm::vec2& out_xy,
        float& out_depth_01) noexcept
    {
        const glm::vec4 clip = view_proj * glm::vec4(world, 1.0f);
        if (clip.w <= 0.001f) return false;
        const glm::vec3 ndc = glm::vec3(clip) / clip.w;
        if (ndc.z < -1.0f || ndc.z > 1.0f) return false;

        out_xy = glm::vec2(
            (ndc.x + 1.0f) * 0.5f * static_cast<float>(width),
            (ndc.y + 1.0f) * 0.5f * static_cast<float>(height));
        out_depth_01 = ndc.z * 0.5f + 0.5f;
        return true;
    }

    inline void rasterize_depth_triangle(
        std::span<float> depth_buffer,
        int width,
        int height,
        const glm::vec2& p0, float z0,
        const glm::vec2& p1, float z1,
        const glm::vec2& p2, float z2) noexcept
    {
        if (depth_buffer.empty() || width <= 0 || height <= 0) return;
        if (depth_buffer.size() < static_cast<size_t>(width) * static_cast<size_t>(height)) return;

        const float area = edge_function(p0, p1, p2);
        if (std::abs(area) <= 1e-6f) return;

        const float min_xf = std::min(p0.x, std::min(p1.x, p2.x));
        const float min_yf = std::min(p0.y, std::min(p1.y, p2.y));
        const float max_xf = std::max(p0.x, std::max(p1.x, p2.x));
        const float max_yf = std::max(p0.y, std::max(p1.y, p2.y));

        const int min_x = std::max(0, static_cast<int>(std::floor(min_xf)));
        const int min_y = std::max(0, static_cast<int>(std::floor(min_yf)));
        const int max_x = std::min(width - 1, static_cast<int>(std::ceil(max_xf)));
        const int max_y = std::min(height - 1, static_cast<int>(std::ceil(max_yf)));
        if (min_x > max_x || min_y > max_y) return;

        const bool ccw = area > 0.0f;
        for (int y = min_y; y <= max_y; ++y)
        {
            for (int x = min_x; x <= max_x; ++x)
            {
                const glm::vec2 p(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f);
                const float w0 = edge_function(p1, p2, p);
                const float w1 = edge_function(p2, p0, p);
                const float w2 = edge_function(p0, p1, p);
                const bool inside = ccw
                    ? (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f)
                    : (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f);
                if (!inside) continue;

                const float iw0 = w0 / area;
                const float iw1 = w1 / area;
                const float iw2 = w2 / area;
                const float depth = iw0 * z0 + iw1 * z1 + iw2 * z2;
                if (depth < 0.0f || depth > 1.0f) continue;

                const size_t depth_idx =
                    static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
                if (depth < depth_buffer[depth_idx]) depth_buffer[depth_idx] = depth;
            }
        }
    }

    inline void rasterize_mesh_depth_transformed(
        std::span<float> depth_buffer,
        int width,
        int height,
        const DebugMesh& mesh_local,
        const glm::mat4& model,
        const glm::mat4& view_proj) noexcept
    {
        for (size_t i = 0; i + 2 < mesh_local.indices.size(); i += 3)
        {
            const glm::vec3 lp0 = mesh_local.vertices[mesh_local.indices[i + 0]];
            const glm::vec3 lp1 = mesh_local.vertices[mesh_local.indices[i + 1]];
            const glm::vec3 lp2 = mesh_local.vertices[mesh_local.indices[i + 2]];

            const glm::vec3 p0 = glm::vec3(model * glm::vec4(lp0, 1.0f));
            const glm::vec3 p1 = glm::vec3(model * glm::vec4(lp1, 1.0f));
            const glm::vec3 p2 = glm::vec3(model * glm::vec4(lp2, 1.0f));

            glm::vec2 s0, s1, s2;
            float z0 = 1.0f, z1 = 1.0f, z2 = 1.0f;
            if (!project_world_to_screen(p0, view_proj, width, height, s0, z0)) continue;
            if (!project_world_to_screen(p1, view_proj, width, height, s1, z1)) continue;
            if (!project_world_to_screen(p2, view_proj, width, height, s2, z2)) continue;

            rasterize_depth_triangle(depth_buffer, width, height, s0, z0, s1, z1, s2, z2);
        }
    }

    inline ScreenRectDepth project_aabb_to_screen_rect(
        const AABB& aabb,
        const glm::mat4& view_proj,
        int width,
        int height) noexcept
    {
        ScreenRectDepth out{};
        if (width <= 0 || height <= 0) return out;

        const glm::vec3 corners[8] = {
            {aabb.minv.x, aabb.minv.y, aabb.minv.z},
            {aabb.maxv.x, aabb.minv.y, aabb.minv.z},
            {aabb.minv.x, aabb.maxv.y, aabb.minv.z},
            {aabb.maxv.x, aabb.maxv.y, aabb.minv.z},
            {aabb.minv.x, aabb.minv.y, aabb.maxv.z},
            {aabb.maxv.x, aabb.minv.y, aabb.maxv.z},
            {aabb.minv.x, aabb.maxv.y, aabb.maxv.z},
            {aabb.maxv.x, aabb.maxv.y, aabb.maxv.z},
        };

        float min_x = static_cast<float>(width);
        float min_y = static_cast<float>(height);
        float max_x = -1.0f;
        float max_y = -1.0f;
        float near_depth = 1.0f;
        bool any = false;

        for (const glm::vec3& c : corners)
        {
            const glm::vec4 clip = view_proj * glm::vec4(c, 1.0f);
            if (clip.w <= 0.001f) continue;
            const glm::vec3 ndc = glm::vec3(clip) / clip.w;
            const float z01 = ndc.z * 0.5f + 0.5f;
            if (z01 < 0.0f || z01 > 1.0f) continue;

            const float sx = (ndc.x + 1.0f) * 0.5f * static_cast<float>(width);
            const float sy = (ndc.y + 1.0f) * 0.5f * static_cast<float>(height);
            min_x = std::min(min_x, sx);
            min_y = std::min(min_y, sy);
            max_x = std::max(max_x, sx);
            max_y = std::max(max_y, sy);
            near_depth = std::min(near_depth, z01);
            any = true;
        }

        if (!any) return out;

        out.x_min = std::max(0, static_cast<int>(std::floor(min_x)));
        out.y_min = std::max(0, static_cast<int>(std::floor(min_y)));
        out.x_max = std::min(width - 1, static_cast<int>(std::ceil(max_x)));
        out.y_max = std::min(height - 1, static_cast<int>(std::ceil(max_y)));
        out.z_near = std::clamp(near_depth, 0.0f, 1.0f);
        out.valid = out.x_min <= out.x_max && out.y_min <= out.y_max;
        return out;
    }

    inline bool is_rect_occluded(
        std::span<const float> depth_buffer,
        int width,
        int height,
        const ScreenRectDepth& rect,
        float epsilon = 1e-4f) noexcept
    {
        if (!rect.valid) return false;
        if (depth_buffer.empty() || width <= 0 || height <= 0) return false;
        if (depth_buffer.size() < static_cast<size_t>(width) * static_cast<size_t>(height)) return false;

        for (int y = rect.y_min; y <= rect.y_max; ++y)
        {
            for (int x = rect.x_min; x <= rect.x_max; ++x)
            {
                const size_t depth_idx =
                    static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
                if (rect.z_near <= depth_buffer[depth_idx] + epsilon) return false;
            }
        }
        return true;
    }

    inline float view_depth_of_aabb_center(
        const AABB& box,
        const glm::mat4& view) noexcept
    {
        const glm::vec3 center = box.center();
        const glm::vec4 v = view * glm::vec4(center, 1.0f);
        return v.z;
    }

    template<typename TObject, typename GetAabbFn, typename GetViewDepthFn,
             typename SetOccludedFn, typename SetVisibleFn, typename RasterizeOccluderFn>
    requires requires(
        TObject& object,
        const GetAabbFn& get_world_aabb,
        const GetViewDepthFn& get_view_depth,
        const SetOccludedFn& set_occluded,
        const SetVisibleFn& set_visible,
        const RasterizeOccluderFn& rasterize_occluder,
        const glm::mat4& view,
        std::span<float> depth_buffer,
        uint32_t object_index,
        bool flag)
    {
        { get_world_aabb(object) } -> std::convertible_to<AABB>;
        { static_cast<float>(get_view_depth(object, view)) } -> std::same_as<float>;
        { set_occluded(object, flag) } -> std::same_as<void>;
        { set_visible(object, flag) } -> std::same_as<void>;
        { rasterize_occluder(object, object_index, depth_buffer) } -> std::same_as<void>;
    }
    inline CullingStats run_software_occlusion_pass(
        std::span<TObject> objects,
        std::span<const uint32_t> frustum_visible_indices,
        bool enable_occlusion,
        std::span<float> occlusion_depth,
        int occlusion_width,
        int occlusion_height,
        const glm::mat4& view,
        const glm::mat4& view_proj,
        const GetAabbFn& get_world_aabb,
        const GetViewDepthFn& get_view_depth,
        const SetOccludedFn& set_occluded,
        const SetVisibleFn& set_visible,
        const RasterizeOccluderFn& rasterize_occluder,
        std::vector<uint32_t>& visible_indices_out,
        float depth_epsilon = 1e-4f)
    {
        visible_indices_out.clear();
        visible_indices_out.reserve(frustum_visible_indices.size());

        if (!enable_occlusion)
        {
            for (const uint32_t idx : frustum_visible_indices)
            {
                if (idx >= objects.size()) continue;
                TObject& object = objects[idx];
                set_occluded(object, false);
                set_visible(object, true);
                visible_indices_out.push_back(idx);
            }

            return make_culling_stats(
                static_cast<uint32_t>(objects.size()),
                static_cast<uint32_t>(frustum_visible_indices.size()),
                static_cast<uint32_t>(visible_indices_out.size()));
        }

        std::fill(occlusion_depth.begin(), occlusion_depth.end(), 1.0f);

        std::vector<uint32_t> sorted_indices(frustum_visible_indices.begin(), frustum_visible_indices.end());
        std::sort(
            sorted_indices.begin(),
            sorted_indices.end(),
            [&](uint32_t a, uint32_t b)
            {
                if (a >= objects.size()) return false;
                if (b >= objects.size()) return true;
                return get_view_depth(objects[a], view) < get_view_depth(objects[b], view);
            });

        uint32_t occluded_count = 0;
        for (const uint32_t idx : sorted_indices)
        {
            if (idx >= objects.size()) continue;
            TObject& object = objects[idx];
            const AABB world_aabb = get_world_aabb(object);
            const ScreenRectDepth rect =
                project_aabb_to_screen_rect(world_aabb, view_proj, occlusion_width, occlusion_height);

            const bool occluded =
                is_rect_occluded(occlusion_depth, occlusion_width, occlusion_height, rect, depth_epsilon);
            set_occluded(object, occluded);
            set_visible(object, !occluded);
            if (occluded)
            {
                ++occluded_count;
                continue;
            }

            visible_indices_out.push_back(idx);
            rasterize_occluder(object, idx, occlusion_depth);
        }

        CullingStats stats = make_culling_stats(
            static_cast<uint32_t>(objects.size()),
            static_cast<uint32_t>(frustum_visible_indices.size()),
            static_cast<uint32_t>(visible_indices_out.size()));
        stats.occluded_count = occluded_count;
        normalize_culling_stats(stats);
        return stats;
    }
}

#endif // SHS_HAS_JOLT
