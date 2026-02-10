#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: jolt_occlusion_culling.hpp
    МОДУЛЬ: geometry
    ЗОРИЛГО: Occlusion culling-ийн суурь API.
            Hi-Z buffer дээр суурилсан software occlusion тест,
            ирээдүйд Jolt BroadPhaseQuery-тэй хослуулах боломжтой.
*/

#if defined(SHS_HAS_JOLT) && ((SHS_HAS_JOLT + 0) == 1)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include <glm/glm.hpp>

#include "shs/geometry/aabb.hpp"
#include "shs/geometry/volumes.hpp"
#include "shs/geometry/jolt_adapter.hpp"
#include "shs/geometry/jolt_shape_traits.hpp"
#include "shs/geometry/scene_shape.hpp"

namespace shs
{
    // =========================================================================
    //  Occlusion culling result
    // =========================================================================

    struct OcclusionResult
    {
        std::vector<bool>   occluded{};
        std::vector<size_t> visible_indices{};
        uint64_t tested = 0;
        uint64_t occluded_count = 0;
        uint64_t visible_count = 0;
    };


    // =========================================================================
    //  Hi-Z based occlusion culling
    //  Tests object's screen-space AABB against a hierarchical depth buffer.
    //  The Hi-Z buffer stores the maximum depth per pixel at various mip levels.
    // =========================================================================

    namespace detail
    {
        /// Project an SHS LH world-space AABB to screen-space min/max and depth.
        struct ScreenRect
        {
            float x_min = 1.0f, x_max = -1.0f;
            float y_min = 1.0f, y_max = -1.0f;
            float z_min = 1.0f; // nearest depth in NDC
            bool valid = false;
        };

        inline ScreenRect project_aabb_to_screen(
            const AABB& aabb,
            const glm::mat4& view_proj,
            uint32_t viewport_w,
            uint32_t viewport_h) noexcept
        {
            ScreenRect rect{};
            rect.z_min = 1.0f;

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

            bool any_in_front = false;
            for (const glm::vec3& c : corners)
            {
                const glm::vec4 clip = view_proj * glm::vec4(c, 1.0f);
                if (clip.w <= 0.0f) continue; // Behind camera.
                any_in_front = true;

                const float inv_w = 1.0f / clip.w;
                const float ndc_x = clip.x * inv_w;
                const float ndc_y = clip.y * inv_w;
                const float ndc_z = clip.z * inv_w;

                // NDC to screen: [0, viewport_w/h]
                const float sx = (ndc_x * 0.5f + 0.5f) * static_cast<float>(viewport_w);
                const float sy = (ndc_y * 0.5f + 0.5f) * static_cast<float>(viewport_h);

                rect.x_min = std::min(rect.x_min, sx);
                rect.x_max = std::max(rect.x_max, sx);
                rect.y_min = std::min(rect.y_min, sy);
                rect.y_max = std::max(rect.y_max, sy);
                rect.z_min = std::min(rect.z_min, ndc_z);
            }

            rect.valid = any_in_front && rect.x_min < rect.x_max && rect.y_min < rect.y_max;
            return rect;
        }

        /// Check if a screen rect is occluded by the Hi-Z buffer.
        /// The Hi-Z buffer is a flat array of max-depth values for each pixel.
        /// Returns true if the object is guaranteed to be hidden behind existing geometry.
        inline bool is_occluded_hiz(
            const ScreenRect& rect,
            uint32_t hiz_width,
            uint32_t hiz_height,
            std::span<const float> hiz_buffer) noexcept
        {
            if (!rect.valid) return false;
            if (hiz_buffer.empty()) return false;

            // Clamp to Hi-Z buffer dimensions.
            const uint32_t px_min = static_cast<uint32_t>(std::clamp(rect.x_min, 0.0f, static_cast<float>(hiz_width - 1)));
            const uint32_t px_max = static_cast<uint32_t>(std::clamp(rect.x_max, 0.0f, static_cast<float>(hiz_width - 1)));
            const uint32_t py_min = static_cast<uint32_t>(std::clamp(rect.y_min, 0.0f, static_cast<float>(hiz_height - 1)));
            const uint32_t py_max = static_cast<uint32_t>(std::clamp(rect.y_max, 0.0f, static_cast<float>(hiz_height - 1)));

            // Find the maximum depth in the Hi-Z buffer within the screen rect.
            float max_hiz_depth = -1.0f;
            for (uint32_t y = py_min; y <= py_max; ++y)
            {
                for (uint32_t x = px_min; x <= px_max; ++x)
                {
                    const uint32_t idx = y * hiz_width + x;
                    if (idx < hiz_buffer.size())
                    {
                        max_hiz_depth = std::max(max_hiz_depth, hiz_buffer[idx]);
                    }
                }
            }

            // Occluded if the object's nearest depth is farther than the max Hi-Z depth.
            return rect.z_min > max_hiz_depth;
        }
    }


    // =========================================================================
    //  Hi-Z Occlusion Cull (batch)
    // =========================================================================

    template<FastCullable T>
    inline OcclusionResult occlusion_cull(
        std::span<const T> objects,
        const glm::mat4& view_proj,
        uint32_t hiz_width,
        uint32_t hiz_height,
        std::span<const float> hiz_buffer)
    {
        OcclusionResult out{};
        const size_t n = objects.size();
        out.occluded.resize(n, false);
        out.visible_indices.reserve(n);
        out.tested = n;

        for (size_t i = 0; i < n; ++i)
        {
            AABB world_box{};
            if constexpr (HasWorldAABB<T>)
            {
                world_box = objects[i].world_aabb();
            }
            else
            {
                // Fallback: use bounding sphere to make AABB.
                const Sphere s = objects[i].bounding_sphere();
                world_box.minv = s.center - glm::vec3(s.radius);
                world_box.maxv = s.center + glm::vec3(s.radius);
            }

            const auto screen_rect = detail::project_aabb_to_screen(
                world_box, view_proj, hiz_width, hiz_height);

            if (detail::is_occluded_hiz(screen_rect, hiz_width, hiz_height, hiz_buffer))
            {
                out.occluded[i] = true;
                ++out.occluded_count;
            }
            else
            {
                out.visible_indices.push_back(i);
                ++out.visible_count;
            }
        }
        return out;
    }
}

#endif // SHS_HAS_JOLT
