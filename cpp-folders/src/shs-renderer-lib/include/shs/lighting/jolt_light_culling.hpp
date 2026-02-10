#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: jolt_light_culling.hpp
    МОДУЛЬ: lighting
    ЗОРИЛГО: Гэрлийн shape-уудыг tile/cluster cell-тэй харьцуулан cull хийх.
            Tiled Forward+, Tiled Depth-Range, Clustered гэсэн 3 алгоритм.

    CONVENTION:
        Бүх coordinate-ууд SHS LH space дотор.
        Light shape-ууд SceneShape (Jolt shape + transform) хэлбэрээр ирнэ.
*/

#if defined(SHS_HAS_JOLT) && ((SHS_HAS_JOLT + 0) == 1)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "shs/geometry/volumes.hpp"
#include "shs/geometry/frustum_culling.hpp"
#include "shs/geometry/jolt_culling.hpp"
#include "shs/geometry/scene_shape.hpp"

namespace shs
{
    // =========================================================================
    //  Tiled light culling result
    // =========================================================================

    struct TiledLightCullingResult
    {
        // Per-tile list of visible light indices.
        std::vector<std::vector<uint32_t>> tile_light_lists{};
        uint32_t tiles_x = 0;
        uint32_t tiles_y = 0;
    };


    // =========================================================================
    //  Tiled Forward+ Light Culling
    //  Divides the screen into 2D tiles and tests each light against each tile.
    // =========================================================================

    inline Plane make_oriented_plane_from_points(
        const glm::vec3& a,
        const glm::vec3& b,
        const glm::vec3& c,
        const glm::vec3& inside_point) noexcept
    {
        glm::vec3 normal = glm::normalize(glm::cross(b - a, c - a));
        float d = -glm::dot(normal, a);
        // Ensure the inside point is on the positive side.
        if (glm::dot(normal, inside_point) + d < 0.0f)
        {
            normal = -normal;
            d = -d;
        }
        return Plane{normal, d};
    }

    inline glm::vec3 unproject_ndc(
        const glm::vec3& ndc,
        const glm::mat4& inv_view_proj) noexcept
    {
        const glm::vec4 clip = inv_view_proj * glm::vec4(ndc, 1.0f);
        return glm::vec3(clip) / clip.w;
    }

    inline CullingCell make_screen_tile_cell(
        uint32_t tile_x, uint32_t tile_y,
        uint32_t tiles_x, uint32_t tiles_y,
        uint32_t tile_size,
        uint32_t viewport_w, uint32_t viewport_h,
        const glm::mat4& inv_view_proj,
        float z_near_ndc = -1.0f,
        float z_far_ndc = 1.0f) noexcept
    {
        const float x0 = static_cast<float>(tile_x * tile_size) / static_cast<float>(viewport_w) * 2.0f - 1.0f;
        const float x1 = static_cast<float>(std::min((tile_x + 1) * tile_size, viewport_w)) / static_cast<float>(viewport_w) * 2.0f - 1.0f;
        const float y0 = static_cast<float>(tile_y * tile_size) / static_cast<float>(viewport_h) * 2.0f - 1.0f;
        const float y1 = static_cast<float>(std::min((tile_y + 1) * tile_size, viewport_h)) / static_cast<float>(viewport_h) * 2.0f - 1.0f;

        // 8 corners of the tile frustum in world space.
        const glm::vec3 nbl = unproject_ndc({x0, y0, z_near_ndc}, inv_view_proj);
        const glm::vec3 nbr = unproject_ndc({x1, y0, z_near_ndc}, inv_view_proj);
        const glm::vec3 ntl = unproject_ndc({x0, y1, z_near_ndc}, inv_view_proj);
        const glm::vec3 ntr = unproject_ndc({x1, y1, z_near_ndc}, inv_view_proj);
        const glm::vec3 fbl = unproject_ndc({x0, y0, z_far_ndc}, inv_view_proj);
        const glm::vec3 fbr = unproject_ndc({x1, y0, z_far_ndc}, inv_view_proj);
        const glm::vec3 ftl = unproject_ndc({x0, y1, z_far_ndc}, inv_view_proj);
        const glm::vec3 ftr = unproject_ndc({x1, y1, z_far_ndc}, inv_view_proj);

        const glm::vec3 inside = (nbl + ntr + fbl + ftr) * 0.25f;

        CullingCell cell{};
        cell.kind = CullingCellKind::ScreenTileCell;
        cell.user_data = glm::uvec4(tile_x, tile_y, 0u, 0u);

        culling_cell_add_plane(cell, make_oriented_plane_from_points(nbl, nbr, ntr, inside)); // near
        culling_cell_add_plane(cell, make_oriented_plane_from_points(fbr, fbl, ftl, inside)); // far
        culling_cell_add_plane(cell, make_oriented_plane_from_points(nbl, ntl, ftl, inside)); // left
        culling_cell_add_plane(cell, make_oriented_plane_from_points(nbr, fbr, ftr, inside)); // right
        culling_cell_add_plane(cell, make_oriented_plane_from_points(nbl, fbl, fbr, inside)); // bottom
        culling_cell_add_plane(cell, make_oriented_plane_from_points(ntl, ntr, ftr, inside)); // top

        return cell;
    }

    inline TiledLightCullingResult cull_lights_tiled(
        std::span<const SceneShape> light_shapes,
        const glm::mat4& view_proj,
        uint32_t viewport_w,
        uint32_t viewport_h,
        uint32_t tile_size = 16)
    {
        TiledLightCullingResult result{};
        result.tiles_x = (viewport_w + tile_size - 1) / tile_size;
        result.tiles_y = (viewport_h + tile_size - 1) / tile_size;
        const uint32_t total_tiles = result.tiles_x * result.tiles_y;
        result.tile_light_lists.resize(total_tiles);

        if (light_shapes.empty()) return result;

        const glm::mat4 inv_vp = glm::inverse(view_proj);

        // Pre-compute bounding spheres for all lights.
        std::vector<Sphere> spheres{};
        spheres.reserve(light_shapes.size());
        for (const SceneShape& ls : light_shapes)
        {
            spheres.push_back(ls.bounding_sphere());
        }

        // First: frustum cull all lights against the full camera frustum.
        const Frustum camera_frustum = extract_frustum_planes(view_proj);
        std::vector<bool> frustum_visible(light_shapes.size(), false);
        for (size_t li = 0; li < light_shapes.size(); ++li)
        {
            if (classify_sphere_vs_frustum(spheres[li], camera_frustum) != CullClass::Outside)
            {
                frustum_visible[li] = true;
            }
        }

        for (uint32_t ty = 0; ty < result.tiles_y; ++ty)
        {
            for (uint32_t tx = 0; tx < result.tiles_x; ++tx)
            {
                const CullingCell cell = make_screen_tile_cell(
                    tx, ty, result.tiles_x, result.tiles_y,
                    tile_size, viewport_w, viewport_h, inv_vp);

                const uint32_t tile_index = ty * result.tiles_x + tx;
                auto& tile_list = result.tile_light_lists[tile_index];

                for (uint32_t li = 0; li < static_cast<uint32_t>(light_shapes.size()); ++li)
                {
                    if (!frustum_visible[li]) continue;

                    const CullClass c = classify_sphere_vs_cell(spheres[li], cell);
                    if (c != CullClass::Outside)
                    {
                        tile_list.push_back(li);
                    }
                }
            }
        }
        return result;
    }


    // =========================================================================
    //  Tiled with Depth Range
    //  Uses per-tile min/max depth to create tighter tile cells.
    // =========================================================================

    inline TiledLightCullingResult cull_lights_tiled_depth_range(
        std::span<const SceneShape> light_shapes,
        const glm::mat4& view_proj,
        uint32_t viewport_w,
        uint32_t viewport_h,
        uint32_t tile_size,
        std::span<const float> tile_min_depths,
        std::span<const float> tile_max_depths)
    {
        TiledLightCullingResult result{};
        result.tiles_x = (viewport_w + tile_size - 1) / tile_size;
        result.tiles_y = (viewport_h + tile_size - 1) / tile_size;
        const uint32_t total_tiles = result.tiles_x * result.tiles_y;
        result.tile_light_lists.resize(total_tiles);

        if (light_shapes.empty()) return result;

        const glm::mat4 inv_vp = glm::inverse(view_proj);

        std::vector<Sphere> spheres{};
        spheres.reserve(light_shapes.size());
        for (const SceneShape& ls : light_shapes)
        {
            spheres.push_back(ls.bounding_sphere());
        }

        const Frustum camera_frustum = extract_frustum_planes(view_proj);
        std::vector<bool> frustum_visible(light_shapes.size(), false);
        for (size_t li = 0; li < light_shapes.size(); ++li)
        {
            if (classify_sphere_vs_frustum(spheres[li], camera_frustum) != CullClass::Outside)
            {
                frustum_visible[li] = true;
            }
        }

        for (uint32_t ty = 0; ty < result.tiles_y; ++ty)
        {
            for (uint32_t tx = 0; tx < result.tiles_x; ++tx)
            {
                const uint32_t tile_index = ty * result.tiles_x + tx;

                // Use per-tile depth range if available.
                float z_near_ndc = -1.0f;
                float z_far_ndc = 1.0f;
                if (tile_index < tile_min_depths.size())
                    z_near_ndc = tile_min_depths[tile_index] * 2.0f - 1.0f;
                if (tile_index < tile_max_depths.size())
                    z_far_ndc = tile_max_depths[tile_index] * 2.0f - 1.0f;

                const CullingCell cell = make_screen_tile_cell(
                    tx, ty, result.tiles_x, result.tiles_y,
                    tile_size, viewport_w, viewport_h, inv_vp,
                    z_near_ndc, z_far_ndc);

                auto& tile_list = result.tile_light_lists[tile_index];
                for (uint32_t li = 0; li < static_cast<uint32_t>(light_shapes.size()); ++li)
                {
                    if (!frustum_visible[li]) continue;

                    const CullClass c = classify_sphere_vs_cell(spheres[li], cell);
                    if (c != CullClass::Outside)
                    {
                        tile_list.push_back(li);
                    }
                }
            }
        }
        return result;
    }


    // =========================================================================
    //  Clustered Light Culling (3D grid)
    //  Divides the view frustum into a 3D grid of clusters.
    //  Each cluster is a frustum sub-volume at a specific depth slice.
    // =========================================================================

    struct ClusteredLightCullingResult
    {
        std::vector<std::vector<uint32_t>> cluster_light_lists{};
        uint32_t clusters_x = 0;
        uint32_t clusters_y = 0;
        uint32_t clusters_z = 0;
    };

    inline ClusteredLightCullingResult cull_lights_clustered(
        std::span<const SceneShape> light_shapes,
        const glm::mat4& view_proj,
        uint32_t viewport_w,
        uint32_t viewport_h,
        uint32_t tile_size = 16,
        uint32_t depth_slices = 16,
        float z_near = 0.1f,
        float z_far = 1000.0f)
    {
        ClusteredLightCullingResult result{};
        result.clusters_x = (viewport_w + tile_size - 1) / tile_size;
        result.clusters_y = (viewport_h + tile_size - 1) / tile_size;
        result.clusters_z = depth_slices;
        const uint32_t total = result.clusters_x * result.clusters_y * result.clusters_z;
        result.cluster_light_lists.resize(total);

        if (light_shapes.empty()) return result;

        const glm::mat4 inv_vp = glm::inverse(view_proj);

        std::vector<Sphere> spheres{};
        spheres.reserve(light_shapes.size());
        for (const SceneShape& ls : light_shapes)
        {
            spheres.push_back(ls.bounding_sphere());
        }

        const Frustum camera_frustum = extract_frustum_planes(view_proj);
        std::vector<bool> frustum_visible(light_shapes.size(), false);
        for (size_t li = 0; li < light_shapes.size(); ++li)
        {
            if (classify_sphere_vs_frustum(spheres[li], camera_frustum) != CullClass::Outside)
            {
                frustum_visible[li] = true;
            }
        }

        // Exponential depth slicing: slice_z(k) = z_near * (z_far/z_near)^(k/depth_slices).
        // Convert to NDC: ndc_z = 2 * ((z - z_near) / (z_far - z_near)) - 1  for LH.
        const float log_ratio = std::log(z_far / z_near);
        const float z_range = z_far - z_near;

        for (uint32_t cz = 0; cz < depth_slices; ++cz)
        {
            const float slice_near = z_near * std::exp(log_ratio * static_cast<float>(cz) / static_cast<float>(depth_slices));
            const float slice_far = z_near * std::exp(log_ratio * static_cast<float>(cz + 1) / static_cast<float>(depth_slices));

            const float ndc_near = 2.0f * ((slice_near - z_near) / z_range) - 1.0f;
            const float ndc_far = 2.0f * ((slice_far - z_near) / z_range) - 1.0f;

            for (uint32_t ty = 0; ty < result.clusters_y; ++ty)
            {
                for (uint32_t tx = 0; tx < result.clusters_x; ++tx)
                {
                    const CullingCell cell = make_screen_tile_cell(
                        tx, ty, result.clusters_x, result.clusters_y,
                        tile_size, viewport_w, viewport_h, inv_vp,
                        ndc_near, ndc_far);

                    const uint32_t cluster_index =
                        cz * (result.clusters_x * result.clusters_y) +
                        ty * result.clusters_x + tx;

                    auto& cluster_list = result.cluster_light_lists[cluster_index];
                    for (uint32_t li = 0; li < static_cast<uint32_t>(light_shapes.size()); ++li)
                    {
                        if (!frustum_visible[li]) continue;

                        const CullClass c = classify_sphere_vs_cell(spheres[li], cell);
                        if (c != CullClass::Outside)
                        {
                            cluster_list.push_back(li);
                        }
                    }
                }
            }
        }
        return result;
    }
}

#endif // SHS_HAS_JOLT
