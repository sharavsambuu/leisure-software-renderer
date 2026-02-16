#pragma once

/*
    SHS renderer lib

    FILE: light_culling_runtime.hpp
    MODULE: lighting
    GOAL: Build per-frame light bins (None/Tiled/TiledDepth/Clustered)
          and expose per-object candidate lookup helpers.
*/

#if defined(SHS_HAS_JOLT) && ((SHS_HAS_JOLT + 0) == 1)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include <glm/glm.hpp>

#include "shs/lighting/jolt_light_culling.hpp"
#include "shs/lighting/light_culling_mode.hpp"
#include "shs/scene/scene_elements.hpp"

namespace shs
{
    struct LightBinCullingConfig
    {
        LightCullingMode mode = LightCullingMode::None;
        uint32_t tile_size = 16u;
        uint32_t cluster_depth_slices = 16u;
        float z_near = 0.1f;
        float z_far = 1000.0f;
    };

    struct TileViewDepthRange
    {
        uint32_t tiles_x = 0;
        uint32_t tiles_y = 0;
        std::vector<float> min_view_depth{};
        std::vector<float> max_view_depth{};

        bool valid() const noexcept
        {
            return !min_view_depth.empty() && min_view_depth.size() == max_view_depth.size();
        }
    };

    struct LightBinCullingData
    {
        LightCullingMode mode = LightCullingMode::None;
        uint32_t bins_x = 0;
        uint32_t bins_y = 0;
        uint32_t bins_z = 1;
        uint32_t tile_size = 16u;
        float z_near = 0.1f;
        float z_far = 1000.0f;

        // Scene indices after frustum/occlusion pre-filtering.
        std::vector<uint32_t> fallback_scene_indices{};
        // Local light index -> light-scene index mapping.
        std::vector<uint32_t> local_to_scene_indices{};
        // Per-bin local light lists (local index in local_to_scene_indices).
        std::vector<std::vector<uint32_t>> bin_local_light_lists{};

        bool has_bins() const noexcept
        {
            return !bin_local_light_lists.empty() && bins_x > 0u && bins_y > 0u && bins_z > 0u;
        }
    };

    inline std::span<const uint32_t> fallback_light_scene_candidates(const LightBinCullingData& data)
    {
        return std::span<const uint32_t>(data.fallback_scene_indices.data(), data.fallback_scene_indices.size());
    }

    inline std::array<glm::vec3, 8> aabb_corners(const AABB& box)
    {
        return {
            glm::vec3(box.minv.x, box.minv.y, box.minv.z),
            glm::vec3(box.maxv.x, box.minv.y, box.minv.z),
            glm::vec3(box.minv.x, box.maxv.y, box.minv.z),
            glm::vec3(box.maxv.x, box.maxv.y, box.minv.z),
            glm::vec3(box.minv.x, box.minv.y, box.maxv.z),
            glm::vec3(box.maxv.x, box.minv.y, box.maxv.z),
            glm::vec3(box.minv.x, box.maxv.y, box.maxv.z),
            glm::vec3(box.maxv.x, box.maxv.y, box.maxv.z)
        };
    }

    inline bool project_aabb_bounds(
        const AABB& box,
        const glm::mat4& view,
        const glm::mat4& view_proj,
        float& out_ndc_min_x,
        float& out_ndc_max_x,
        float& out_ndc_min_y,
        float& out_ndc_max_y,
        float& out_min_view_depth,
        float& out_max_view_depth,
        float z_near,
        float z_far)
    {
        bool any = false;
        out_ndc_min_x = 1.0f;
        out_ndc_max_x = -1.0f;
        out_ndc_min_y = 1.0f;
        out_ndc_max_y = -1.0f;
        out_min_view_depth = z_far;
        out_max_view_depth = z_near;

        const auto corners = aabb_corners(box);
        for (const glm::vec3& p : corners)
        {
            const glm::vec4 clip = view_proj * glm::vec4(p, 1.0f);
            if (clip.w <= 1e-5f) continue;

            const glm::vec3 ndc = glm::vec3(clip) / clip.w;
            out_ndc_min_x = std::min(out_ndc_min_x, ndc.x);
            out_ndc_max_x = std::max(out_ndc_max_x, ndc.x);
            out_ndc_min_y = std::min(out_ndc_min_y, ndc.y);
            out_ndc_max_y = std::max(out_ndc_max_y, ndc.y);

            const float view_depth = (view * glm::vec4(p, 1.0f)).z;
            if (view_depth > 1e-5f)
            {
                out_min_view_depth = std::min(out_min_view_depth, view_depth);
                out_max_view_depth = std::max(out_max_view_depth, view_depth);
            }

            any = true;
        }

        if (!any) return false;

        out_ndc_min_x = std::clamp(out_ndc_min_x, -1.0f, 1.0f);
        out_ndc_max_x = std::clamp(out_ndc_max_x, -1.0f, 1.0f);
        out_ndc_min_y = std::clamp(out_ndc_min_y, -1.0f, 1.0f);
        out_ndc_max_y = std::clamp(out_ndc_max_y, -1.0f, 1.0f);

        if (out_ndc_min_x > out_ndc_max_x) std::swap(out_ndc_min_x, out_ndc_max_x);
        if (out_ndc_min_y > out_ndc_max_y) std::swap(out_ndc_min_y, out_ndc_max_y);

        out_min_view_depth = std::clamp(out_min_view_depth, z_near, z_far);
        out_max_view_depth = std::clamp(out_max_view_depth, z_near, z_far);
        if (out_min_view_depth > out_max_view_depth)
        {
            out_min_view_depth = z_near;
            out_max_view_depth = z_far;
        }
        return true;
    }

    inline uint32_t ndc_x_to_bin(float ndc_x, uint32_t bins_x)
    {
        if (bins_x == 0u) return 0u;
        const float u = std::clamp(ndc_x * 0.5f + 0.5f, 0.0f, 0.999999f);
        return std::min(static_cast<uint32_t>(u * static_cast<float>(bins_x)), bins_x - 1u);
    }

    inline uint32_t ndc_y_to_bin_top_origin(float ndc_y, uint32_t bins_y)
    {
        if (bins_y == 0u) return 0u;
        const float v = std::clamp(1.0f - (ndc_y * 0.5f + 0.5f), 0.0f, 0.999999f);
        return std::min(static_cast<uint32_t>(v * static_cast<float>(bins_y)), bins_y - 1u);
    }

    inline uint32_t view_depth_to_cluster_slice(
        float view_depth,
        float z_near,
        float z_far,
        uint32_t cluster_slices)
    {
        if (cluster_slices <= 1u) return 0u;

        const float zn = std::max(z_near, 1e-4f);
        const float zf = std::max(z_far, zn + 1e-3f);
        const float d = std::clamp(view_depth, zn, zf);
        const float log_ratio = std::log(zf / zn);
        if (log_ratio <= 1e-6f) return 0u;

        const float t = std::clamp(std::log(d / zn) / log_ratio, 0.0f, 0.999999f);
        return std::min(static_cast<uint32_t>(t * static_cast<float>(cluster_slices)), cluster_slices - 1u);
    }

    inline TileViewDepthRange build_tile_view_depth_range_from_scene(
        std::span<const uint32_t> visible_scene_indices,
        const SceneElementSet& scene,
        const glm::mat4& view,
        const glm::mat4& view_proj,
        uint32_t viewport_w,
        uint32_t viewport_h,
        uint32_t tile_size,
        float z_near,
        float z_far)
    {
        TileViewDepthRange out{};
        if (viewport_w == 0u || viewport_h == 0u || tile_size == 0u) return out;

        out.tiles_x = (viewport_w + tile_size - 1u) / tile_size;
        out.tiles_y = (viewport_h + tile_size - 1u) / tile_size;
        const uint32_t total_tiles = out.tiles_x * out.tiles_y;
        out.min_view_depth.assign(total_tiles, z_far);
        out.max_view_depth.assign(total_tiles, z_near);
        std::vector<uint8_t> has_depth(total_tiles, 0u);

        for (const uint32_t scene_idx : visible_scene_indices)
        {
            if (scene_idx >= scene.size()) continue;
            const AABB box = scene[scene_idx].geometry.world_aabb();

            float min_x = 0.0f;
            float max_x = 0.0f;
            float min_y = 0.0f;
            float max_y = 0.0f;
            float min_depth = z_near;
            float max_depth = z_far;
            if (!project_aabb_bounds(
                    box,
                    view,
                    view_proj,
                    min_x,
                    max_x,
                    min_y,
                    max_y,
                    min_depth,
                    max_depth,
                    z_near,
                    z_far))
            {
                continue;
            }

            const uint32_t tx0 = ndc_x_to_bin(min_x, out.tiles_x);
            const uint32_t tx1 = ndc_x_to_bin(max_x, out.tiles_x);
            const uint32_t ty0 = ndc_y_to_bin_top_origin(max_y, out.tiles_y);
            const uint32_t ty1 = ndc_y_to_bin_top_origin(min_y, out.tiles_y);

            for (uint32_t ty = ty0; ty <= ty1; ++ty)
            {
                for (uint32_t tx = tx0; tx <= tx1; ++tx)
                {
                    const uint32_t tile_idx = ty * out.tiles_x + tx;
                    if (tile_idx >= total_tiles) continue;
                    out.min_view_depth[tile_idx] = std::min(out.min_view_depth[tile_idx], min_depth);
                    out.max_view_depth[tile_idx] = std::max(out.max_view_depth[tile_idx], max_depth);
                    has_depth[tile_idx] = 1u;
                }
            }
        }

        for (uint32_t i = 0; i < total_tiles; ++i)
        {
            if (has_depth[i] == 0u || out.min_view_depth[i] > out.max_view_depth[i])
            {
                out.min_view_depth[i] = z_near;
                out.max_view_depth[i] = z_far;
            }
        }

        return out;
    }

    inline LightBinCullingData build_light_bin_culling(
        std::span<const uint32_t> visible_light_scene_indices,
        const SceneElementSet& light_scene,
        const glm::mat4& view_proj,
        uint32_t viewport_w,
        uint32_t viewport_h,
        const LightBinCullingConfig& cfg,
        std::span<const float> tile_min_view_depth = {},
        std::span<const float> tile_max_view_depth = {})
    {
        LightBinCullingData out{};
        out.mode = cfg.mode;
        out.tile_size = std::max(cfg.tile_size, 1u);
        out.z_near = std::max(cfg.z_near, 1e-4f);
        out.z_far = std::max(cfg.z_far, out.z_near + 1e-3f);
        out.fallback_scene_indices.assign(visible_light_scene_indices.begin(), visible_light_scene_indices.end());

        if (cfg.mode == LightCullingMode::None || visible_light_scene_indices.empty())
        {
            return out;
        }

        std::vector<SceneShape> light_shapes{};
        light_shapes.reserve(visible_light_scene_indices.size());
        out.local_to_scene_indices.reserve(visible_light_scene_indices.size());

        for (const uint32_t scene_idx : visible_light_scene_indices)
        {
            if (scene_idx >= light_scene.size()) continue;
            light_shapes.push_back(light_scene[scene_idx].geometry);
            out.local_to_scene_indices.push_back(scene_idx);
        }

        if (light_shapes.empty())
        {
            return out;
        }

        if (cfg.mode == LightCullingMode::Clustered)
        {
            const ClusteredLightCullingResult clustered = cull_lights_clustered(
                std::span<const SceneShape>(light_shapes.data(), light_shapes.size()),
                view_proj,
                viewport_w,
                viewport_h,
                out.tile_size,
                std::max(cfg.cluster_depth_slices, 1u),
                out.z_near,
                out.z_far);

            out.bins_x = clustered.clusters_x;
            out.bins_y = clustered.clusters_y;
            out.bins_z = std::max(clustered.clusters_z, 1u);
            out.bin_local_light_lists = clustered.cluster_light_lists;
            return out;
        }

        TiledLightCullingResult tiled{};
        if (cfg.mode == LightCullingMode::TiledDepthRange)
        {
            const uint32_t tiles_x = (viewport_w + out.tile_size - 1u) / out.tile_size;
            const uint32_t tiles_y = (viewport_h + out.tile_size - 1u) / out.tile_size;
            const uint32_t total_tiles = tiles_x * tiles_y;
            const bool depth_ok =
                (tile_min_view_depth.size() == total_tiles) &&
                (tile_max_view_depth.size() == total_tiles);

            if (depth_ok)
            {
                tiled = cull_lights_tiled_view_depth_range(
                    std::span<const SceneShape>(light_shapes.data(), light_shapes.size()),
                    view_proj,
                    viewport_w,
                    viewport_h,
                    out.tile_size,
                    tile_min_view_depth,
                    tile_max_view_depth,
                    out.z_near,
                    out.z_far);
            }
            else
            {
                tiled = cull_lights_tiled(
                    std::span<const SceneShape>(light_shapes.data(), light_shapes.size()),
                    view_proj,
                    viewport_w,
                    viewport_h,
                    out.tile_size);
            }
        }
        else
        {
            tiled = cull_lights_tiled(
                std::span<const SceneShape>(light_shapes.data(), light_shapes.size()),
                view_proj,
                viewport_w,
                viewport_h,
                out.tile_size);
        }

        out.bins_x = tiled.tiles_x;
        out.bins_y = tiled.tiles_y;
        out.bins_z = 1u;
        out.bin_local_light_lists = tiled.tile_light_lists;
        return out;
    }

    inline std::span<const uint32_t> gather_light_scene_candidates_for_aabb(
        const LightBinCullingData& data,
        const AABB& world_aabb,
        const glm::mat4& view,
        const glm::mat4& view_proj,
        std::vector<uint32_t>& scratch_scene_indices)
    {
        if (!data.has_bins() || data.mode == LightCullingMode::None)
        {
            return fallback_light_scene_candidates(data);
        }

        float min_x = 0.0f;
        float max_x = 0.0f;
        float min_y = 0.0f;
        float max_y = 0.0f;
        float min_depth = data.z_near;
        float max_depth = data.z_far;
        if (!project_aabb_bounds(
                world_aabb,
                view,
                view_proj,
                min_x,
                max_x,
                min_y,
                max_y,
                min_depth,
                max_depth,
                data.z_near,
                data.z_far))
        {
            return fallback_light_scene_candidates(data);
        }

        const uint32_t tx0 = ndc_x_to_bin(min_x, data.bins_x);
        const uint32_t tx1 = ndc_x_to_bin(max_x, data.bins_x);
        const uint32_t ty0 = ndc_y_to_bin_top_origin(max_y, data.bins_y);
        const uint32_t ty1 = ndc_y_to_bin_top_origin(min_y, data.bins_y);

        uint32_t tz0 = 0u;
        uint32_t tz1 = std::max(data.bins_z, 1u) - 1u;
        if (data.mode == LightCullingMode::Clustered && data.bins_z > 1u)
        {
            tz0 = view_depth_to_cluster_slice(min_depth, data.z_near, data.z_far, data.bins_z);
            tz1 = view_depth_to_cluster_slice(max_depth, data.z_near, data.z_far, data.bins_z);
            if (tz0 > tz1) std::swap(tz0, tz1);
        }

        scratch_scene_indices.clear();
        for (uint32_t tz = tz0; tz <= tz1; ++tz)
        {
            for (uint32_t ty = ty0; ty <= ty1; ++ty)
            {
                for (uint32_t tx = tx0; tx <= tx1; ++tx)
                {
                    const uint32_t bin_idx = tz * (data.bins_x * data.bins_y) + ty * data.bins_x + tx;
                    if (bin_idx >= data.bin_local_light_lists.size()) continue;

                    const auto& local_list = data.bin_local_light_lists[bin_idx];
                    for (const uint32_t local_idx : local_list)
                    {
                        if (local_idx >= data.local_to_scene_indices.size()) continue;
                        const uint32_t scene_idx = data.local_to_scene_indices[local_idx];
                        if (std::find(scratch_scene_indices.begin(), scratch_scene_indices.end(), scene_idx) ==
                            scratch_scene_indices.end())
                        {
                            scratch_scene_indices.push_back(scene_idx);
                        }
                    }
                }
            }
        }

        return std::span<const uint32_t>(scratch_scene_indices.data(), scratch_scene_indices.size());
    }
}

#endif // SHS_HAS_JOLT

