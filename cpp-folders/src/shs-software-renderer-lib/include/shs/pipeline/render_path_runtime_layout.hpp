#pragma once

/*
    SHS RENDERER SAN

    FILE: render_path_runtime_layout.hpp
    MODULE: pipeline
    PURPOSE: Runtime allocation/layout helpers derived from recipe + resource plan.
*/


#include <algorithm>
#include <cstdint>
#include <string_view>

#include "shs/pipeline/render_path_compiler.hpp"
#include "shs/pipeline/render_path_recipe.hpp"
#include "shs/pipeline/render_path_resource_plan.hpp"

namespace shs
{
    struct RenderPathLightGridRuntimeLayout
    {
        uint32_t frame_width = 0u;
        uint32_t frame_height = 0u;
        uint32_t tile_size = 16u;
        uint32_t tile_count_x = 0u;
        uint32_t tile_count_y = 0u;
        uint32_t cluster_z_slices = 1u;
        uint64_t tile_count = 0u;
        uint64_t cluster_count = 0u;
        uint64_t list_count = 1u;
        bool uses_light_grid = false;
        bool uses_light_clusters = false;
        bool valid = false;
    };

    struct RenderPathLightGridBufferSizes
    {
        uint64_t counts_bytes = 0u;
        uint64_t indices_bytes = 0u;
        uint64_t depth_ranges_bytes = 0u;
    };

    inline bool render_path_plan_has_pass(const RenderPathExecutionPlan& plan, std::string_view pass_id)
    {
        const PassId pid = parse_pass_id(pass_id);
        for (const auto& pass : plan.pass_chain)
        {
            if (pass.id == pass_id) return true;
            if (pass_id_is_standard(pid) &&
                (pass.pass_id == pid || parse_pass_id(pass.id) == pid))
            {
                return true;
            }
        }
        return false;
    }

    inline bool render_path_plan_has_pass(const RenderPathExecutionPlan& plan, PassId pass_id)
    {
        if (!pass_id_is_standard(pass_id)) return false;
        for (const auto& pass : plan.pass_chain)
        {
            if (pass.pass_id == pass_id) return true;
            if (parse_pass_id(pass.id) == pass_id) return true;
        }
        return false;
    }

    inline RenderPathLightGridRuntimeLayout make_render_path_light_grid_runtime_layout(
        const RenderPathExecutionPlan& plan,
        const RenderPathRecipe& recipe,
        const RenderPathResourcePlan& resource_plan,
        uint32_t frame_width,
        uint32_t frame_height)
    {
        RenderPathLightGridRuntimeLayout out{};
        out.frame_width = frame_width;
        out.frame_height = frame_height;

        const RenderPathResourceSpec* grid =
            find_render_path_resource_by_semantic(resource_plan, PassSemantic::LightGrid);
        const RenderPathResourceSpec* clusters =
            find_render_path_resource_by_semantic(resource_plan, PassSemantic::LightClusters);

        const bool culling_pass_present =
            render_path_plan_has_pass(plan, PassId::LightCulling) ||
            render_path_plan_has_pass(plan, PassId::ClusterBuild) ||
            render_path_plan_has_pass(plan, PassId::ClusterLightAssign);

        out.uses_light_grid = (grid != nullptr) || culling_pass_present;
        out.uses_light_clusters = (clusters != nullptr) ||
                                  render_path_plan_has_pass(plan, PassId::ClusterBuild) ||
                                  render_path_plan_has_pass(plan, PassId::ClusterLightAssign);

        out.tile_size = std::max(1u, recipe.light_tile_size);
        if (grid) out.tile_size = std::max(1u, grid->tile_size);

        out.cluster_z_slices = std::max(1u, recipe.cluster_z_slices);
        if (clusters) out.cluster_z_slices = std::max(1u, clusters->layers);

        if (frame_width == 0u || frame_height == 0u)
        {
            out.valid = false;
            out.list_count = 1u;
            return out;
        }

        out.tile_count_x = std::max(1u, (frame_width + out.tile_size - 1u) / out.tile_size);
        out.tile_count_y = std::max(1u, (frame_height + out.tile_size - 1u) / out.tile_size);
        out.tile_count = static_cast<uint64_t>(out.tile_count_x) * static_cast<uint64_t>(out.tile_count_y);
        out.cluster_count = out.uses_light_clusters ? (out.tile_count * static_cast<uint64_t>(out.cluster_z_slices))
                                                    : out.tile_count;
        out.list_count = std::max<uint64_t>(1u, std::max(out.tile_count, out.cluster_count));
        out.valid = true;
        return out;
    }

    inline RenderPathLightGridBufferSizes make_render_path_light_grid_buffer_sizes(
        const RenderPathLightGridRuntimeLayout& layout,
        uint32_t max_lights_per_list)
    {
        RenderPathLightGridBufferSizes out{};
        const uint64_t list_count = std::max<uint64_t>(1u, layout.list_count);
        const uint64_t tile_count = std::max<uint64_t>(1u, layout.tile_count);
        const uint64_t per_list = static_cast<uint64_t>(std::max(1u, max_lights_per_list));

        out.counts_bytes = list_count * sizeof(uint32_t);
        out.indices_bytes = out.counts_bytes * per_list;
        out.depth_ranges_bytes = tile_count * (sizeof(float) * 2u);
        return out;
    }

    inline bool light_grid_runtime_layout_allocation_equal(
        const RenderPathLightGridRuntimeLayout& a,
        const RenderPathLightGridRuntimeLayout& b)
    {
        return
            a.frame_width == b.frame_width &&
            a.frame_height == b.frame_height &&
            a.tile_size == b.tile_size &&
            a.tile_count_x == b.tile_count_x &&
            a.tile_count_y == b.tile_count_y &&
            a.cluster_z_slices == b.cluster_z_slices &&
            a.tile_count == b.tile_count &&
            a.cluster_count == b.cluster_count &&
            a.list_count == b.list_count &&
            a.uses_light_grid == b.uses_light_grid &&
            a.uses_light_clusters == b.uses_light_clusters;
    }
}
