#pragma once

/*
    SHS RENDERER SAN

    FILE: vk_shape_culler.hpp
    MODULE: rhi/drivers/vulkan
    PURPOSE: Backward-compatible include wrapper.
             New code should include vk_shape_cell_culler.hpp.
*/

#include "shs/rhi/drivers/vulkan/vk_shape_cell_culler.hpp"

namespace shs
{
#ifdef SHS_HAS_VULKAN
    inline constexpr uint32_t k_vk_shape_culler_group_size_x = k_vk_shape_cell_culler_group_size_x;
    inline constexpr uint32_t k_vk_shape_culler_set_index = k_vk_shape_cell_culler_set_index;
    inline constexpr uint32_t k_vk_shape_culler_binding_shapes = k_vk_shape_cell_culler_binding_shapes;
    inline constexpr uint32_t k_vk_shape_culler_binding_cells = k_vk_shape_cell_culler_binding_cells;
    inline constexpr uint32_t k_vk_shape_culler_binding_jobs = k_vk_shape_cell_culler_binding_jobs;
    inline constexpr uint32_t k_vk_shape_culler_binding_results = k_vk_shape_cell_culler_binding_results;
    inline constexpr uint32_t k_vk_shape_culler_binding_aux_vertices = k_vk_shape_cell_culler_binding_aux_vertices;
    inline constexpr uint32_t k_vk_shape_payload_flag_has_aux_vertices = k_vk_shape_cell_payload_flag_has_aux_vertices;
    inline constexpr uint32_t k_vk_shape_payload_flag_broad_fallback = k_vk_shape_cell_payload_flag_broad_fallback;

    inline uint32_t vk_shape_culler_dispatch_groups(uint32_t job_count)
    {
        return vk_shape_cell_culler_dispatch_groups(job_count);
    }

    using VkShapeCullerPushConstants = VkShapeCellCullerPushConstants;
    using VkShapeCullerPipeline = VkShapeCellCullerPipeline;

    inline VkShapeCullerPushConstants vk_make_shape_culler_push_constants(
        uint32_t job_count,
        uint32_t shape_count,
        uint32_t cell_count,
        float outside_eps = 1e-5f,
        float inside_eps = 1e-5f,
        uint32_t flags = 0u)
    {
        return vk_make_shape_cell_culler_push_constants(
            job_count,
            shape_count,
            cell_count,
            outside_eps,
            inside_eps,
            flags);
    }

    inline void vk_destroy_shape_culler_pipeline(VkDevice device, VkShapeCullerPipeline& pipeline)
    {
        vk_destroy_shape_cell_culler_pipeline(device, pipeline);
    }

    inline bool vk_create_shape_culler_pipeline(
        VkDevice device,
        VkShaderModule compute_shader_module,
        VkShapeCullerPipeline& out_pipeline)
    {
        return vk_create_shape_cell_culler_pipeline(device, compute_shader_module, out_pipeline);
    }

    inline void vk_cmd_dispatch_shape_culler(
        VkCommandBuffer cmd,
        const VkShapeCullerPipeline& pipeline,
        VkDescriptorSet descriptor_set,
        const VkShapeCullerPushConstants& push)
    {
        vk_cmd_dispatch_shape_cell_culler(cmd, pipeline, descriptor_set, push);
    }
#else
    struct VkShapeCullerPushConstants {};
    struct VkShapeCullerPipeline {};
#endif
}
