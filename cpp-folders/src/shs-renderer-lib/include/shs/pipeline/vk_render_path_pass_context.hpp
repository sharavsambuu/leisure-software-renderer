#pragma once

/*
    SHS RENDERER SAN

    FILE: vk_render_path_pass_context.hpp
    MODULE: pipeline
    PURPOSE: Shared Vulkan pass-dispatch context contract used by render-path hosts.
*/


#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

namespace shs
{
    template <typename TFrameInfo>
    struct VkRenderPathPassExecutionContext
    {
        TFrameInfo* fi = nullptr;
        uint32_t frame_slot = 0u;
        VkDescriptorSet global_set = VK_NULL_HANDLE;
        const std::vector<VkCommandBuffer>* depth_secondaries = nullptr;
        const std::vector<VkCommandBuffer>* scene_secondaries = nullptr;

        bool depth_prepass_enabled = false;
        bool scene_enabled = false;
        bool light_culling_enabled = false;
        bool gpu_light_culler_enabled = false;

        bool depth_pass_executed = false;
        bool light_culling_executed = false;
        bool gbuffer_pass_executed = false;
        bool ssao_pass_executed = false;
        bool deferred_lighting_pass_executed = false;
        bool motion_blur_pass_executed = false;
        bool depth_of_field_pass_executed = false;
        bool deferred_emulated_scene_pass = false;
        bool taa_pass_executed = false;
        bool scene_pass_executed = false;
        bool light_grid_cleared = false;

        bool has_motion_blur_pass = false;
        bool has_depth_of_field_pass = false;
        bool post_color_valid = false;
        uint32_t post_color_source = 0u; // 0:none, 1:post_a, 2:post_b
    };
}
