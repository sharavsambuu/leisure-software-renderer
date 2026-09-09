#pragma once

/*
    SHS RENDERER SAN

    FILE: vk_render_path_temporal_resources.hpp
    MODULE: rhi/drivers/vulkan
    PURPOSE: Shared Vulkan temporal history-resource ownership helpers for render-path hosts.
*/


#include <algorithm>
#include <cstdint>

#include "shs/pipeline/pass_contract.hpp"
#include "shs/pipeline/render_path_resource_plan.hpp"
#include "shs/rhi/drivers/vulkan/vk_memory_utils.hpp"

#ifdef SHS_HAS_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace shs
{
#ifdef SHS_HAS_VULKAN
    struct VkRenderPathHistoryColorTarget
    {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_UNDEFINED;
        uint32_t width = 0u;
        uint32_t height = 0u;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        bool valid = false;
    };

    struct VkRenderPathTemporalResources
    {
        VkRenderPathHistoryColorTarget history_color{};
        bool history_copy_support_warning_emitted = false;
    };

    inline bool vk_render_path_plan_requires_history_color(const RenderPathResourcePlan& plan)
    {
        const RenderPathResourceSpec* spec =
            find_render_path_resource_by_semantic(plan, PassSemantic::HistoryColor);
        if (!spec) return false;
        if (spec->kind != RenderPathResourceKind::Texture2D) return false;
        return spec->history;
    }

    inline void vk_destroy_render_path_history_color_target(VkDevice device, VkRenderPathHistoryColorTarget& target)
    {
        if (device == VK_NULL_HANDLE) return;
        if (target.view != VK_NULL_HANDLE)
        {
            vkDestroyImageView(device, target.view, nullptr);
            target.view = VK_NULL_HANDLE;
        }
        if (target.image != VK_NULL_HANDLE)
        {
            vkDestroyImage(device, target.image, nullptr);
            target.image = VK_NULL_HANDLE;
        }
        if (target.memory != VK_NULL_HANDLE)
        {
            vkFreeMemory(device, target.memory, nullptr);
            target.memory = VK_NULL_HANDLE;
        }
        target.format = VK_FORMAT_UNDEFINED;
        target.width = 0u;
        target.height = 0u;
        target.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        target.valid = false;
    }

    inline bool vk_create_render_path_history_color_target(
        VkDevice device,
        VkPhysicalDevice physical_device,
        uint32_t width,
        uint32_t height,
        VkFormat format,
        VkRenderPathHistoryColorTarget& target)
    {
        if (device == VK_NULL_HANDLE || physical_device == VK_NULL_HANDLE) return false;
        if (width == 0u || height == 0u || format == VK_FORMAT_UNDEFINED) return false;

        vk_destroy_render_path_history_color_target(device, target);

        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.extent.width = width;
        ici.extent.height = height;
        ici.extent.depth = 1;
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.format = format;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateImage(device, &ici, nullptr, &target.image) != VK_SUCCESS)
        {
            return false;
        }

        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(device, target.image, &req);

        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = vk_find_memory_type(
            physical_device,
            req.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mai.memoryTypeIndex == UINT32_MAX)
        {
            vk_destroy_render_path_history_color_target(device, target);
            return false;
        }
        if (vkAllocateMemory(device, &mai, nullptr, &target.memory) != VK_SUCCESS)
        {
            vk_destroy_render_path_history_color_target(device, target);
            return false;
        }
        if (vkBindImageMemory(device, target.image, target.memory, 0) != VK_SUCCESS)
        {
            vk_destroy_render_path_history_color_target(device, target);
            return false;
        }

        VkImageViewCreateInfo iv{};
        iv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        iv.image = target.image;
        iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
        iv.format = format;
        iv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        iv.subresourceRange.baseMipLevel = 0;
        iv.subresourceRange.levelCount = 1;
        iv.subresourceRange.baseArrayLayer = 0;
        iv.subresourceRange.layerCount = 1;
        if (vkCreateImageView(device, &iv, nullptr, &target.view) != VK_SUCCESS)
        {
            vk_destroy_render_path_history_color_target(device, target);
            return false;
        }

        target.format = format;
        target.width = width;
        target.height = height;
        target.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        target.valid = false;
        return true;
    }

    inline bool vk_render_path_temporal_resources_allocation_equal(
        const VkRenderPathTemporalResources& resources,
        const RenderPathResourcePlan& plan,
        uint32_t frame_width,
        uint32_t frame_height,
        VkFormat color_format)
    {
        const bool requires_history = vk_render_path_plan_requires_history_color(plan);
        if (!requires_history)
        {
            return resources.history_color.image == VK_NULL_HANDLE;
        }

        const VkRenderPathHistoryColorTarget& history = resources.history_color;
        return
            history.image != VK_NULL_HANDLE &&
            history.view != VK_NULL_HANDLE &&
            history.width == frame_width &&
            history.height == frame_height &&
            history.format == color_format;
    }

    inline bool vk_ensure_render_path_temporal_resources(
        VkDevice device,
        VkPhysicalDevice physical_device,
        const RenderPathResourcePlan& plan,
        uint32_t frame_width,
        uint32_t frame_height,
        VkFormat color_format,
        VkRenderPathTemporalResources& resources)
    {
        if (device == VK_NULL_HANDLE || physical_device == VK_NULL_HANDLE) return false;
        if (frame_width == 0u || frame_height == 0u || color_format == VK_FORMAT_UNDEFINED) return false;

        const bool requires_history = vk_render_path_plan_requires_history_color(plan);
        if (!requires_history)
        {
            vk_destroy_render_path_history_color_target(device, resources.history_color);
            return true;
        }

        if (vk_render_path_temporal_resources_allocation_equal(
                resources,
                plan,
                frame_width,
                frame_height,
                color_format))
        {
            return true;
        }

        return vk_create_render_path_history_color_target(
            device,
            physical_device,
            frame_width,
            frame_height,
            color_format,
            resources.history_color);
    }

    inline void vk_destroy_render_path_temporal_resources(
        VkDevice device,
        VkRenderPathTemporalResources& resources)
    {
        vk_destroy_render_path_history_color_target(device, resources.history_color);
        resources.history_copy_support_warning_emitted = false;
    }

    inline bool vk_render_path_supports_swapchain_history_copy(VkImageUsageFlags usage)
    {
        return (usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0;
    }

    inline VkImageView vk_render_path_history_color_view(const VkRenderPathTemporalResources& resources)
    {
        return resources.history_color.view;
    }

    inline bool vk_render_path_history_color_valid(const VkRenderPathTemporalResources& resources)
    {
        return resources.history_color.valid;
    }

    inline void vk_render_path_invalidate_history_color(VkRenderPathTemporalResources& resources)
    {
        resources.history_color.valid = false;
    }

    inline void vk_render_path_mark_history_color_valid(VkRenderPathTemporalResources& resources, bool valid)
    {
        resources.history_color.valid = valid;
    }

    inline void vk_render_path_cmd_image_layout_barrier(
        const VulkanRenderBackend& backend,
        VkCommandBuffer cmd,
        VkImage image,
        VkImageLayout old_layout,
        VkImageLayout new_layout,
        VkAccessFlags src_access,
        VkAccessFlags dst_access,
        VkPipelineStageFlags src_stage,
        VkPipelineStageFlags dst_stage,
        VkImageAspectFlags aspect_mask = VK_IMAGE_ASPECT_COLOR_BIT)
    {
        if (cmd == VK_NULL_HANDLE || image == VK_NULL_HANDLE || old_layout == new_layout) return;

        VkImageSubresourceRange range{};
        range.aspectMask = aspect_mask;
        range.baseMipLevel = 0;
        range.levelCount = 1;
        range.baseArrayLayer = 0;
        range.layerCount = 1;

        backend.transition_image_layout(
            cmd,
            image,
            old_layout,
            new_layout,
            range,
            src_stage,
            src_access,
            dst_stage,
            dst_access);
    }

    inline void vk_render_path_cmd_memory_barrier(
        VkCommandBuffer cmd,
        VkPipelineStageFlags src_stage,
        VkAccessFlags src_access,
        VkPipelineStageFlags dst_stage,
        VkAccessFlags dst_access)
    {
        if (cmd == VK_NULL_HANDLE) return;
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = src_access;
        mb.dstAccessMask = dst_access;
        vkCmdPipelineBarrier(
            cmd,
            src_stage,
            dst_stage,
            0u,
            1u,
            &mb,
            0u,
            nullptr,
            0u,
            nullptr);
    }

    inline bool vk_render_path_record_swapchain_copy_to_shader_read_image(
        const VulkanRenderBackend& backend,
        VkCommandBuffer cmd,
        VkImage swapchain_image,
        VkExtent2D swapchain_extent,
        VkImage dst_image,
        VkExtent2D dst_extent,
        VkImageLayout dst_current_layout,
        VkAccessFlags dst_current_access,
        VkPipelineStageFlags dst_current_stage)
    {
        if (cmd == VK_NULL_HANDLE || swapchain_image == VK_NULL_HANDLE || dst_image == VK_NULL_HANDLE) return false;
        if (dst_extent.width == 0u || dst_extent.height == 0u) return false;

        vk_render_path_cmd_image_layout_barrier(
            backend,
            cmd,
            swapchain_image,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);

        vk_render_path_cmd_image_layout_barrier(
            backend,
            cmd,
            dst_image,
            dst_current_layout,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            dst_current_access,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            dst_current_stage,
            VK_PIPELINE_STAGE_TRANSFER_BIT);

        VkImageCopy copy{};
        copy.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.srcSubresource.mipLevel = 0;
        copy.srcSubresource.baseArrayLayer = 0;
        copy.srcSubresource.layerCount = 1;
        copy.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.dstSubresource.mipLevel = 0;
        copy.dstSubresource.baseArrayLayer = 0;
        copy.dstSubresource.layerCount = 1;
        copy.extent.width = std::min(swapchain_extent.width, dst_extent.width);
        copy.extent.height = std::min(swapchain_extent.height, dst_extent.height);
        copy.extent.depth = 1;
        if (copy.extent.width == 0u || copy.extent.height == 0u) return false;
        vkCmdCopyImage(
            cmd,
            swapchain_image,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            dst_image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &copy);

        vk_render_path_cmd_image_layout_barrier(
            backend,
            cmd,
            dst_image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

        vk_render_path_cmd_image_layout_barrier(
            backend,
            cmd,
            swapchain_image,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_ACCESS_TRANSFER_READ_BIT,
            0u,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
        return true;
    }

    inline bool vk_render_path_record_swapchain_copy_to_host_buffer(
        const VulkanRenderBackend& backend,
        VkCommandBuffer cmd,
        VkImage swapchain_image,
        VkExtent2D swapchain_extent,
        VkBuffer dst_buffer)
    {
        if (cmd == VK_NULL_HANDLE || swapchain_image == VK_NULL_HANDLE || dst_buffer == VK_NULL_HANDLE) return false;
        if (swapchain_extent.width == 0u || swapchain_extent.height == 0u) return false;

        vk_render_path_cmd_image_layout_barrier(
            backend,
            cmd,
            swapchain_image,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);

        VkBufferImageCopy copy_region{};
        copy_region.bufferOffset = 0u;
        copy_region.bufferRowLength = 0u;
        copy_region.bufferImageHeight = 0u;
        copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy_region.imageSubresource.mipLevel = 0u;
        copy_region.imageSubresource.baseArrayLayer = 0u;
        copy_region.imageSubresource.layerCount = 1u;
        copy_region.imageOffset = {0, 0, 0};
        copy_region.imageExtent = {swapchain_extent.width, swapchain_extent.height, 1u};
        vkCmdCopyImageToBuffer(
            cmd,
            swapchain_image,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            dst_buffer,
            1u,
            &copy_region);

        vk_render_path_cmd_memory_barrier(
            cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_ACCESS_HOST_READ_BIT);

        vk_render_path_cmd_image_layout_barrier(
            backend,
            cmd,
            swapchain_image,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_ACCESS_TRANSFER_READ_BIT,
            0u,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
        return true;
    }

    inline void vk_render_path_ensure_history_color_shader_read_layout(
        const VulkanRenderBackend& backend,
        VkCommandBuffer cmd,
        VkRenderPathTemporalResources& resources)
    {
        VkRenderPathHistoryColorTarget& history = resources.history_color;
        if (history.image == VK_NULL_HANDLE) return;
        if (history.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) return;

        const VkAccessFlags src_access =
            (history.layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
                ? static_cast<VkAccessFlags>(VK_ACCESS_TRANSFER_WRITE_BIT)
                : static_cast<VkAccessFlags>(0u);
        const VkPipelineStageFlags src_stage =
            (history.layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
                ? VK_PIPELINE_STAGE_TRANSFER_BIT
                : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

        vk_render_path_cmd_image_layout_barrier(
            backend,
            cmd,
            history.image,
            history.layout,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            src_access,
            VK_ACCESS_SHADER_READ_BIT,
            src_stage,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        history.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    inline bool vk_render_path_record_history_color_copy(
        const VulkanRenderBackend& backend,
        VkCommandBuffer cmd,
        VkImage swapchain_image,
        VkExtent2D swapchain_extent,
        VkRenderPathTemporalResources& resources)
    {
        if (cmd == VK_NULL_HANDLE || swapchain_image == VK_NULL_HANDLE) return false;
        VkRenderPathHistoryColorTarget& history = resources.history_color;
        if (history.image == VK_NULL_HANDLE) return false;

        const bool history_shader_read_layout =
            (history.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        const VkAccessFlags history_src_access = history_shader_read_layout
            ? static_cast<VkAccessFlags>(VK_ACCESS_SHADER_READ_BIT)
            : static_cast<VkAccessFlags>(0u);
        const VkPipelineStageFlags history_src_stage = history_shader_read_layout
            ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
            : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        if (!vk_render_path_record_swapchain_copy_to_shader_read_image(
                backend,
                cmd,
                swapchain_image,
                swapchain_extent,
                history.image,
                VkExtent2D{history.width, history.height},
                history.layout,
                history_src_access,
                history_src_stage))
        {
            return false;
        }
        history.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        history.valid = true;
        return true;
    }
#endif
}
