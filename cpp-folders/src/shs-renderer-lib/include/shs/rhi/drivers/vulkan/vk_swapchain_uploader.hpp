#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: vk_swapchain_uploader.hpp
    МОДУЛЬ: rhi/drivers/vulkan
    ЗОРИЛГО: CPU RGBA8 буферыг swapchain image рүү transfer copy хийж present хийх туслах utility.
*/


#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "shs/rhi/drivers/vulkan/vk_backend.hpp"
#include "shs/rhi/drivers/vulkan/vk_memory_utils.hpp"

namespace shs
{
    class VulkanSwapchainUploader
    {
    public:
        ~VulkanSwapchainUploader()
        {
            shutdown();
        }

        bool record_upload_rgba8(
            VulkanRenderBackend& backend,
            const VulkanRenderBackend::FrameInfo& frame,
            const uint8_t* src_rgba8,
            int width,
            int height,
            int src_pitch_bytes
        )
        {
#ifdef SHS_HAS_VULKAN
            if (!src_rgba8 || width <= 0 || height <= 0) return false;
            if (src_pitch_bytes < width * 4) return false;
            if (frame.cmd == VK_NULL_HANDLE) return false;
            if (frame.extent.width == 0 || frame.extent.height == 0) return false;
            if (backend.swapchain_image(frame.image_index) == VK_NULL_HANDLE) return false;
            if ((backend.swapchain_usage_flags() & VK_IMAGE_USAGE_TRANSFER_DST_BIT) == 0) return false;
            if ((uint32_t)width > frame.extent.width || (uint32_t)height > frame.extent.height) return false;
            if (!ensure_swapchain_state(backend, frame.image_index)) return false;

            const size_t bytes = (size_t)width * (size_t)height * 4u;
            if (!ensure_staging(backend, bytes)) return false;
            if (!mapped_ptr_) return false;

            auto* dst = static_cast<uint8_t*>(mapped_ptr_);
            for (int y = 0; y < height; ++y)
            {
                std::memcpy(dst + (size_t)y * (size_t)width * 4u, src_rgba8 + (size_t)y * (size_t)src_pitch_bytes, (size_t)width * 4u);
            }

            VkImage swap_img = backend.swapchain_image(frame.image_index);
            const bool image_initialized = (image_initialized_[frame.image_index] != 0);
            const bool full_overwrite = (uint32_t)width == frame.extent.width && (uint32_t)height == frame.extent.height;

            VkImageMemoryBarrier to_transfer{};
            to_transfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            to_transfer.oldLayout = image_initialized ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : VK_IMAGE_LAYOUT_UNDEFINED;
            to_transfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to_transfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_transfer.image = swap_img;
            to_transfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            to_transfer.subresourceRange.baseMipLevel = 0;
            to_transfer.subresourceRange.levelCount = 1;
            to_transfer.subresourceRange.baseArrayLayer = 0;
            to_transfer.subresourceRange.layerCount = 1;
            to_transfer.srcAccessMask = image_initialized ? VK_ACCESS_MEMORY_READ_BIT : 0;
            to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            const VkPipelineStageFlags to_transfer_src_stage =
                image_initialized ? VK_PIPELINE_STAGE_ALL_COMMANDS_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            vkCmdPipelineBarrier(
                frame.cmd,
                to_transfer_src_stage,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0, nullptr,
                0, nullptr,
                1, &to_transfer
            );

            if (!image_initialized && !full_overwrite)
            {
                VkImageSubresourceRange clear_range{};
                clear_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                clear_range.baseMipLevel = 0;
                clear_range.levelCount = 1;
                clear_range.baseArrayLayer = 0;
                clear_range.layerCount = 1;
                VkClearColorValue clear_color{};
                clear_color.float32[0] = 0.0f;
                clear_color.float32[1] = 0.0f;
                clear_color.float32[2] = 0.0f;
                clear_color.float32[3] = 1.0f;
                vkCmdClearColorImage(
                    frame.cmd,
                    swap_img,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    &clear_color,
                    1,
                    &clear_range
                );
            }

            VkBufferImageCopy copy{};
            copy.bufferOffset = 0;
            copy.bufferRowLength = (uint32_t)width;
            copy.bufferImageHeight = (uint32_t)height;
            copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copy.imageSubresource.mipLevel = 0;
            copy.imageSubresource.baseArrayLayer = 0;
            copy.imageSubresource.layerCount = 1;
            copy.imageOffset = VkOffset3D{0, 0, 0};
            copy.imageExtent = VkExtent3D{(uint32_t)width, (uint32_t)height, 1u};
            vkCmdCopyBufferToImage(
                frame.cmd,
                staging_buffer_,
                swap_img,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &copy
            );

            VkImageMemoryBarrier to_present{};
            to_present.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            to_present.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to_present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            to_present.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_present.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_present.image = swap_img;
            to_present.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            to_present.subresourceRange.baseMipLevel = 0;
            to_present.subresourceRange.levelCount = 1;
            to_present.subresourceRange.baseArrayLayer = 0;
            to_present.subresourceRange.layerCount = 1;
            to_present.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            to_present.dstAccessMask = 0;
            vkCmdPipelineBarrier(
                frame.cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                0,
                0, nullptr,
                0, nullptr,
                1, &to_present
            );
            image_initialized_[frame.image_index] = 1;
            return true;
#else
            (void)backend;
            (void)frame;
            (void)src_rgba8;
            (void)width;
            (void)height;
            (void)src_pitch_bytes;
            return false;
#endif
        }

        void shutdown()
        {
#ifdef SHS_HAS_VULKAN
            if (mapped_ptr_ && mapped_device_ != VK_NULL_HANDLE && staging_memory_ != VK_NULL_HANDLE)
            {
                vkUnmapMemory(mapped_device_, staging_memory_);
            }
            mapped_ptr_ = nullptr;
            if (mapped_device_ != VK_NULL_HANDLE && staging_buffer_ != VK_NULL_HANDLE)
            {
                vkDestroyBuffer(mapped_device_, staging_buffer_, nullptr);
            }
            if (mapped_device_ != VK_NULL_HANDLE && staging_memory_ != VK_NULL_HANDLE)
            {
                vkFreeMemory(mapped_device_, staging_memory_, nullptr);
            }
#endif
            staging_buffer_ = 0;
            staging_memory_ = 0;
            mapped_device_ = 0;
            staging_bytes_ = 0;
            tracked_swapchain_generation_ = 0;
            image_initialized_.clear();
        }

    private:
#ifdef SHS_HAS_VULKAN
        bool ensure_swapchain_state(const VulkanRenderBackend& backend, uint32_t image_index)
        {
            const uint64_t generation = backend.swapchain_generation();
            if (tracked_swapchain_generation_ != generation)
            {
                tracked_swapchain_generation_ = generation;
                image_initialized_.clear();
            }
            if (image_index >= image_initialized_.size())
            {
                image_initialized_.resize(image_index + 1u, 0u);
            }
            return true;
        }

        bool ensure_staging(VulkanRenderBackend& backend, size_t bytes)
        {
            VkDevice dev = backend.device();
            VkPhysicalDevice gpu = backend.physical_device();
            if (dev == VK_NULL_HANDLE || gpu == VK_NULL_HANDLE) return false;

            if (mapped_device_ == dev && staging_buffer_ != VK_NULL_HANDLE && staging_memory_ != VK_NULL_HANDLE && staging_bytes_ >= bytes)
            {
                return true;
            }

            shutdown();

            if (!vk_create_buffer(
                    dev,
                    gpu,
                    static_cast<VkDeviceSize>(bytes),
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    staging_buffer_,
                    staging_memory_))
            {
                return false;
            }
            if (vkMapMemory(dev, staging_memory_, 0, bytes, 0, &mapped_ptr_) != VK_SUCCESS)
            {
                shutdown();
                return false;
            }

            mapped_device_ = dev;
            staging_bytes_ = bytes;
            return true;
        }

        VkBuffer staging_buffer_ = VK_NULL_HANDLE;
        VkDeviceMemory staging_memory_ = VK_NULL_HANDLE;
        VkDevice mapped_device_ = VK_NULL_HANDLE;
        void* mapped_ptr_ = nullptr;
#else
        uint64_t staging_buffer_ = 0;
        uint64_t staging_memory_ = 0;
        uint64_t mapped_device_ = 0;
#endif
        size_t staging_bytes_ = 0;
        uint64_t tracked_swapchain_generation_ = 0;
        std::vector<uint8_t> image_initialized_{};
    };
}
