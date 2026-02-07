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

#include "shs/rhi/drivers/vulkan/vk_backend.hpp"

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
            if ((int)frame.extent.width != width || (int)frame.extent.height != height) return false;

            const size_t bytes = (size_t)width * (size_t)height * 4u;
            if (!ensure_staging(backend, bytes)) return false;
            if (!mapped_ptr_) return false;

            auto* dst = static_cast<uint8_t*>(mapped_ptr_);
            for (int y = 0; y < height; ++y)
            {
                std::memcpy(dst + (size_t)y * (size_t)width * 4u, src_rgba8 + (size_t)y * (size_t)src_pitch_bytes, (size_t)width * 4u);
            }

            VkImage swap_img = backend.swapchain_image(frame.image_index);
            VkImageMemoryBarrier to_transfer{};
            to_transfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            to_transfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            to_transfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to_transfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_transfer.image = swap_img;
            to_transfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            to_transfer.subresourceRange.baseMipLevel = 0;
            to_transfer.subresourceRange.levelCount = 1;
            to_transfer.subresourceRange.baseArrayLayer = 0;
            to_transfer.subresourceRange.layerCount = 1;
            to_transfer.srcAccessMask = 0;
            to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(
                frame.cmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0, nullptr,
                0, nullptr,
                1, &to_transfer
            );

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
                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                0,
                0, nullptr,
                0, nullptr,
                1, &to_present
            );
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
        }

    private:
#ifdef SHS_HAS_VULKAN
        static uint32_t find_memory_type(VkPhysicalDevice gpu, uint32_t type_bits, VkMemoryPropertyFlags props)
        {
            VkPhysicalDeviceMemoryProperties mp{};
            vkGetPhysicalDeviceMemoryProperties(gpu, &mp);
            for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            {
                const bool matches_type = (type_bits & (1u << i)) != 0;
                const bool matches_props = (mp.memoryTypes[i].propertyFlags & props) == props;
                if (matches_type && matches_props) return i;
            }
            return UINT32_MAX;
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

            VkBufferCreateInfo bci{};
            bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bci.size = bytes;
            bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            if (vkCreateBuffer(dev, &bci, nullptr, &staging_buffer_) != VK_SUCCESS) return false;

            VkMemoryRequirements req{};
            vkGetBufferMemoryRequirements(dev, staging_buffer_, &req);
            const uint32_t memory_type = find_memory_type(gpu, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (memory_type == UINT32_MAX) return false;

            VkMemoryAllocateInfo mai{};
            mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            mai.allocationSize = req.size;
            mai.memoryTypeIndex = memory_type;
            if (vkAllocateMemory(dev, &mai, nullptr, &staging_memory_) != VK_SUCCESS) return false;
            if (vkBindBufferMemory(dev, staging_buffer_, staging_memory_, 0) != VK_SUCCESS) return false;
            if (vkMapMemory(dev, staging_memory_, 0, bytes, 0, &mapped_ptr_) != VK_SUCCESS) return false;

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
    };
}
