#pragma once

#include <cstring>
#include <expected>
#include <span>
#include "shs/rhi/vulkan/value/vk_resources.hpp"
#include "shs/rhi/vulkan/value/vk_commands.hpp"
#include "shs/rhi/vulkan/value/vk_submit.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace rhi
    {
    enum class VulkanExecutionError { NotPrepared, InvalidDescriptor, CreationFailed, RecordingFailed, SubmissionFailed, ReadbackFailed, UploadFailed };
    struct VulkanExecutionFailure
    {
        VulkanExecutionError code;
        VkResult result = VK_SUCCESS;
        VulkanRecordingFailure recording{VulkanRecordingError::InvalidCommand};
    };

    // Synchronous transfer owner. Device must outlive this object. No concurrent
    // use; queue synchronization is the caller's responsibility.
    class VulkanReadback
    {
    public:
        VulkanReadback() = default;
        ~VulkanReadback() { reset(); }
        VulkanReadback(const VulkanReadback&) = delete;
        VulkanReadback& operator=(const VulkanReadback&) = delete;

        [[nodiscard]] bool initialize(VulkanDeviceManager& device, VkExtent2D extent)
        {
            if (device_ || !device.device_available() || !extent.width || !extent.height ||
                uint64_t(extent.width) * extent.height > SIZE_MAX / 4) return false;
            device_ = device.device();
            queue_ = device.graphics_queue();
            extent_ = extent;
            bytes_ = uint64_t(extent.width) * extent.height * 4;
            const auto fail = [&] { reset(); return false; };
            VkCommandPoolCreateInfo pool{};
            pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pool.queueFamilyIndex = device.info().graphics_queue_family;
            if (vkCreateCommandPool(device_, &pool, nullptr, &pool_) != VK_SUCCESS) return fail();
            VkCommandBufferAllocateInfo command{};
            command.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            command.commandPool = pool_;
            command.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            command.commandBufferCount = 1;
            if (vkAllocateCommandBuffers(device_, &command, &command_) != VK_SUCCESS) return fail();
            VkBufferCreateInfo buffer{};
            buffer.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            buffer.size = bytes_;
            buffer.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            if (vkCreateBuffer(device_, &buffer, nullptr, &buffer_) != VK_SUCCESS) return fail();
            VkMemoryRequirements req{};
            vkGetBufferMemoryRequirements(device_, buffer_, &req);
            VkPhysicalDeviceMemoryProperties props{};
            vkGetPhysicalDeviceMemoryProperties(device.physical(), &props);
            const auto type = vk_pick_memory_type(props, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
            if (type == UINT32_MAX) return fail();
            VkMemoryAllocateInfo allocation{};
            allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocation.allocationSize = req.size;
            allocation.memoryTypeIndex = type;
            if (vkAllocateMemory(device_, &allocation, nullptr, &memory_) != VK_SUCCESS) return fail();
            if (vkBindBufferMemory(device_, buffer_, memory_, 0) != VK_SUCCESS) return fail();
            return true;
        }

        void reset()
        {
            if (pool_) vkDestroyCommandPool(device_, pool_, nullptr);
            if (buffer_) vkDestroyBuffer(device_, buffer_, nullptr);
            if (memory_) vkFreeMemory(device_, memory_, nullptr);
            pool_ = VK_NULL_HANDLE; command_ = VK_NULL_HANDLE;
            buffer_ = VK_NULL_HANDLE; memory_ = VK_NULL_HANDLE;
            device_ = VK_NULL_HANDLE; queue_ = VK_NULL_HANDLE;
            extent_ = {}; bytes_ = 0;
        }

        // RecordFn emits a complete pass ending in COLOR_ATTACHMENT_OPTIMAL.
        template <typename RecordFn>
        [[nodiscard]] std::expected<void, VulkanExecutionFailure> execute(
            VkImage image, std::span<uint8_t> output, RecordFn&& record)
        {
            if (!device_) return std::unexpected(VulkanExecutionFailure{VulkanExecutionError::NotPrepared});
            if (!image || output.size() != bytes_)
                return std::unexpected(VulkanExecutionFailure{VulkanExecutionError::InvalidDescriptor});
            const auto fail = [](VkResult result) {
                return std::unexpected(VulkanExecutionFailure{VulkanExecutionError::ReadbackFailed, result});
            };
            auto result = vkResetCommandPool(device_, pool_, 0);
            if (result != VK_SUCCESS) return fail(result);
            VkCommandBufferBeginInfo begin{};
            begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            result = vkBeginCommandBuffer(command_, &begin);
            if (result != VK_SUCCESS) return fail(result);
            if (auto recorded = record(command_); !recorded)
                return std::unexpected(VulkanExecutionFailure{
                    VulkanExecutionError::RecordingFailed, VK_SUCCESS, recorded.error()});
            VkImageMemoryBarrier image_barrier{};
            image_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            image_barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            image_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            image_barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            image_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            image_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            image_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            image_barrier.image = image;
            image_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdPipelineBarrier(command_, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &image_barrier);
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {extent_.width, extent_.height, 1};
            vkCmdCopyImageToBuffer(command_, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer_, 1, &copy);
            VkMemoryBarrier host{};
            host.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(command_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                0, 1, &host, 0, nullptr, 0, nullptr);
            result = vkEndCommandBuffer(command_);
            if (result != VK_SUCCESS) return fail(result);
            if (auto submitted = vulkan_submit_sync(device_, queue_, command_); !submitted)
                return std::unexpected(VulkanExecutionFailure{
                    VulkanExecutionError::SubmissionFailed, submitted.error().result});
            void* mapped = nullptr;
            result = vkMapMemory(device_, memory_, 0, VK_WHOLE_SIZE, 0, &mapped);
            if (result != VK_SUCCESS) return fail(result);
            // Whole allocation avoids nonCoherentAtomSize alignment pitfalls.
            VkMappedMemoryRange range{};
            range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            range.memory = memory_;
            range.size = VK_WHOLE_SIZE;
            result = vkInvalidateMappedMemoryRanges(device_, 1, &range);
            if (result == VK_SUCCESS) std::memcpy(output.data(), mapped, output.size());
            vkUnmapMemory(device_, memory_);
            if (result != VK_SUCCESS) return fail(result);
            return {};
        }

    private:
        VkDevice device_ = VK_NULL_HANDLE;
        VkQueue queue_ = VK_NULL_HANDLE;
        VkCommandPool pool_ = VK_NULL_HANDLE;
        VkCommandBuffer command_ = VK_NULL_HANDLE;
        VkBuffer buffer_ = VK_NULL_HANDLE;
        VkDeviceMemory memory_ = VK_NULL_HANDLE;
        VkExtent2D extent_{};
        uint64_t bytes_ = 0;
    };

    } // inline namespace rhi
}
