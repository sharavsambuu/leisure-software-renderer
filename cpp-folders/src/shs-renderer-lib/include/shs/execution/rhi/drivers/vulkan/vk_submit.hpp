#pragma once

#include <expected>
#include <vulkan/vulkan.h>

namespace shs
{
    enum class VulkanSubmitStage { Prerequisite, FenceCreation, Submission, Completion };

    struct VulkanSubmitFailure
    {
        VulkanSubmitStage stage;
        VkResult result;
    };

    // Borrowed executable primary command buffer and compatible queue. Caller
    // externally serializes queue access. Success means execution is complete;
    // this function neither begins/ends recording nor resets the command buffer.
    [[nodiscard]] inline std::expected<void, VulkanSubmitFailure> vulkan_submit_sync(
        VkDevice device, VkQueue queue, VkCommandBuffer command)
    {
        if (!device || !queue || !command)
            return std::unexpected(VulkanSubmitFailure{
                VulkanSubmitStage::Prerequisite, VK_ERROR_INITIALIZATION_FAILED});
        VkFenceCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = VK_NULL_HANDLE;
        auto result = vkCreateFence(device, &info, nullptr, &fence);
        if (result != VK_SUCCESS)
            return std::unexpected(VulkanSubmitFailure{VulkanSubmitStage::FenceCreation, result});
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        result = vkQueueSubmit(queue, 1, &submit, fence);
        auto stage = VulkanSubmitStage::Submission;
        if (result == VK_SUCCESS)
        {
            stage = VulkanSubmitStage::Completion;
            result = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
            // A failed wait is not proof of retirement. Drain before releasing
            // the fence; device loss remains an error, never a successful frame.
            if (result != VK_SUCCESS) vkDeviceWaitIdle(device);
        }
        vkDestroyFence(device, fence, nullptr);
        if (result != VK_SUCCESS)
            return std::unexpected(VulkanSubmitFailure{stage, result});
        return {};
    }
}
