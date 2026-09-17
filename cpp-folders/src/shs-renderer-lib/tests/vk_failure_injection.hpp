#pragma once

// Test-only interposition: real Vulkan runs except the selected call. Include
// before driver headers. Never inject a fake successful handle or device loss.
#include <vulkan/vulkan.h>

namespace vk_test
{
    enum class Fault { None, Allocation, Pipeline, Fence, Submit, Map };
    inline Fault fault = Fault::None;
    inline unsigned countdown = 0;
    inline bool triggered = false;
    inline void arm(Fault value, unsigned nth = 1)
    {
        fault = value; countdown = nth; triggered = false;
    }
    inline bool fail(Fault value)
    {
        if (fault != value || !countdown || --countdown) return false;
        triggered = true; fault = Fault::None; return true;
    }
    inline VkResult allocate(VkDevice d, const VkMemoryAllocateInfo* i, const VkAllocationCallbacks* a, VkDeviceMemory* m)
    {
        if (fail(Fault::Allocation)) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        return vkAllocateMemory(d, i, a, m);
    }
    inline VkResult pipeline(VkDevice d, VkPipelineCache c, uint32_t n,
        const VkGraphicsPipelineCreateInfo* i, const VkAllocationCallbacks* a, VkPipeline* p)
    {
        if (fail(Fault::Pipeline)) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        return vkCreateGraphicsPipelines(d, c, n, i, a, p);
    }
    inline VkResult fence(VkDevice d, const VkFenceCreateInfo* i, const VkAllocationCallbacks* a, VkFence* f)
    {
        if (fail(Fault::Fence)) return VK_ERROR_OUT_OF_HOST_MEMORY;
        return vkCreateFence(d, i, a, f);
    }
    inline VkResult submit(VkQueue q, uint32_t n, const VkSubmitInfo* i, VkFence f)
    {
        if (fail(Fault::Submit)) return VK_ERROR_OUT_OF_HOST_MEMORY;
        return vkQueueSubmit(q, n, i, f);
    }
    inline VkResult map(VkDevice d, VkDeviceMemory m, VkDeviceSize o, VkDeviceSize s, VkMemoryMapFlags f, void** p)
    {
        if (fail(Fault::Map)) return VK_ERROR_MEMORY_MAP_FAILED;
        return vkMapMemory(d, m, o, s, f, p);
    }
}
#define vkAllocateMemory vk_test::allocate
#define vkCreateGraphicsPipelines vk_test::pipeline
#define vkCreateFence vk_test::fence
#define vkQueueSubmit vk_test::submit
#define vkMapMemory vk_test::map
