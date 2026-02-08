#pragma once

/*
    SHS RENDERER SAN

    FILE: vk_memory_utils.hpp
    MODULE: rhi/drivers/vulkan
    PURPOSE: Shared Vulkan memory/buffer helpers for demos and runtime utilities.
*/


#include <cstdint>

#ifdef SHS_HAS_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace shs
{
#ifdef SHS_HAS_VULKAN
    inline uint32_t vk_find_memory_type(
        VkPhysicalDevice physical_device,
        uint32_t type_bits,
        VkMemoryPropertyFlags required_props)
    {
        if (physical_device == VK_NULL_HANDLE) return UINT32_MAX;

        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(physical_device, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        {
            const bool type_ok = (type_bits & (1u << i)) != 0;
            const bool props_ok = (mp.memoryTypes[i].propertyFlags & required_props) == required_props;
            if (type_ok && props_ok)
            {
                return i;
            }
        }
        return UINT32_MAX;
    }

    inline bool vk_create_buffer(
        VkDevice device,
        VkPhysicalDevice physical_device,
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags memory_props,
        VkBuffer& out_buffer,
        VkDeviceMemory& out_memory)
    {
        out_buffer = VK_NULL_HANDLE;
        out_memory = VK_NULL_HANDLE;
        if (device == VK_NULL_HANDLE || physical_device == VK_NULL_HANDLE || size == 0)
        {
            return false;
        }

        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size;
        bci.usage = usage;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device, &bci, nullptr, &out_buffer) != VK_SUCCESS)
        {
            return false;
        }

        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device, out_buffer, &req);
        const uint32_t memory_type = vk_find_memory_type(physical_device, req.memoryTypeBits, memory_props);
        if (memory_type == UINT32_MAX)
        {
            vkDestroyBuffer(device, out_buffer, nullptr);
            out_buffer = VK_NULL_HANDLE;
            return false;
        }

        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = memory_type;
        if (vkAllocateMemory(device, &mai, nullptr, &out_memory) != VK_SUCCESS)
        {
            vkDestroyBuffer(device, out_buffer, nullptr);
            out_buffer = VK_NULL_HANDLE;
            return false;
        }

        if (vkBindBufferMemory(device, out_buffer, out_memory, 0) != VK_SUCCESS)
        {
            vkFreeMemory(device, out_memory, nullptr);
            out_memory = VK_NULL_HANDLE;
            vkDestroyBuffer(device, out_buffer, nullptr);
            out_buffer = VK_NULL_HANDLE;
            return false;
        }
        return true;
    }

    inline void vk_destroy_buffer(VkDevice device, VkBuffer& buffer, VkDeviceMemory& memory)
    {
        if (device != VK_NULL_HANDLE)
        {
            if (buffer != VK_NULL_HANDLE)
            {
                vkDestroyBuffer(device, buffer, nullptr);
            }
            if (memory != VK_NULL_HANDLE)
            {
                vkFreeMemory(device, memory, nullptr);
            }
        }
        buffer = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
    }
#endif
}
