#pragma once

/*
    SHS RENDERER SAN

    FILE: vk_cmd_utils.hpp
    MODULE: rhi/drivers/vulkan
    PURPOSE: Shared Vulkan command recording helpers.
*/


#include <cstdint>

#ifdef SHS_HAS_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace shs
{
#ifdef SHS_HAS_VULKAN
    inline VkViewport vk_make_viewport(uint32_t width, uint32_t height, bool flip_y)
    {
        VkViewport vp{};
        vp.x = 0.0f;
        vp.y = flip_y ? static_cast<float>(height) : 0.0f;
        vp.width = static_cast<float>(width);
        vp.height = flip_y ? -static_cast<float>(height) : static_cast<float>(height);
        vp.minDepth = 0.0f;
        vp.maxDepth = 1.0f;
        return vp;
    }

    inline VkRect2D vk_make_scissor(uint32_t width, uint32_t height)
    {
        VkRect2D sc{};
        sc.offset = {0, 0};
        sc.extent = {width, height};
        return sc;
    }

    inline void vk_cmd_set_viewport_scissor(VkCommandBuffer cmd, uint32_t width, uint32_t height, bool flip_y)
    {
        const VkViewport vp = vk_make_viewport(width, height, flip_y);
        const VkRect2D sc = vk_make_scissor(width, height);
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
    }
#endif
}
