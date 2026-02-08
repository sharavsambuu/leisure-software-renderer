#pragma once

/*
    SHS RENDERER SAN

    FILE: vk_shader_utils.hpp
    MODULE: rhi/drivers/vulkan
    PURPOSE: Shared Vulkan shader/file helpers used by demos and reusable runtime code.
*/


#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef SHS_HAS_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace shs
{
    inline bool vk_try_read_binary_file(const char* path, std::vector<char>& out_bytes) noexcept
    {
        out_bytes.clear();
        if (!path || path[0] == '\0')
        {
            return false;
        }

        std::ifstream f(path, std::ios::ate | std::ios::binary);
        if (!f.is_open())
        {
            return false;
        }

        const std::streampos end_pos = f.tellg();
        if (end_pos <= 0)
        {
            return false;
        }
        const size_t sz = static_cast<size_t>(end_pos);

        out_bytes.resize(sz);
        f.seekg(0);
        f.read(out_bytes.data(), static_cast<std::streamsize>(sz));
        if (!f)
        {
            out_bytes.clear();
            return false;
        }
        return true;
    }

    inline std::vector<char> vk_read_binary_file(const char* path)
    {
        std::vector<char> out{};
        if (!vk_try_read_binary_file(path, out))
        {
            throw std::runtime_error(std::string("Failed to read binary file: ") + (path ? path : "<null>"));
        }
        return out;
    }

#ifdef SHS_HAS_VULKAN
    inline bool vk_try_create_shader_module(
        VkDevice device,
        const std::vector<char>& spirv_code,
        VkShaderModule& out_shader_module) noexcept
    {
        out_shader_module = VK_NULL_HANDLE;
        if (device == VK_NULL_HANDLE)
        {
            return false;
        }
        if (spirv_code.empty() || (spirv_code.size() % 4) != 0)
        {
            return false;
        }

        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = spirv_code.size();
        ci.pCode = reinterpret_cast<const uint32_t*>(spirv_code.data());

        return vkCreateShaderModule(device, &ci, nullptr, &out_shader_module) == VK_SUCCESS;
    }

    inline VkShaderModule vk_create_shader_module(VkDevice device, const std::vector<char>& spirv_code)
    {
        VkShaderModule out = VK_NULL_HANDLE;
        if (!vk_try_create_shader_module(device, spirv_code, out))
        {
            throw std::runtime_error("vk_create_shader_module: failed to create module");
        }
        return out;
    }
#endif
}
