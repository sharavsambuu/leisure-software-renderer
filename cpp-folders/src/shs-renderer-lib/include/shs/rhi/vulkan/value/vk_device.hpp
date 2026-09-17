#pragma once

/*
    SHS RENDERER SAN

    FILE: vk_device.hpp
    MODULE: rhi/drivers/vulkan
    PURPOSE: Vulkan driver pod — device/instance/queue bootstrap + the pure
            value-desc → Vulkan translation layer (arch doc §4).
            Every mapping here is a pure function so it is testable without a
            device. No Vk* handle ever crosses upward out of the driver zone.
*/

#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>

#include <vulkan/vulkan.h>

#include "shs/rhi/resource/resource_desc.hpp"
#include "shs/rhi/pipeline/pipeline_desc.hpp"
#include "shs/rhi/sync/sync_desc.hpp"
#include "shs/rhi/command/command_desc.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace rhi
    {
    [[nodiscard]] inline VkFormat vk_format_of(RHIFormat format)
    {
        switch (format)
        {
            case RHIFormat::RGBA8_UNorm: return VK_FORMAT_R8G8B8A8_UNORM;
            case RHIFormat::BGRA8_UNorm: return VK_FORMAT_B8G8R8A8_UNORM;
            case RHIFormat::RGBA16F: return VK_FORMAT_R16G16B16A16_SFLOAT;
            case RHIFormat::RGBA32F: return VK_FORMAT_R32G32B32A32_SFLOAT;
            case RHIFormat::D24S8: return VK_FORMAT_D24_UNORM_S8_UINT;
            case RHIFormat::D32F: return VK_FORMAT_D32_SFLOAT;
            case RHIFormat::Unknown: break;
        }
        return VK_FORMAT_UNDEFINED;
    }

    [[nodiscard]] inline VkBufferUsageFlags vk_buffer_usage_of(uint32_t usage)
    {
        VkBufferUsageFlags out = 0;
        if (usage & RHIBufferUsage_Vertex) out |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        if (usage & RHIBufferUsage_Index) out |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        if (usage & RHIBufferUsage_Uniform) out |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        if (usage & RHIBufferUsage_Storage) out |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (usage & RHIBufferUsage_TransferSrc) out |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (usage & RHIBufferUsage_TransferDst) out |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        return out;
    }

    [[nodiscard]] inline VkImageUsageFlags vk_image_usage_of(uint32_t usage)
    {
        VkImageUsageFlags out = 0;
        if (usage & RHIImageUsage_Sampled) out |= VK_IMAGE_USAGE_SAMPLED_BIT;
        if (usage & RHIImageUsage_ColorAttachment) out |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (usage & RHIImageUsage_DepthStencilAttachment) out |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        if (usage & RHIImageUsage_Storage) out |= VK_IMAGE_USAGE_STORAGE_BIT;
        if (usage & RHIImageUsage_TransferSrc) out |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        if (usage & RHIImageUsage_TransferDst) out |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        return out;
    }

    [[nodiscard]] inline VkImageCreateFlags vk_image_flags_of(RHIImageType type)
    {
        return type == RHIImageType::TexCube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
    }

    [[nodiscard]] inline VkMemoryPropertyFlags vk_memory_props_of(RHIMemoryClass memory)
    {
        switch (memory)
        {
            case RHIMemoryClass::CPUVisible: return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            case RHIMemoryClass::Readback: return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
            case RHIMemoryClass::GPUOnly: return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            case RHIMemoryClass::Auto: break;
        }
        return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    }

    [[nodiscard]] inline VkPipelineStageFlags vk_stage_of(RHIPipelineStage stage)
    {
        switch (stage)
        {
            case RHIPipelineStage::Top: return VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            case RHIPipelineStage::DrawIndirect: return VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
            case RHIPipelineStage::VertexInput: return VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
            case RHIPipelineStage::VertexShader: return VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;
            case RHIPipelineStage::FragmentShader: return VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            case RHIPipelineStage::ColorOutput: return VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            case RHIPipelineStage::ComputeShader: return VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            case RHIPipelineStage::Transfer: return VK_PIPELINE_STAGE_TRANSFER_BIT;
            case RHIPipelineStage::Bottom: return VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        }
        return VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    }

    [[nodiscard]] inline VkAccessFlags vk_access_of(RHIAccess access)
    {
        switch (access)
        {
            case RHIAccess::Read: return VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_READ_BIT;
            case RHIAccess::Write: return VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            case RHIAccess::ReadWrite: return VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            case RHIAccess::None: break;
        }
        return 0;
    }

    [[nodiscard]] inline VkCullModeFlags vk_cull_mode_of(RHICullMode mode)
    {
        switch (mode)
        {
            case RHICullMode::None: return VK_CULL_MODE_NONE;
            case RHICullMode::Back: return VK_CULL_MODE_BACK_BIT;
            case RHICullMode::Front: return VK_CULL_MODE_FRONT_BIT;
        }
        return VK_CULL_MODE_NONE;
    }

    [[nodiscard]] inline VkFrontFace vk_front_face_of(RHIFrontFace face)
    {
        return face == RHIFrontFace::CW ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
    }

    [[nodiscard]] inline VkShaderStageFlagBits vk_shader_stage_of(RHIShaderStage stage)
    {
        switch (stage)
        {
            case RHIShaderStage::Vertex: return VK_SHADER_STAGE_VERTEX_BIT;
            case RHIShaderStage::Fragment: return VK_SHADER_STAGE_FRAGMENT_BIT;
            case RHIShaderStage::Compute: return VK_SHADER_STAGE_COMPUTE_BIT;
        }
        return VK_SHADER_STAGE_VERTEX_BIT;
    }

    [[nodiscard]] inline VkFilter vk_filter_of(RHIFilter filter)
    {
        return filter == RHIFilter::Nearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
    }

    [[nodiscard]] inline VkSamplerAddressMode vk_address_mode_of(RHIAddressMode mode)
    {
        switch (mode)
        {
            case RHIAddressMode::ClampToEdge: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            case RHIAddressMode::Repeat: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
            case RHIAddressMode::MirrorRepeat: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        }
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    }

    // ------------------------------------------------------------------
    // Descriptor hashing — the explicit cache key vocabulary (arch §4 rule 3).
    // ------------------------------------------------------------------

    [[nodiscard]] inline uint64_t rhi_hash_bytes(const void* data, uint64_t size, uint64_t seed = 0xcbf29ce484222325ull)
    {
        const auto* bytes = static_cast<const uint8_t*>(data);
        uint64_t h = seed;
        for (uint64_t i = 0; i < size; ++i)
        {
            h ^= (uint64_t)bytes[i];
            h *= 0x100000001b3ull; // FNV-1a
        }
        return h;
    }

    template <typename T>
    [[nodiscard]] inline uint64_t rhi_hash_value(uint64_t seed, const T& value)
    {
        return rhi_hash_bytes(&value, (uint64_t)sizeof(T), seed);
    }

    [[nodiscard]] inline uint64_t hash_buffer_desc(const RHIBufferDesc& d)
    {
        uint64_t h = rhi_hash_value(0xcbf29ce484222325ull, d.size_bytes);
        h = rhi_hash_value(h, d.usage);
        h = rhi_hash_value(h, (uint8_t)d.memory);
        return h;
    }

    [[nodiscard]] inline uint64_t hash_image_desc(const RHIImageDesc& d)
    {
        uint64_t h = rhi_hash_value(0xcbf29ce484222325ull, (uint8_t)d.type);
        h = rhi_hash_value(h, d.width);
        h = rhi_hash_value(h, d.height);
        h = rhi_hash_value(h, d.mip_levels);
        h = rhi_hash_value(h, d.layers);
        h = rhi_hash_value(h, (uint32_t)d.format);
        h = rhi_hash_value(h, d.usage);
        h = rhi_hash_value(h, (uint8_t)d.memory);
        return h;
    }

    [[nodiscard]] inline uint64_t hash_sampler_desc(const RHISamplerDesc& d)
    {
        uint64_t h = rhi_hash_value(0xcbf29ce484222325ull, (uint8_t)d.min_filter);
        h = rhi_hash_value(h, (uint8_t)d.mag_filter);
        h = rhi_hash_value(h, (uint8_t)d.address_u);
        h = rhi_hash_value(h, (uint8_t)d.address_v);
        h = rhi_hash_value(h, (uint8_t)d.address_w);
        h = rhi_hash_value(h, d.enable_anisotropy ? 1ull : 0ull);
        h = rhi_hash_bytes(&d.max_anisotropy, sizeof(float), h);
        return h;
    }

    [[nodiscard]] inline uint64_t hash_shader_module_desc(const RHIShaderModuleDesc& d)
    {
        uint64_t h = rhi_hash_value(0xcbf29ce484222325ull, (uint8_t)d.stage);
        h = rhi_hash_value(h, d.bytecode_size);
        h = rhi_hash_bytes(d.bytecode, d.bytecode_size, h);
        const uint64_t entry_hash = (d.entry != nullptr) ? rhi_hash_bytes(d.entry, (uint64_t)std::char_traits<char>::length(d.entry)) : 0;
        return rhi_hash_value(h, entry_hash);
    }

    [[nodiscard]] inline uint64_t hash_graphics_pipeline_desc(const RHIGraphicsPipelineDesc& d)
    {
        uint64_t h = hash_shader_module_desc(d.vs);
        h = rhi_hash_value(h, (uint8_t)d.fs.stage);
        h = rhi_hash_value(h, d.fs.bytecode_size);
        h = rhi_hash_bytes(d.fs.bytecode, d.fs.bytecode_size, h);
        h = rhi_hash_value(h, d.raster.cull);
        h = rhi_hash_value(h, d.raster.front_face);
        h = rhi_hash_value(h, d.raster.depth_clamp);
        h = rhi_hash_value(h, d.depth.enable_test);
        h = rhi_hash_value(h, d.depth.enable_write);
        h = rhi_hash_value(h, d.blend.enable);
        h = rhi_hash_value(h, d.rt.color_format);
        h = rhi_hash_value(h, d.rt.depth_format);
        h = rhi_hash_value(h, d.rt.has_depth);
        h = rhi_hash_value(h, d.vertex_layout);
        const uint64_t fragment_entry = d.fs.entry ? rhi_hash_bytes(d.fs.entry,
            std::char_traits<char>::length(d.fs.entry)) : 0;
        return rhi_hash_value(h, fragment_entry);
    }

    [[nodiscard]] inline uint64_t hash_compute_pipeline_desc(const RHIComputePipelineDesc& d)
    {
        return hash_shader_module_desc(d.cs);
    }

    // ------------------------------------------------------------------
    // Device bootstrap (value-desc → instance/device/queues; edge, runs once).
    // ------------------------------------------------------------------

    struct VulkanDeviceDesc
    {
        std::string_view app_name = "shs-renderer-lib";
        bool enable_validation = false;
        uint32_t api_version = VK_API_VERSION_1_1;
    };

    // What the rest of the engine sees — a stable value. No Vk* types.
    struct VulkanDeviceInfo
    {
        uint32_t graphics_queue_family = UINT32_MAX;
        uint32_t compute_queue_family = UINT32_MAX;
        uint32_t transfer_queue_family = UINT32_MAX;
        uint32_t api_version = 0;
        char device_name[256] = {};
    };

    class VulkanDeviceManager
    {
    public:
        // Returns false when no loader/ICD is available — the backend then
        // runs in headless capability mode (no device creation, no GPU work).
        [[nodiscard]] bool initialize(const VulkanDeviceDesc& desc)
        {
            VkApplicationInfo app{};
            app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
            app.pApplicationName = desc.app_name.data();
            app.applicationVersion = VK_MAKE_VERSION(0, 2, 0);
            app.pEngineName = "shs";
            app.engineVersion = VK_MAKE_VERSION(0, 2, 0);
            app.apiVersion = desc.api_version;

            VkInstanceCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
            ci.pApplicationInfo = &app;

            if (vkCreateInstance(&ci, nullptr, &instance_) != VK_SUCCESS) return false;

            uint32_t device_count = 0;
            if (vkEnumeratePhysicalDevices(instance_, &device_count, nullptr) != VK_SUCCESS || device_count == 0)
            {
                shutdown();
                return false; // loader present, no ICD → headless mode
            }
            VkPhysicalDevice devices[8]{};
            if (device_count > 8) device_count = 8;
            if (vkEnumeratePhysicalDevices(instance_, &device_count, devices) != VK_SUCCESS || device_count == 0)
            {
                shutdown();
                return false;
            }

            physical_ = devices[0];
            for (uint32_t i = 0; i < device_count; ++i)
            {
                VkPhysicalDeviceProperties props{};
                vkGetPhysicalDeviceProperties(devices[i], &props);
                if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
                {
                    physical_ = devices[i];
                    break;
                }
            }
            if (!pick_queue_families()) { shutdown(); return false; }

            const float priority = 1.0f;
            VkDeviceQueueCreateInfo qci{};
            qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            qci.queueFamilyIndex = info_.graphics_queue_family;
            qci.queueCount = 1;
            qci.pQueuePriorities = &priority;

            VkDeviceCreateInfo dci{};
            dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
            dci.queueCreateInfoCount = 1;
            dci.pQueueCreateInfos = &qci;

            if (vkCreateDevice(physical_, &dci, nullptr, &device_) != VK_SUCCESS) { shutdown(); return false; }
            vkGetDeviceQueue(device_, info_.graphics_queue_family, 0, &graphics_queue_);

            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(physical_, &props);
            std::snprintf(info_.device_name, sizeof(info_.device_name), "%s", props.deviceName);
            info_.api_version = desc.api_version;
            return true;
        }

        void shutdown()
        {
            if (device_ != VK_NULL_HANDLE) vkDestroyDevice(device_, nullptr);
            if (instance_ != VK_NULL_HANDLE) vkDestroyInstance(instance_, nullptr);
            device_ = VK_NULL_HANDLE;
            instance_ = VK_NULL_HANDLE;
            graphics_queue_ = VK_NULL_HANDLE;
            physical_ = VK_NULL_HANDLE;
            info_ = VulkanDeviceInfo{};
        }

        [[nodiscard]] VkInstance instance() const { return instance_; }
        [[nodiscard]] VkPhysicalDevice physical() const { return physical_; }
        [[nodiscard]] VkDevice device() const { return device_; }
        [[nodiscard]] VkQueue graphics_queue() const { return graphics_queue_; }
        [[nodiscard]] const VulkanDeviceInfo& info() const { return info_; }
        [[nodiscard]] bool device_available() const { return device_ != VK_NULL_HANDLE; }

    private:
        [[nodiscard]] bool pick_queue_families()
        {
            VkQueueFamilyProperties props[64]{};
            uint32_t count = 64;
            vkGetPhysicalDeviceQueueFamilyProperties(physical_, &count, props);
            if (count == 0 || count > 64) return false;

            for (uint32_t i = 0; i < count; ++i)
            {
                if ((props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) { info_.graphics_queue_family = i; break; }
            }
            if (info_.graphics_queue_family == UINT32_MAX) return false;

            info_.compute_queue_family = info_.graphics_queue_family;
            for (uint32_t i = 0; i < count; ++i)
            {
                if ((props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) { info_.compute_queue_family = i; break; }
            }
            info_.transfer_queue_family = info_.graphics_queue_family;
            for (uint32_t i = 0; i < count; ++i)
            {
                if ((props[i].queueFlags & VK_QUEUE_TRANSFER_BIT) != 0 && (props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0)
                {
                    info_.transfer_queue_family = i;
                    break;
                }
            }
            return true;
        }

        VkInstance instance_ = VK_NULL_HANDLE;
        VkPhysicalDevice physical_ = VK_NULL_HANDLE;
        VkDevice device_ = VK_NULL_HANDLE;
        VkQueue graphics_queue_ = VK_NULL_HANDLE;
        VulkanDeviceInfo info_{};
    };

    } // inline namespace rhi
}
