#pragma once

/*
    SHS RENDERER SAN

    FILE: vk_render_path_descriptors.hpp
    MODULE: rhi/drivers/vulkan
    PURPOSE: Shared Vulkan descriptor layout/pool/update helpers for render-path global bindings.
*/


#include <array>
#include <cstdint>

#ifdef SHS_HAS_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace shs
{
#ifdef SHS_HAS_VULKAN
    enum class VkRenderPathGlobalBinding : uint32_t
    {
        CameraUBO = 0u,
        LightsSSBO = 1u,
        LightTileCountsSSBO = 2u,
        LightTileIndicesSSBO = 3u,
        LightTileDepthRangesSSBO = 4u,
        DepthSampler = 5u,
        SunShadowSampler = 6u,
        LocalShadowSampler = 7u,
        PointShadowSampler = 8u,
        ShadowLightsSSBO = 9u,
        BindlessTextures = 10u
    };

    constexpr uint32_t vk_render_path_global_binding_count()
    {
        return 10u;
    }

    inline std::array<VkDescriptorSetLayoutBinding, 10> vk_make_render_path_global_set_layout_bindings()
    {
        std::array<VkDescriptorSetLayoutBinding, 10> bindings{};

        bindings[0].binding = static_cast<uint32_t>(VkRenderPathGlobalBinding::CameraUBO);
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;

        for (uint32_t i = 1u; i < 5u; ++i)
        {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
        }

        bindings[5].binding = static_cast<uint32_t>(VkRenderPathGlobalBinding::DepthSampler);
        bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[5].descriptorCount = 1;
        bindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        bindings[6].binding = static_cast<uint32_t>(VkRenderPathGlobalBinding::SunShadowSampler);
        bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[6].descriptorCount = 1;
        bindings[6].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        bindings[7].binding = static_cast<uint32_t>(VkRenderPathGlobalBinding::LocalShadowSampler);
        bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[7].descriptorCount = 1;
        bindings[7].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        bindings[8].binding = static_cast<uint32_t>(VkRenderPathGlobalBinding::PointShadowSampler);
        bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[8].descriptorCount = 1;
        bindings[8].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        bindings[9].binding = static_cast<uint32_t>(VkRenderPathGlobalBinding::ShadowLightsSSBO);
        bindings[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[9].descriptorCount = 1;
        bindings[9].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        return bindings;
    }

    inline bool vk_create_render_path_global_descriptor_set_layout(
        VkDevice device,
        VkDescriptorSetLayout* out_layout)
    {
        if (device == VK_NULL_HANDLE || !out_layout) return false;
        const std::array<VkDescriptorSetLayoutBinding, 10> bindings =
            vk_make_render_path_global_set_layout_bindings();

        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = vk_render_path_global_binding_count();
        ci.pBindings = bindings.data();
        return vkCreateDescriptorSetLayout(device, &ci, nullptr, out_layout) == VK_SUCCESS;
    }

    inline bool vk_create_bindless_descriptor_set_layout(
        VkDevice device,
        uint32_t max_textures,
        VkDescriptorSetLayout* out_layout)
    {
        if (device == VK_NULL_HANDLE || !out_layout) return false;

        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binding.descriptorCount = max_textures;
        binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        binding.pImmutableSamplers = nullptr;

        VkDescriptorBindingFlagsEXT flags =
            VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT |
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT_EXT |
            VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT_EXT;

        VkDescriptorSetLayoutBindingFlagsCreateInfoEXT layout_flags{};
        layout_flags.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT;
        layout_flags.bindingCount = 1;
        layout_flags.pBindingFlags = &flags;

        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.pNext = &layout_flags;
        ci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT_EXT;
        ci.bindingCount = 1;
        ci.pBindings = &binding;

        return vkCreateDescriptorSetLayout(device, &ci, nullptr, out_layout) == VK_SUCCESS;
    }

    inline std::array<VkDescriptorPoolSize, 3> vk_make_render_path_global_pool_sizes(uint32_t set_count)
    {
        const uint32_t n = (set_count == 0u) ? 1u : set_count;
        std::array<VkDescriptorPoolSize, 3> sizes{};
        sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        sizes[0].descriptorCount = 1u * n;
        sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        sizes[1].descriptorCount = 5u * n;
        sizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[2].descriptorCount = 4u * n;
        return sizes;
    }

    inline bool vk_create_render_path_global_descriptor_pool(
        VkDevice device,
        uint32_t set_count,
        VkDescriptorPool* out_pool)
    {
        if (device == VK_NULL_HANDLE || !out_pool) return false;
        const uint32_t n = (set_count == 0u) ? 1u : set_count;
        const std::array<VkDescriptorPoolSize, 3> sizes = vk_make_render_path_global_pool_sizes(n);

        VkDescriptorPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        ci.maxSets = n;
        ci.poolSizeCount = static_cast<uint32_t>(sizes.size());
        ci.pPoolSizes = sizes.data();
        return vkCreateDescriptorPool(device, &ci, nullptr, out_pool) == VK_SUCCESS;
    }

    inline bool vk_create_bindless_descriptor_pool(
        VkDevice device,
        uint32_t max_textures,
        VkDescriptorPool* out_pool)
    {
        if (device == VK_NULL_HANDLE || !out_pool) return false;

        VkDescriptorPoolSize size{};
        size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        size.descriptorCount = max_textures;

        VkDescriptorPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        ci.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT_EXT;
        ci.maxSets = 1;
        ci.poolSizeCount = 1;
        ci.pPoolSizes = &size;

        return vkCreateDescriptorPool(device, &ci, nullptr, out_pool) == VK_SUCCESS;
    }

    struct VkRenderPathGlobalDescriptorFrameData
    {
        VkDescriptorSet dst_set = VK_NULL_HANDLE;
        VkBuffer camera_buffer = VK_NULL_HANDLE;
        VkDeviceSize camera_range = 0;
        VkBuffer lights_buffer = VK_NULL_HANDLE;
        VkDeviceSize lights_range = 0;
        VkBuffer tile_counts_buffer = VK_NULL_HANDLE;
        VkDeviceSize tile_counts_range = 0;
        VkBuffer tile_indices_buffer = VK_NULL_HANDLE;
        VkDeviceSize tile_indices_range = 0;
        VkBuffer tile_depth_ranges_buffer = VK_NULL_HANDLE;
        VkDeviceSize tile_depth_ranges_range = 0;
        VkBuffer shadow_lights_buffer = VK_NULL_HANDLE;
        VkDeviceSize shadow_lights_range = 0;
        VkSampler sampler = VK_NULL_HANDLE;
        VkImageView depth_view = VK_NULL_HANDLE;
        VkImageView sun_shadow_view = VK_NULL_HANDLE;
        VkImageView local_shadow_view = VK_NULL_HANDLE;
        VkImageView point_shadow_view = VK_NULL_HANDLE;
    };

    inline bool vk_update_render_path_global_descriptor_set(
        VkDevice device,
        const VkRenderPathGlobalDescriptorFrameData& frame)
    {
        if (device == VK_NULL_HANDLE || frame.dst_set == VK_NULL_HANDLE) return false;
        if (frame.sampler == VK_NULL_HANDLE) return false;

        VkDescriptorBufferInfo camera_info{};
        camera_info.buffer = frame.camera_buffer;
        camera_info.offset = 0;
        camera_info.range = frame.camera_range;

        VkDescriptorBufferInfo lights_info{};
        lights_info.buffer = frame.lights_buffer;
        lights_info.offset = 0;
        lights_info.range = frame.lights_range;

        VkDescriptorBufferInfo tile_counts_info{};
        tile_counts_info.buffer = frame.tile_counts_buffer;
        tile_counts_info.offset = 0;
        tile_counts_info.range = frame.tile_counts_range;

        VkDescriptorBufferInfo tile_indices_info{};
        tile_indices_info.buffer = frame.tile_indices_buffer;
        tile_indices_info.offset = 0;
        tile_indices_info.range = frame.tile_indices_range;

        VkDescriptorBufferInfo tile_depth_ranges_info{};
        tile_depth_ranges_info.buffer = frame.tile_depth_ranges_buffer;
        tile_depth_ranges_info.offset = 0;
        tile_depth_ranges_info.range = frame.tile_depth_ranges_range;

        VkDescriptorBufferInfo shadow_lights_info{};
        shadow_lights_info.buffer = frame.shadow_lights_buffer;
        shadow_lights_info.offset = 0;
        shadow_lights_info.range = frame.shadow_lights_range;

        VkDescriptorImageInfo depth_image{};
        depth_image.sampler = frame.sampler;
        depth_image.imageView = frame.depth_view;
        depth_image.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo sun_shadow_image{};
        sun_shadow_image.sampler = frame.sampler;
        sun_shadow_image.imageView = frame.sun_shadow_view;
        sun_shadow_image.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo local_shadow_image{};
        local_shadow_image.sampler = frame.sampler;
        local_shadow_image.imageView = frame.local_shadow_view;
        local_shadow_image.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo point_shadow_image{};
        point_shadow_image.sampler = frame.sampler;
        point_shadow_image.imageView = frame.point_shadow_view;
        point_shadow_image.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        std::array<VkWriteDescriptorSet, 10> writes{};
        for (uint32_t i = 0u; i < vk_render_path_global_binding_count(); ++i)
        {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = frame.dst_set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
        }

        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].pBufferInfo = &camera_info;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &lights_info;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].pBufferInfo = &tile_counts_info;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[3].pBufferInfo = &tile_indices_info;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[4].pBufferInfo = &tile_depth_ranges_info;
        writes[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[5].pImageInfo = &depth_image;
        writes[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[6].pImageInfo = &sun_shadow_image;
        writes[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[7].pImageInfo = &local_shadow_image;
        writes[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[8].pImageInfo = &point_shadow_image;
        writes[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[9].pBufferInfo = &shadow_lights_info;

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        return true;
    }

    inline void vk_update_bindless_texture(
        VkDevice device,
        VkDescriptorSet set,
        uint32_t slot,
        VkSampler sampler,
        VkImageView view)
    {
        VkDescriptorImageInfo image{};
        image.sampler = sampler;
        image.imageView = view;
        image.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = set;
        write.dstBinding = 0;
        write.dstArrayElement = slot;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = &image;

        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }
#endif
}
