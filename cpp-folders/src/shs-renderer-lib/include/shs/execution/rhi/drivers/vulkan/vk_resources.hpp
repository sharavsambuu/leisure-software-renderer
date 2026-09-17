#pragma once

/*
    SHS RENDERER SAN

    FILE: vk_resources.hpp
    MODULE: rhi/drivers/vulkan
    PURPOSE: Vulkan driver pod — ResourceDesc → VkBuffer/VkImage/Sampler.
            The bookkeeping layer (stable-ID registry, descriptor-hash-keyed
            explicit cache, stats) is pure and GPU-free testable; only the
            factory talks to vkCreate/vkDestroy calls.
            No per-node allocation: keyed state lives in shs::containers::FlatMap.
*/

#include <cstdint>

#include <vulkan/vulkan.h>

#include "shs/containers/flat_map.hpp"
#include "shs/execution/rhi/resource/resource_desc.hpp"
#include "shs/execution/rhi/drivers/vulkan/vk_device.hpp"

namespace shs
{
    // ------------------------------------------------------------------
    // Pure create-info builders (GPU-free testable).
    // ------------------------------------------------------------------

    [[nodiscard]] inline VkBufferCreateInfo vk_buffer_create_info(const RHIBufferDesc& d)
    {
        VkBufferCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ci.size = d.size_bytes;
        ci.usage = vk_buffer_usage_of(d.usage);
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        return ci;
    }

    [[nodiscard]] inline VkImageCreateInfo vk_image_create_info(const RHIImageDesc& d)
    {
        VkImageCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = vk_format_of(d.format);
        ci.extent = VkExtent3D{(uint32_t)d.width, (uint32_t)d.height, 1};
        ci.mipLevels = d.mip_levels > 0 ? (uint32_t)d.mip_levels : 1;
        ci.arrayLayers = d.layers > 0 ? (uint32_t)d.layers : 1;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        ci.usage = vk_image_usage_of(d.usage);
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ci.flags = vk_image_flags_of(d.type);
        return ci;
    }

    [[nodiscard]] inline VkSamplerCreateInfo vk_sampler_create_info(const RHISamplerDesc& d)
    {
        VkSamplerCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        ci.magFilter = vk_filter_of(d.mag_filter);
        ci.minFilter = vk_filter_of(d.min_filter);
        ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        ci.addressModeU = vk_address_mode_of(d.address_u);
        ci.addressModeV = vk_address_mode_of(d.address_v);
        ci.addressModeW = vk_address_mode_of(d.address_w);
        ci.anisotropyEnable = d.enable_anisotropy ? VK_TRUE : VK_FALSE;
        ci.maxAnisotropy = d.max_anisotropy;
        ci.compareEnable = VK_FALSE;
        return ci;
    }

    // Pure memory-type picker — testable with a fabricated properties value.
    [[nodiscard]] inline uint32_t vk_pick_memory_type(const VkPhysicalDeviceMemoryProperties& props,
                                                      uint32_t type_bits, VkMemoryPropertyFlags wanted)
    {
        for (uint32_t i = 0; i < props.memoryTypeCount; ++i)
        {
            if ((type_bits & (1u << i)) != 0 &&
                (props.memoryTypes[i].propertyFlags & wanted) == wanted)
            {
                return i;
            }
        }
        return UINT32_MAX;
    }

    // ------------------------------------------------------------------
    // Descriptor-hash-keyed explicit resource cache (arch §4 rule 3).
    // Pure bookkeeping: the creation hook is a callable, so tests exercise
    // the whole cache/ID logic with a counting lambda instead of a device.
    // ------------------------------------------------------------------

    struct VulkanResourceStats
    {
        uint64_t create_calls = 0;   // actual backend creations (misses)
        uint64_t cache_hits = 0;     // identical-desc reuses
        uint64_t live_buffers = 0;
        uint64_t live_images = 0;
        uint64_t live_samplers = 0;
    };

    struct VulkanResourceRecord
    {
        uint64_t id = 0;         // stable ID handed upward (never a Vk*)
        uint64_t desc_hash = 0;  // explicit cache key
    };

    class VulkanResourceRegistry
    {
    public:
        // Stable resource-ID namespace. Top byte tags the kind.
        static constexpr uint64_t kBufferIdBase = 0x5200000000000000ull; // 'R'
        static constexpr uint64_t kImageIdBase = 0x5300000000000000ull;  // 'S'
        static constexpr uint64_t kSamplerIdBase = 0x5400000000000000ull;

        explicit VulkanResourceRegistry(std::pmr::memory_resource* resource = std::pmr::get_default_resource())
            : buffers_(resource), images_(resource), samplers_(resource),
              buffers_by_id_(resource), images_by_id_(resource), samplers_by_id_(resource) {}

        // Returns the stable ID for the desc. Identical desc → same ID with
        // zero extra creation calls (explicit hash-keyed cache contract).
        template <typename CreateFn>
        [[nodiscard]] uint64_t intern_buffer(const RHIBufferDesc& d, CreateFn&& create)
        {
            const uint64_t h = hash_buffer_desc(d);
            if (const VulkanResourceRecord* hit = buffers_.find(h))
            {
                stats_.cache_hits++;
                return hit->id;
            }
            const uint64_t id = kBufferIdBase + ++next_buffer_id_;
            if (!create(id, d)) return 0; // creation failed → no record, no ID
            stats_.create_calls++;
            stats_.live_buffers++;
            const VulkanResourceRecord rec{id, h};
            buffers_.insert_or_assign(h, rec);
            buffers_by_id_.insert_or_assign(id, rec);
            return id;
        }

        template <typename CreateFn>
        [[nodiscard]] uint64_t intern_image(const RHIImageDesc& d, CreateFn&& create)
        {
            const uint64_t h = hash_image_desc(d);
            if (const VulkanResourceRecord* hit = images_.find(h))
            {
                stats_.cache_hits++;
                return hit->id;
            }
            const uint64_t id = kImageIdBase + ++next_image_id_;
            if (!create(id, d)) return 0;
            stats_.create_calls++;
            stats_.live_images++;
            const VulkanResourceRecord rec{id, h};
            images_.insert_or_assign(h, rec);
            images_by_id_.insert_or_assign(id, rec);
            return id;
        }

        template <typename CreateFn>
        [[nodiscard]] uint64_t intern_sampler(const RHISamplerDesc& d, CreateFn&& create)
        {
            const uint64_t h = hash_sampler_desc(d);
            if (const VulkanResourceRecord* hit = samplers_.find(h))
            {
                stats_.cache_hits++;
                return hit->id;
            }
            const uint64_t id = kSamplerIdBase + ++next_sampler_id_;
            if (!create(id, d)) return 0;
            stats_.create_calls++;
            stats_.live_samplers++;
            const VulkanResourceRecord rec{id, h};
            samplers_.insert_or_assign(h, rec);
            samplers_by_id_.insert_or_assign(id, rec);
            return id;
        }

        [[nodiscard]] const VulkanResourceRecord* find_buffer(uint64_t id) const { return buffers_by_id_.find(id); }
        [[nodiscard]] const VulkanResourceRecord* find_image(uint64_t id) const { return images_by_id_.find(id); }
        [[nodiscard]] const VulkanResourceRecord* find_sampler(uint64_t id) const { return samplers_by_id_.find(id); }

        // Retire lookup records, but never recycle stable IDs or lifetime counters.
        void clear()
        {
            buffers_.clear(); images_.clear(); samplers_.clear();
            buffers_by_id_.clear(); images_by_id_.clear(); samplers_by_id_.clear();
            stats_.live_buffers = stats_.live_images = stats_.live_samplers = 0;
        }

        [[nodiscard]] const VulkanResourceStats& stats() const { return stats_; }

    private:
        using RecordMap = containers::FlatMap<uint64_t, VulkanResourceRecord>;
        // hash-keyed dedupe map + id-keyed lookup index (both contiguous, no nodes)
        RecordMap buffers_;
        RecordMap images_;
        RecordMap samplers_;
        RecordMap buffers_by_id_;
        RecordMap images_by_id_;
        RecordMap samplers_by_id_;
        uint64_t next_buffer_id_ = 0;
        uint64_t next_image_id_ = 0;
        uint64_t next_sampler_id_ = 0;
        VulkanResourceStats stats_{};
    };
}
