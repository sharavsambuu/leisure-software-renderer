#pragma once

/*
    SHS RENDERER SAN

    FILE: vk_pipelines.hpp
    MODULE: rhi/drivers/vulkan
    PURPOSE: Vulkan driver pod — PipelineDesc → VkPipeline/PipelineLayout.
            Descriptor-hash-keyed EXPLICIT caches: the caller probes the cache
            with a desc and a create hook; identical descs reuse the cached
            pipeline. No lazy hidden caches, no per-node allocation (FlatMap).
            The bookkeeping layer is GPU-free testable via the create hook.
*/

#include <cstdint>

#include <vulkan/vulkan.h>

#include "shs/containers/flat_map.hpp"
#include "shs/execution/rhi/pipeline/pipeline_desc.hpp"
#include "shs/execution/rhi/drivers/vulkan/vk_device.hpp"

namespace shs
{
    struct VulkanPipelineStats
    {
        uint64_t shader_module_creates = 0;
        uint64_t graphics_pipeline_creates = 0;
        uint64_t compute_pipeline_creates = 0;
        uint64_t cache_hits = 0;
    };

    // Explicit cache slot — handed upward as a stable 64-bit ID.
    struct VulkanPipelineRecord
    {
        uint64_t id = 0;
        uint64_t desc_hash = 0;
    };

    class VulkanPipelineCache
    {
    public:
        // Stable pipeline-ID namespace (top byte tags the kind).
        static constexpr uint64_t kShaderModuleIdBase = 0x4d00000000000000ull;   // 'M'
        static constexpr uint64_t kGraphicsPipelineIdBase = 0x4700000000000000ull; // 'G'
        static constexpr uint64_t kComputePipelineIdBase = 0x4300000000000000ull;  // 'C'

        explicit VulkanPipelineCache(std::pmr::memory_resource* resource = std::pmr::get_default_resource())
            : shader_modules_(resource), graphics_(resource), compute_(resource),
              graphics_hash_by_id_(resource), compute_hash_by_id_(resource) {}

        template <typename CreateFn>
        [[nodiscard]] uint64_t intern_shader_module(const RHIShaderModuleDesc& d, CreateFn&& create)
        {
            const uint64_t h = hash_shader_module_desc(d);
            if (const VulkanPipelineRecord* hit = shader_modules_.find(h))
            {
                stats_.cache_hits++;
                return hit->id;
            }
            const uint64_t id = kShaderModuleIdBase + ++next_shader_module_id_;
            if (!create(id, d)) return 0;
            stats_.shader_module_creates++;
            shader_modules_.insert_or_assign(h, VulkanPipelineRecord{id, h});
            return id;
        }

        template <typename CreateFn>
        [[nodiscard]] uint64_t intern_graphics(const RHIGraphicsPipelineDesc& d, CreateFn&& create)
        {
            const uint64_t h = hash_graphics_pipeline_desc(d);
            if (const VulkanPipelineRecord* hit = graphics_.find(h))
            {
                stats_.cache_hits++;
                return hit->id;
            }
            const uint64_t id = kGraphicsPipelineIdBase + ++next_graphics_id_;
            if (!create(id, d)) return 0;
            stats_.graphics_pipeline_creates++;
            graphics_.insert_or_assign(h, VulkanPipelineRecord{id, h});
            graphics_hash_by_id_.insert_or_assign(id, h);
            return id;
        }

        template <typename CreateFn>
        [[nodiscard]] uint64_t intern_compute(const RHIComputePipelineDesc& d, CreateFn&& create)
        {
            const uint64_t h = hash_compute_pipeline_desc(d);
            if (const VulkanPipelineRecord* hit = compute_.find(h))
            {
                stats_.cache_hits++;
                return hit->id;
            }
            const uint64_t id = kComputePipelineIdBase + ++next_compute_id_;
            if (!create(id, d)) return 0;
            stats_.compute_pipeline_creates++;
            compute_.insert_or_assign(h, VulkanPipelineRecord{id, h});
            compute_hash_by_id_.insert_or_assign(id, h);
            return id;
        }

        [[nodiscard]] const VulkanPipelineRecord* find_graphics(uint64_t id) const
        {
            const uint64_t* hash = graphics_hash_by_id_.find(id);
            return hash ? graphics_.find(*hash) : nullptr;
        }

        [[nodiscard]] const VulkanPipelineRecord* find_compute(uint64_t id) const
        {
            const uint64_t* hash = compute_hash_by_id_.find(id);
            return hash ? compute_.find(*hash) : nullptr;
        }

        [[nodiscard]] const VulkanPipelineStats& stats() const { return stats_; }

    private:
        using RecordMap = containers::FlatMap<uint64_t, VulkanPipelineRecord>;
        RecordMap shader_modules_;
        RecordMap graphics_;
        RecordMap compute_;
        // Descriptor hashes dedupe creation; public IDs resolve through this index.
        containers::FlatMap<uint64_t, uint64_t> graphics_hash_by_id_;
        containers::FlatMap<uint64_t, uint64_t> compute_hash_by_id_;
        uint64_t next_shader_module_id_ = 0;
        uint64_t next_graphics_id_ = 0;
        uint64_t next_compute_id_ = 0;
        VulkanPipelineStats stats_{};
    };

    // ------------------------------------------------------------------
    // Pure create-info builders (GPU-free testable).
    // ------------------------------------------------------------------

    [[nodiscard]] inline VkShaderModuleCreateInfo vk_shader_module_create_info(const RHIShaderModuleDesc& d)
    {
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = d.bytecode_size;
        ci.pCode = static_cast<const uint32_t*>(d.bytecode);
        return ci;
    }

    [[nodiscard]] inline VkPipelineShaderStageCreateInfo vk_pipeline_shader_stage(const RHIShaderModuleDesc& d)
    {
        VkPipelineShaderStageCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage = vk_shader_stage_of(d.stage);
        ci.pName = d.entry;
        return ci;
    }

    [[nodiscard]] inline VkPipelineRasterizationStateCreateInfo vk_raster_state(const RHIRasterStateDesc& d)
    {
        VkPipelineRasterizationStateCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        ci.polygonMode = VK_POLYGON_MODE_FILL;
        ci.cullMode = vk_cull_mode_of(d.cull);
        ci.frontFace = vk_front_face_of(d.front_face);
        ci.depthClampEnable = d.depth_clamp ? VK_TRUE : VK_FALSE;
        ci.lineWidth = 1.0f;
        return ci;
    }

    [[nodiscard]] inline VkPipelineDepthStencilStateCreateInfo vk_depth_state(const RHIDepthStateDesc& d)
    {
        VkPipelineDepthStencilStateCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ci.depthTestEnable = d.enable_test ? VK_TRUE : VK_FALSE;
        ci.depthWriteEnable = d.enable_write ? VK_TRUE : VK_FALSE;
        ci.depthCompareOp = VK_COMPARE_OP_LESS;
        return ci;
    }

    // attachment is caller-owned storage (pAttachments must outlive the call).
    [[nodiscard]] inline VkPipelineColorBlendStateCreateInfo vk_blend_state(const RHIBlendStateDesc& d, VkPipelineColorBlendAttachmentState& attachment)
    {
        attachment = VkPipelineColorBlendAttachmentState{};
        attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        attachment.blendEnable = d.enable ? VK_TRUE : VK_FALSE;
        if (d.enable)
        {
            attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            attachment.colorBlendOp = VK_BLEND_OP_ADD;
            attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            attachment.alphaBlendOp = VK_BLEND_OP_ADD;
        }
        VkPipelineColorBlendStateCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        ci.attachmentCount = 1;
        ci.pAttachments = &attachment;
        return ci;
    }
}
