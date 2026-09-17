#pragma once

/* SHS RENDERER SAN — Explicit G2 graphics realization; no lazy creation. */
#include <cstring>
#include <limits>
#include "shs/execution/rhi/drivers/vulkan/vk_offscreen.hpp"
#include "shs/execution/rhi/drivers/vulkan/vk_pipelines.hpp"

namespace shs
{
    // Fixed ABI: triangle list, no vertex attributes, descriptors or push constants,
    // one RGBA8 output, no depth/blending, static viewport/scissor.
    // Caller supplies trusted, validated SPIR-V matching that ABI. supports() checks
    // descriptor/header shape only; it is NOT a SPIR-V validator or reflection pass.
    // Device must outlive this object; retire command references before reset().
    // Use initialize() in the cache's explicit create hook; retain this owner
    // separately. A cached ID alone never authorizes a bind.
    class VulkanOffscreenPipeline
    {
    public:
        VulkanOffscreenPipeline() = default;
        ~VulkanOffscreenPipeline() { reset(); }
        VulkanOffscreenPipeline(const VulkanOffscreenPipeline&) = delete;
        VulkanOffscreenPipeline& operator=(const VulkanOffscreenPipeline&) = delete;

        [[nodiscard]] static bool supports(const RHIGraphicsPipelineDesc& d)
        {
            return shader_supported(d.vs, RHIShaderStage::Vertex) &&
                shader_supported(d.fs, RHIShaderStage::Fragment) &&
                d.rt.color_format == RHIFormat::RGBA8_UNorm && !d.rt.has_depth &&
                !d.depth.enable_test && !d.depth.enable_write && !d.blend.enable &&
                !d.raster.depth_clamp &&
                (d.raster.cull == RHICullMode::None || d.raster.cull == RHICullMode::Back ||
                 d.raster.cull == RHICullMode::Front) &&
                (d.raster.front_face == RHIFrontFace::CCW || d.raster.front_face == RHIFrontFace::CW);
        }

        [[nodiscard]] bool initialize(uint64_t id, const RHIGraphicsPipelineDesc& d,
                                      const VulkanOffscreenPass& pass)
        {
            if (device_ || !id || !pass.device() || !pass.render_pass() || !supports(d)) return false;
            device_ = pass.device();
            VkShaderModule modules[2]{};
            const auto release_modules = [&] {
                for (auto module : modules) if (module) vkDestroyShaderModule(device_, module, nullptr);
            };
            const RHIShaderModuleDesc shaders[] = {d.vs, d.fs};
            VkPipelineShaderStageCreateInfo stages[2]{};
            for (int i = 0; i < 2; ++i)
            {
                const auto ci = vk_shader_module_create_info(shaders[i]);
                if (vkCreateShaderModule(device_, &ci, nullptr, &modules[i]) != VK_SUCCESS)
                {
                    release_modules(); reset(); return false;
                }
                stages[i] = vk_pipeline_shader_stage(shaders[i]);
                stages[i].module = modules[i];
            }
            VkPipelineLayoutCreateInfo li{};
            li.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            if (vkCreatePipelineLayout(device_, &li, nullptr, &layout_) != VK_SUCCESS)
            {
                release_modules(); reset(); return false;
            }
            const auto extent = pass.extent();
            VkViewport viewport{0, 0, static_cast<float>(extent.width), static_cast<float>(extent.height), 0, 1};
            VkRect2D scissor{{0, 0}, extent};
            VkPipelineViewportStateCreateInfo vp{};
            vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            vp.viewportCount = vp.scissorCount = 1;
            vp.pViewports = &viewport;
            vp.pScissors = &scissor;
            VkPipelineVertexInputStateCreateInfo vertex{};
            vertex.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            VkPipelineInputAssemblyStateCreateInfo assembly{};
            assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            auto raster = vk_raster_state(d.raster);
            auto depth = vk_depth_state(d.depth);
            VkPipelineColorBlendAttachmentState attachment{};
            auto blend = vk_blend_state(d.blend, attachment);
            VkPipelineMultisampleStateCreateInfo samples{};
            samples.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            samples.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            VkGraphicsPipelineCreateInfo pi{};
            pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pi.stageCount = 2;
            pi.pStages = stages;
            pi.pVertexInputState = &vertex;
            pi.pInputAssemblyState = &assembly;
            pi.pViewportState = &vp;
            pi.pRasterizationState = &raster;
            pi.pMultisampleState = &samples;
            pi.pDepthStencilState = &depth;
            pi.pColorBlendState = &blend;
            pi.layout = layout_;
            pi.renderPass = pass.render_pass();
            const auto result = vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &pipeline_);
            release_modules();
            if (result != VK_SUCCESS) { reset(); return false; }
            id_ = id;
            desc_hash_ = hash_graphics_pipeline_desc(d);
            extent_ = extent;
            return true;
        }

        void reset()
        {
            if (pipeline_) vkDestroyPipeline(device_, pipeline_, nullptr);
            if (layout_) vkDestroyPipelineLayout(device_, layout_, nullptr);
            pipeline_ = VK_NULL_HANDLE;
            layout_ = VK_NULL_HANDLE;
            device_ = VK_NULL_HANDLE;
            id_ = desc_hash_ = 0;
            extent_ = {};
        }

        // Offscreen passes share a compatible layout; baked viewport/scissor
        // additionally require an identical extent. Never bind bookkeeping alone.
        [[nodiscard]] bool accepts(const VulkanPipelineRecord& record, const VulkanOffscreenPass& pass) const
        {
            return pipeline_ && record.id == id_ && record.desc_hash == desc_hash_ &&
                device_ == pass.device() && pass.render_pass() &&
                extent_.width == pass.extent().width && extent_.height == pass.extent().height;
        }
        [[nodiscard]] VkPipeline pipeline() const { return pipeline_; }
        [[nodiscard]] VkPipelineLayout layout() const { return layout_; }

    private:
        [[nodiscard]] static bool shader_supported(const RHIShaderModuleDesc& d, RHIShaderStage stage)
        {
            if (d.stage != stage || !d.bytecode || d.bytecode_size < 20 || d.bytecode_size % 4 ||
                d.bytecode_size > std::numeric_limits<size_t>::max() ||
                reinterpret_cast<uintptr_t>(d.bytecode) % alignof(uint32_t) || !d.entry || !*d.entry) return false;
            uint32_t header[5]{};
            std::memcpy(header, d.bytecode, sizeof(header));
            // Vulkan 1.1 core supports SPIR-V through 1.3 (no new extensions).
            return header[0] == 0x07230203 && header[1] >= 0x00010000 &&
                header[1] <= 0x00010300 && header[3] != 0 && header[4] == 0;
        }
        VkDevice device_ = VK_NULL_HANDLE;
        VkPipeline pipeline_ = VK_NULL_HANDLE;
        VkPipelineLayout layout_ = VK_NULL_HANDLE;
        VkExtent2D extent_{};
        uint64_t id_ = 0;
        uint64_t desc_hash_ = 0;
    };
}
