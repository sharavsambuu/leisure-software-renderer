#pragma once

/*
    SHS RENDERER SAN
    FILE: vk_offscreen.hpp
    MODULE: rhi/drivers/vulkan
    PURPOSE: Explicit, owned Vulkan 1.1 offscreen attachment realization.
    G2 first slice: RGBA8, one mip/layer, clear black, no depth. No lazy creation.
*/

#include "shs/rhi/vulkan/value/vk_resources.hpp"

namespace shs
{
    class VulkanOffscreenPass
    {
    public:
        VulkanOffscreenPass() = default;
        ~VulkanOffscreenPass() { reset(); }
        VulkanOffscreenPass(const VulkanOffscreenPass&) = delete;
        VulkanOffscreenPass& operator=(const VulkanOffscreenPass&) = delete;

        [[nodiscard]] static bool supports(const RHIImageDesc& d)
        {
            return d.type == RHIImageType::Tex2D && d.format == RHIFormat::RGBA8_UNorm &&
                d.width > 0 && d.height > 0 && d.mip_levels == 1 && d.layers == 1 &&
                (d.usage & RHIImageUsage_ColorAttachment) != 0;
        }

        // Borrowed device/image must outlive this object and all submitted work.
        // Re-preparation is refused: the caller must retire work and reset first.
        [[nodiscard]] bool initialize(VkDevice device, uint64_t id, VkImage image, const RHIImageDesc& d)
        {
            if (device_ || !device || !id || !image || !supports(d)) return false;
            device_ = device;
            VkImageViewCreateInfo vi{};
            vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vi.image = image;
            vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format = VK_FORMAT_R8G8B8A8_UNORM;
            vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            if (vkCreateImageView(device_, &vi, nullptr, &view_) != VK_SUCCESS) { reset(); return false; }

            VkAttachmentDescription attachment{};
            attachment.format = vi.format;
            attachment.samples = VK_SAMPLE_COUNT_1_BIT;
            attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            VkAttachmentReference color{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
            VkSubpassDescription subpass{};
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.colorAttachmentCount = 1;
            subpass.pColorAttachments = &color;
            // Serialize repeated clears/writes even though prior contents are discarded.
            VkSubpassDependency dependency{};
            dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
            dependency.dstSubpass = 0;
            dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            VkRenderPassCreateInfo ri{};
            ri.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            ri.attachmentCount = 1;
            ri.pAttachments = &attachment;
            ri.subpassCount = 1;
            ri.pSubpasses = &subpass;
            ri.dependencyCount = 1;
            ri.pDependencies = &dependency;
            if (vkCreateRenderPass(device_, &ri, nullptr, &pass_) != VK_SUCCESS) { reset(); return false; }

            VkFramebufferCreateInfo fi{};
            fi.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            fi.renderPass = pass_;
            fi.attachmentCount = 1;
            fi.pAttachments = &view_;
            fi.width = static_cast<uint32_t>(d.width);
            fi.height = static_cast<uint32_t>(d.height);
            fi.layers = 1;
            if (vkCreateFramebuffer(device_, &fi, nullptr, &framebuffer_) != VK_SUCCESS) { reset(); return false; }
            extent_ = {fi.width, fi.height};
            target_ = id;
            return true;
        }

        // Caller must ensure no pending command buffer refers to these objects.
        void reset()
        {
            if (framebuffer_) vkDestroyFramebuffer(device_, framebuffer_, nullptr);
            if (pass_) vkDestroyRenderPass(device_, pass_, nullptr);
            if (view_) vkDestroyImageView(device_, view_, nullptr);
            framebuffer_ = VK_NULL_HANDLE;
            pass_ = VK_NULL_HANDLE;
            view_ = VK_NULL_HANDLE;
            device_ = VK_NULL_HANDLE;
            target_ = 0;
            extent_ = {};
        }

        [[nodiscard]] bool accepts(const RHICmdBeginPassDesc& d) const
        {
            return framebuffer_ && d.color_target == target_ && !d.depth_target &&
                d.clear_color && !d.clear_depth;
        }
        [[nodiscard]] VkDevice device() const { return device_; }
        [[nodiscard]] VkRenderPass render_pass() const { return pass_; }
        [[nodiscard]] VkFramebuffer framebuffer() const { return framebuffer_; }
        [[nodiscard]] VkExtent2D extent() const { return extent_; }

    private:
        VkDevice device_ = VK_NULL_HANDLE;
        VkImageView view_ = VK_NULL_HANDLE;
        VkRenderPass pass_ = VK_NULL_HANDLE;
        VkFramebuffer framebuffer_ = VK_NULL_HANDLE;
        VkExtent2D extent_{};
        uint64_t target_ = 0;
    };
}
