#pragma once

/*
    exps-rendering-adventures — tier0 common (Vulkan side).

    Minimal offscreen Vulkan harness shared by all tier0 *_vk demos:
    instance/device (no swapchain, no window), one render pass over an
    R8G8B8A8_UNORM color image + D24_UNORM_S8_UINT depth-stencil image,
    vertex buffer, push-constant MVP, optional combined-image-sampler
    descriptor (demo 04), pixel readback -> PNG.

    Conventions (pins from docs/roadmap/slang_utilization_plan.md):
      * NEGATIVE viewport height — NDC +Y renders at the top, matching the
        software twins' y-flip.
      * Depth range [0, 1] (GLM_FORCE_DEPTH_ZERO_TO_ONE in the demos).
      * Shaders are Slang-compiled SPIR-V, loaded via the library's
        vk_shader_utils (loader only, compiler-agnostic).
*/

#include <cstdint>
#include <functional>
#include <vector>

#include <vulkan/vulkan.h>

namespace adventures
{
    struct VkPipelineSetup
    {
        const char* vs_spv_path  = nullptr; // SPIR-V from slangc
        const char* fs_spv_path  = nullptr;
        bool        blend        = false;   // SRC_ALPHA / ONE_MINUS_SRC_ALPHA
        bool        depth_test   = true;
        bool        depth_write  = true;
        bool        textured     = false;   // bind combined image sampler (set 0)
        // stencil (applied to front & back faces)
        bool        stencil_test   = false;
        bool        stencil_write  = false;
        bool        stencil_invert = false; // NOT_EQUAL instead of EQUAL
        uint8_t     stencil_ref    = 1;
    };

    struct VkDraw
    {
        uint32_t    pipeline     = 0;
        uint32_t    first_vertex = 0;
        uint32_t    vertex_count = 0;
        const void* push         = nullptr; // must match the shader push-constant block
        uint32_t    push_size    = 0;
        int         texture_set  = -1;      // index into texture sets (textured pipelines)
    };

    class OffscreenVulkan
    {
    public:
        OffscreenVulkan() = default;
        ~OffscreenVulkan();
        OffscreenVulkan(const OffscreenVulkan&) = delete;
        OffscreenVulkan& operator=(const OffscreenVulkan&) = delete;

        bool init(uint32_t width, uint32_t height);
        int add_pipeline(const VkPipelineSetup& setup); // returns pipeline index or -1
        bool upload_vertices(const void* data, size_t byte_size);
        // Creates one descriptor set (set 0, binding 0 = combined image sampler)
        // with the requested filtering; returns set index or -1.
        int upload_texture_rgba(const uint8_t* rgba, uint32_t w, uint32_t h, bool bilinear);
        bool render(const std::vector<VkDraw>& draws, const VkRect2D* scissor_override = nullptr);
        bool save_png(const char* path) const;

        uint32_t width() const { return width_; }
        uint32_t height() const { return height_; }

    private:
        uint32_t find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags props) const;
        bool create_image(uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage,
                          VkImage* image, VkDeviceMemory* memory) const;
        bool init_render_targets(); // render pass, images, views, framebuffer, layout
        bool record_one_shot(const std::function<void(VkCommandBuffer)>& recorder) const;
        bool submit_and_wait() const;

        uint32_t width_  = 0;
        uint32_t height_ = 0;

        VkInstance       instance_     = VK_NULL_HANDLE;
        VkPhysicalDevice physical_     = VK_NULL_HANDLE;
        VkDevice         device_       = VK_NULL_HANDLE;
        uint32_t         queue_family_ = 0;
        VkQueue          queue_        = VK_NULL_HANDLE;
        VkCommandPool    cmd_pool_     = VK_NULL_HANDLE;
        VkCommandBuffer  cmd_          = VK_NULL_HANDLE;

        VkRenderPass          render_pass_        = VK_NULL_HANDLE;
        VkPipelineLayout      pipeline_layout_    = VK_NULL_HANDLE;
        VkDescriptorSetLayout texture_set_layout_ = VK_NULL_HANDLE;
        VkDescriptorPool      descriptor_pool_    = VK_NULL_HANDLE;

        VkImage        color_image_  = VK_NULL_HANDLE;
        VkDeviceMemory color_memory_ = VK_NULL_HANDLE;
        VkImageView    color_view_   = VK_NULL_HANDLE;
        VkImage        depth_image_  = VK_NULL_HANDLE;
        VkDeviceMemory depth_memory_ = VK_NULL_HANDLE;
        VkImageView    depth_view_   = VK_NULL_HANDLE;
        VkFramebuffer  framebuffer_  = VK_NULL_HANDLE;

        VkBuffer       vertex_buffer_ = VK_NULL_HANDLE;
        VkDeviceMemory vertex_memory_ = VK_NULL_HANDLE;
        size_t         vertex_bytes_  = 0;

        struct TextureSet
        {
            VkImage         image   = VK_NULL_HANDLE;
            VkDeviceMemory  memory  = VK_NULL_HANDLE;
            VkImageView     view    = VK_NULL_HANDLE;
            VkSampler       sampler = VK_NULL_HANDLE;
            VkDescriptorSet set     = VK_NULL_HANDLE;
        };
        std::vector<TextureSet> texture_sets_{};

        std::vector<VkPipeline> pipelines_{};
        std::vector<char> color_readback_{};

        // per-pipeline push-constant sizes etc. resolved at render time
        static constexpr uint32_t kMaxPushConstants = 128;
    };
}
