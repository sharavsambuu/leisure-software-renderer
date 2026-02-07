#define SDL_MAIN_HANDLED

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <vulkan/vulkan.h>

#include <shs/core/context.hpp>
#include <shs/rhi/backend/backend_factory.hpp>
#include <shs/rhi/drivers/vulkan/vk_backend.hpp>

namespace
{
constexpr int kDefaultW = 960;
constexpr int kDefaultH = 640;

std::vector<char> read_file(const char* path)
{
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    if (!f.is_open()) throw std::runtime_error(std::string("Failed to open file: ") + path);
    const size_t sz = static_cast<size_t>(f.tellg());
    if (sz == 0) throw std::runtime_error(std::string("Empty shader file: ") + path);
    std::vector<char> bytes(sz);
    f.seekg(0);
    f.read(bytes.data(), static_cast<std::streamsize>(sz));
    return bytes;
}

VkShaderModule create_shader_module(VkDevice dev, const std::vector<char>& code)
{
    if ((code.size() % 4) != 0) throw std::runtime_error("SPIR-V size is not multiple of 4");
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = code.size();
    ci.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule out = VK_NULL_HANDLE;
    if (vkCreateShaderModule(dev, &ci, nullptr, &out) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateShaderModule failed");
    }
    return out;
}

class HelloVulkanTriangleApp
{
public:
    ~HelloVulkanTriangleApp()
    {
        cleanup();
    }

    void run()
    {
        init_sdl();
        init_backend();
        create_pipeline();
        main_loop();
    }

private:
    void init_sdl()
    {
        if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0)
        {
            throw std::runtime_error(std::string("SDL_Init failed: ") + SDL_GetError());
        }
        sdl_ready_ = true;
        win_ = SDL_CreateWindow(
            "HelloVulkanTriangle",
            SDL_WINDOWPOS_CENTERED,
            SDL_WINDOWPOS_CENTERED,
            kDefaultW,
            kDefaultH,
            SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE
        );
        if (!win_) throw std::runtime_error(std::string("SDL_CreateWindow failed: ") + SDL_GetError());
    }

    void init_backend()
    {
        shs::RenderBackendCreateResult created = shs::create_render_backend(shs::RenderBackendType::Vulkan);
        if (!created.note.empty())
        {
            std::fprintf(stderr, "[shs] %s\n", created.note.c_str());
        }
        if (!created.backend)
        {
            throw std::runtime_error("Backend factory did not return a backend");
        }
        keep_.push_back(std::move(created.backend));
        for (auto& aux : created.auxiliary_backends)
        {
            if (aux) keep_.push_back(std::move(aux));
        }
        for (const auto& b : keep_)
        {
            ctx_.register_backend(b.get());
        }

        if (created.active != shs::RenderBackendType::Vulkan)
        {
            throw std::runtime_error("Vulkan backend is not active in this build/configuration.");
        }

        vk_ = dynamic_cast<shs::VulkanRenderBackend*>(ctx_.backend(shs::RenderBackendType::Vulkan));
        if (!vk_)
        {
            throw std::runtime_error("Factory returned non-Vulkan backend instance for Vulkan request.");
        }

        int dw = 0;
        int dh = 0;
        SDL_Vulkan_GetDrawableSize(win_, &dw, &dh);
        if (dw <= 0 || dh <= 0)
        {
            dw = kDefaultW;
            dh = kDefaultH;
        }

        shs::VulkanRenderBackend::InitDesc init{};
        init.window = win_;
        init.width = dw;
        init.height = dh;
        init.enable_validation = true;
        init.app_name = "HelloVulkanTriangle";
        if (!vk_->init_sdl(init))
        {
            throw std::runtime_error("Vulkan backend init_sdl failed");
        }

        ctx_.set_primary_backend(vk_);
        std::fprintf(stderr, "[shs] active backend: %s\n", ctx_.active_backend_name());
    }

    void create_pipeline()
    {
        destroy_pipeline();
        const VkDevice dev = vk_ ? vk_->device() : VK_NULL_HANDLE;
        if (dev == VK_NULL_HANDLE) throw std::runtime_error("Vulkan device not ready");
        if (vk_->render_pass() == VK_NULL_HANDLE) throw std::runtime_error("Vulkan render pass not ready");

        const std::vector<char> vs_code = read_file(SHS_VK_TRIANGLE_VERT_SPV);
        const std::vector<char> fs_code = read_file(SHS_VK_TRIANGLE_FRAG_SPV);
        VkShaderModule vs = create_shader_module(dev, vs_code);
        VkShaderModule fs = create_shader_module(dev, fs_code);

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vs;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fs;
        stages[1].pName = "main";

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vp_state{};
        vp_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp_state.viewportCount = 1;
        vp_state.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        // Viewport is y-flipped (negative height), so set front-face to keep
        // winding convention aligned with software rasterizer's CCW-in-NDC intent.
        rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rs.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = VK_FALSE;
        ds.depthWriteEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT;
        cba.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1;
        cb.pAttachments = &cba;

        const VkDynamicState dyn_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dyn{};
        dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dyn.dynamicStateCount = 2;
        dyn.pDynamicStates = dyn_states;

        VkPipelineLayoutCreateInfo pl{};
        pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        if (vkCreatePipelineLayout(dev, &pl, nullptr, &pipeline_layout_) != VK_SUCCESS)
        {
            vkDestroyShaderModule(dev, vs, nullptr);
            vkDestroyShaderModule(dev, fs, nullptr);
            throw std::runtime_error("vkCreatePipelineLayout failed");
        }

        VkGraphicsPipelineCreateInfo gp{};
        gp.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gp.stageCount = 2;
        gp.pStages = stages;
        gp.pVertexInputState = &vi;
        gp.pInputAssemblyState = &ia;
        gp.pViewportState = &vp_state;
        gp.pRasterizationState = &rs;
        gp.pMultisampleState = &ms;
        gp.pDepthStencilState = &ds;
        gp.pColorBlendState = &cb;
        gp.pDynamicState = &dyn;
        gp.layout = pipeline_layout_;
        gp.renderPass = vk_->render_pass();
        gp.subpass = 0;
        if (vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, nullptr, &pipeline_) != VK_SUCCESS)
        {
            vkDestroyShaderModule(dev, vs, nullptr);
            vkDestroyShaderModule(dev, fs, nullptr);
            throw std::runtime_error("vkCreateGraphicsPipelines failed");
        }
        vkDestroyShaderModule(dev, vs, nullptr);
        vkDestroyShaderModule(dev, fs, nullptr);

        pipeline_gen_ = vk_->swapchain_generation();
    }

    void destroy_pipeline()
    {
        if (!vk_) return;
        const VkDevice dev = vk_->device();
        if (dev == VK_NULL_HANDLE)
        {
            pipeline_ = VK_NULL_HANDLE;
            pipeline_layout_ = VK_NULL_HANDLE;
            return;
        }
        if (pipeline_ != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(dev, pipeline_, nullptr);
            pipeline_ = VK_NULL_HANDLE;
        }
        if (pipeline_layout_ != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(dev, pipeline_layout_, nullptr);
            pipeline_layout_ = VK_NULL_HANDLE;
        }
    }

    void main_loop()
    {
        bool running = true;
        while (running)
        {
            SDL_Event e{};
            while (SDL_PollEvent(&e))
            {
                if (e.type == SDL_QUIT) running = false;
                if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE) running = false;
                if (e.type == SDL_WINDOWEVENT &&
                    (e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED || e.window.event == SDL_WINDOWEVENT_RESIZED))
                {
                    vk_->request_resize(e.window.data1, e.window.data2);
                }
            }
            draw_frame();
        }

        if (vk_ && vk_->device() != VK_NULL_HANDLE)
        {
            (void)vkDeviceWaitIdle(vk_->device());
        }
    }

    void draw_frame()
    {
        int dw = 0;
        int dh = 0;
        SDL_Vulkan_GetDrawableSize(win_, &dw, &dh);
        if (dw <= 0 || dh <= 0)
        {
            SDL_Delay(16);
            return;
        }

        shs::RenderBackendFrameInfo frame{};
        frame.frame_index = ctx_.frame_index;
        frame.width = dw;
        frame.height = dh;

        shs::VulkanRenderBackend::FrameInfo fi{};
        if (!vk_->begin_frame(ctx_, frame, fi))
        {
            SDL_Delay(4);
            return;
        }

        if (pipeline_ == VK_NULL_HANDLE || pipeline_gen_ != vk_->swapchain_generation())
        {
            create_pipeline();
        }

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(fi.cmd, &bi) != VK_SUCCESS)
        {
            throw std::runtime_error("vkBeginCommandBuffer failed");
        }

        VkClearValue clear{};
        clear.color = {{0.04f, 0.05f, 0.09f, 1.0f}};
        VkRenderPassBeginInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp.renderPass = fi.render_pass;
        rp.framebuffer = fi.framebuffer;
        rp.renderArea.offset = {0, 0};
        rp.renderArea.extent = fi.extent;
        rp.clearValueCount = 1;
        rp.pClearValues = &clear;
        vkCmdBeginRenderPass(fi.cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(fi.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

        VkViewport vp{};
        vp.x = 0.0f;
        // Vulkan framebuffer coordinates are top-left based by default.
        // Use negative viewport height so NDC +Y stays "up" (same convention as software path).
        vp.y = static_cast<float>(fi.extent.height);
        vp.width = static_cast<float>(fi.extent.width);
        vp.height = -static_cast<float>(fi.extent.height);
        vp.minDepth = 0.0f;
        vp.maxDepth = 1.0f;
        const VkRect2D scissor{{0, 0}, fi.extent};
        vkCmdSetViewport(fi.cmd, 0, 1, &vp);
        vkCmdSetScissor(fi.cmd, 0, 1, &scissor);
        vkCmdDraw(fi.cmd, 3, 1, 0, 0);
        vkCmdEndRenderPass(fi.cmd);

        if (vkEndCommandBuffer(fi.cmd) != VK_SUCCESS)
        {
            throw std::runtime_error("vkEndCommandBuffer failed");
        }

        vk_->end_frame(fi);
        ctx_.frame_index++;
    }

    void cleanup()
    {
        if (cleaned_up_) return;
        cleaned_up_ = true;

        if (vk_ && vk_->device() != VK_NULL_HANDLE)
        {
            (void)vkDeviceWaitIdle(vk_->device());
        }
        destroy_pipeline();
        keep_.clear();
        vk_ = nullptr;

        if (win_)
        {
            SDL_DestroyWindow(win_);
            win_ = nullptr;
        }
        if (sdl_ready_)
        {
            SDL_Quit();
            sdl_ready_ = false;
        }
    }

private:
    bool cleaned_up_ = false;
    bool sdl_ready_ = false;
    SDL_Window* win_ = nullptr;
    shs::Context ctx_{};
    std::vector<std::unique_ptr<shs::IRenderBackend>> keep_{};
    shs::VulkanRenderBackend* vk_ = nullptr;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    uint64_t pipeline_gen_ = 0;
};
}

int main()
{
    try
    {
        HelloVulkanTriangleApp app{};
        app.run();
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "Fatal: %s\n", e.what());
        return 1;
    }
}
