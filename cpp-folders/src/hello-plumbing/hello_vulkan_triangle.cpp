#define SDL_MAIN_HANDLED

#include <algorithm>
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
constexpr uint32_t WIN_W = 960;
constexpr uint32_t WIN_H = 640;

std::vector<char> read_file(const char* path)
{
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    if (!f.is_open()) throw std::runtime_error(std::string("Failed to open file: ") + path);
    const size_t sz = static_cast<size_t>(f.tellg());
    if (sz == 0) throw std::runtime_error(std::string("Empty file: ") + path);
    std::vector<char> out(sz);
    f.seekg(0);
    f.read(out.data(), static_cast<std::streamsize>(sz));
    return out;
}

VkShaderModule make_shader_module(VkDevice dev, const std::vector<char>& code)
{
    if ((code.size() % 4) != 0) throw std::runtime_error("Shader bytecode size not multiple of 4");
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = code.size();
    ci.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(dev, &ci, nullptr, &mod) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateShaderModule failed");
    }
    return mod;
}

class TriangleApp
{
public:
    void run()
    {
        init_sdl();
        init_backend();
        create_pipeline();
        loop();
        cleanup();
    }

private:
    void init_sdl()
    {
        if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0)
        {
            throw std::runtime_error(std::string("SDL_Init failed: ") + SDL_GetError());
        }
        win_ = SDL_CreateWindow("HelloVulkanTriangle", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
            static_cast<int>(WIN_W), static_cast<int>(WIN_H), SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
        if (!win_) throw std::runtime_error(std::string("SDL_CreateWindow failed: ") + SDL_GetError());
    }

    void init_backend()
    {
        auto backend = shs::create_render_backend("vulkan");
        keep_.push_back(std::move(backend.backend));
        for (auto& b : backend.auxiliary_backends) keep_.push_back(std::move(b));
        if (keep_.empty() || !keep_[0]) throw std::runtime_error("No Vulkan backend created");

        vk_ = dynamic_cast<shs::VulkanRenderBackend*>(keep_[0].get());
        if (!vk_) throw std::runtime_error("Vulkan backend type mismatch");

        int dw = 0;
        int dh = 0;
        SDL_Vulkan_GetDrawableSize(win_, &dw, &dh);

        shs::VulkanRenderBackend::InitDesc init{};
        init.window = win_;
        init.width = dw;
        init.height = dh;
        init.enable_validation = true;
        init.app_name = "HelloVulkanTriangle";
        if (!vk_->init_sdl(init)) throw std::runtime_error("Vulkan backend init failed");

        ctx_.set_primary_backend(vk_);
        std::fprintf(stderr, "[shs] active backend: %s\n", vk_->name());
    }

    void create_pipeline()
    {
        destroy_pipeline();
        const VkDevice dev = vk_->device();
        if (dev == VK_NULL_HANDLE) throw std::runtime_error("Vulkan device not ready");
        if (vk_->render_pass() == VK_NULL_HANDLE) throw std::runtime_error("Vulkan render pass not ready");

        auto vs_code = read_file(SHS_VK_TRIANGLE_VERT_SPV);
        auto fs_code = read_file(SHS_VK_TRIANGLE_FRAG_SPV);
        VkShaderModule vs = make_shader_module(dev, vs_code);
        VkShaderModule fs = make_shader_module(dev, fs_code);

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
        rs.cullMode = VK_CULL_MODE_BACK_BIT;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = VK_FALSE;
        ds.depthWriteEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState cb_att{};
        cb_att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1;
        cb.pAttachments = &cb_att;

        VkDynamicState dyn_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
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
        VkDevice dev = vk_->device();
        if (pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(dev, pipeline_, nullptr);
        if (pipeline_layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(dev, pipeline_layout_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
        pipeline_layout_ = VK_NULL_HANDLE;
    }

    void loop()
    {
        bool running = true;
        while (running)
        {
            SDL_Event e{};
            while (SDL_PollEvent(&e))
            {
                if (e.type == SDL_QUIT) running = false;
                if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE) running = false;
                if (e.type == SDL_WINDOWEVENT && e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)
                {
                    vk_->request_resize(e.window.data1, e.window.data2);
                }
            }
            draw();
        }
        if (vk_ && vk_->device() != VK_NULL_HANDLE) vkDeviceWaitIdle(vk_->device());
    }

    void draw()
    {
        shs::RenderBackendFrameInfo frame{};
        frame.frame_index = ctx_.frame_index;
        frame.width = static_cast<int>(WIN_W);
        frame.height = static_cast<int>(WIN_H);

        shs::VulkanRenderBackend::FrameInfo fi{};
        if (!vk_->begin_frame(ctx_, frame, fi))
        {
            SDL_Delay(8);
            return;
        }

        if (pipeline_ == VK_NULL_HANDLE || pipeline_gen_ != vk_->swapchain_generation())
        {
            create_pipeline();
        }

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        if (vkBeginCommandBuffer(fi.cmd, &bi) != VK_SUCCESS) throw std::runtime_error("vkBeginCommandBuffer failed");

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
        vp.y = 0.0f;
        vp.width = static_cast<float>(fi.extent.width);
        vp.height = static_cast<float>(fi.extent.height);
        vp.minDepth = 0.0f;
        vp.maxDepth = 1.0f;
        VkRect2D scissor{{0, 0}, fi.extent};
        vkCmdSetViewport(fi.cmd, 0, 1, &vp);
        vkCmdSetScissor(fi.cmd, 0, 1, &scissor);
        vkCmdDraw(fi.cmd, 3, 1, 0, 0);
        vkCmdEndRenderPass(fi.cmd);

        if (vkEndCommandBuffer(fi.cmd) != VK_SUCCESS) throw std::runtime_error("vkEndCommandBuffer failed");
        vk_->end_frame(fi);
        ctx_.frame_index++;
    }

    void cleanup()
    {
        destroy_pipeline();
        keep_.clear();
        if (win_) SDL_DestroyWindow(win_);
        SDL_Quit();
    }

private:
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
        TriangleApp app{};
        app.run();
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "Fatal: %s\n", e.what());
        return 1;
    }
}
