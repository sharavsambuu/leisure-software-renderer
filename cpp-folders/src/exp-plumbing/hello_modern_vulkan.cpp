#define SDL_MAIN_HANDLED
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <random>

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <shs/core/context.hpp>
#include <shs/rhi/backend/backend_factory.hpp>
#include <shs/rhi/drivers/vulkan/vk_backend.hpp>
#include <shs/rhi/drivers/vulkan/vk_cmd_utils.hpp>
#include <shs/rhi/drivers/vulkan/vk_shader_utils.hpp>
#include <shs/rhi/drivers/vulkan/vk_memory_utils.hpp>
#include <shs/rhi/drivers/vulkan/vk_render_path_descriptors.hpp>
#include <shs/rhi/drivers/vulkan/vk_vma.hpp>

namespace
{
constexpr int kDefaultW = 1280;
constexpr int kDefaultH = 720;
constexpr uint32_t kMaxInstances = 20000;
constexpr uint32_t kMaxTextures = 4096;

struct InstanceData
{
    glm::mat4 model;
    uint32_t texture_index;
    float padding[3];
};

class HelloModernVulkanApp
{
public:
    ~HelloModernVulkanApp() { cleanup(); }

    void run()
    {
        init_sdl();
        init_backend();
        create_resources();
        create_pipeline();
        main_loop();
    }

private:
    void init_sdl()
    {
        if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0)
            throw std::runtime_error("SDL_Init failed");
        sdl_ready_ = true;
        win_ = SDL_CreateWindow("HelloModernVulkan (Bindless + Dynamic Rendering + VMA)", 
            SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, kDefaultW, kDefaultH, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
        if (!win_) throw std::runtime_error("SDL_CreateWindow failed");
    }

    void init_backend()
    {
        auto created = shs::create_render_backend(shs::RenderBackendType::Vulkan);
        vk_ = dynamic_cast<shs::VulkanRenderBackend*>(created.backend.get());
        keep_.push_back(std::move(created.backend));
        ctx_.register_backend(vk_);

        shs::VulkanRenderBackend::InitDesc init{};
        init.window = win_;
        init.width = kDefaultW;
        init.height = kDefaultH;
        init.enable_validation = true;
        if (!vk_->init(init)) throw std::runtime_error("Vulkan init failed");
        
        if (!vk_->capabilities().features.dynamic_rendering)
            throw std::runtime_error("Dynamic Rendering not supported on this GPU!");
        if (!vk_->capabilities().features.descriptor_indexing)
            throw std::runtime_error("Descriptor Indexing (Bindless) not supported on this GPU!");
    }

    void create_resources()
    {
        // Allocate instance buffer using VMA
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        size_t size = sizeof(InstanceData) * kMaxInstances;
        
        uint32_t image_count = vk_->swapchain_image_count();
        instance_buffers_.resize(image_count, VK_NULL_HANDLE);
        instance_allocations_.resize(image_count, VK_NULL_HANDLE);
        ssbo_sets_.resize(image_count, VK_NULL_HANDLE);
        
        for (uint32_t i = 0; i < image_count; ++i)
        {
            if (!shs::vma_create_buffer(vk_->allocator(), size, usage, VMA_MEMORY_USAGE_CPU_TO_GPU, 
                instance_buffers_[i], instance_allocations_[i], VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT))
                throw std::runtime_error("Failed to create instance buffer via VMA");
        }

        // Initialize instances with random data
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
        std::uniform_int_distribution<uint32_t> tex_dist(0, 3); // 4 textures

        for (uint32_t i = 0; i < kMaxInstances; ++i)
        {
            instance_data_[i].model = glm::translate(glm::mat4(1.0f), glm::vec3(dist(rng), dist(rng), dist(rng)));
            instance_data_[i].texture_index = tex_dist(rng);
        }

        // Create 4 dummy textures with different colors
        uint32_t colors[4] = { 0xFF0000FF, 0xFF00FF00, 0xFFFF0000, 0xFFFFFFFF };
        (void)colors;
        for (int i = 0; i < 4; ++i)
        {
            VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ici.imageType = VK_IMAGE_TYPE_2D;
            ici.format = VK_FORMAT_R8G8B8A8_UNORM;
            ici.extent = {1, 1, 1};
            ici.mipLevels = 1;
            ici.arrayLayers = 1;
            ici.samples = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling = VK_IMAGE_TILING_OPTIMAL;
            ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

            if (!shs::vma_create_image(vk_->allocator(), ici, VMA_MEMORY_USAGE_GPU_ONLY, textures_[i], texture_allocs_[i]))
                throw std::runtime_error("Failed to create texture via VMA");

            VkImageViewCreateInfo iv{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            iv.image = textures_[i];
            iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
            iv.format = ici.format;
            iv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            iv.subresourceRange.levelCount = 1;
            iv.subresourceRange.layerCount = 1;
            vkCreateImageView(vk_->device(), &iv, nullptr, &texture_views_[i]);
        }

        // Create sampler
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = VK_FILTER_NEAREST;
        sci.minFilter = VK_FILTER_NEAREST;
        vkCreateSampler(vk_->device(), &sci, nullptr, &sampler_);

        // ---- Set 0: SSBO for instance data (vertex stage) ----
        {
            VkDescriptorSetLayoutBinding ssbo_binding{};
            ssbo_binding.binding = 0;
            ssbo_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            ssbo_binding.descriptorCount = 1;
            ssbo_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

            VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            ci.bindingCount = 1;
            ci.pBindings = &ssbo_binding;
            vkCreateDescriptorSetLayout(vk_->device(), &ci, nullptr, &ssbo_layout_);

            VkDescriptorPoolSize pool_size{};
            pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            pool_size.descriptorCount = image_count; // one per swapchain image slot

            VkDescriptorPoolCreateInfo pool_ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            pool_ci.maxSets = image_count;
            pool_ci.poolSizeCount = 1;
            pool_ci.pPoolSizes = &pool_size;
            vkCreateDescriptorPool(vk_->device(), &pool_ci, nullptr, &ssbo_pool_);

            std::vector<VkDescriptorSetLayout> layouts(image_count, ssbo_layout_);
            VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ai.descriptorPool = ssbo_pool_;
            ai.descriptorSetCount = image_count;
            ai.pSetLayouts = layouts.data();
            vkAllocateDescriptorSets(vk_->device(), &ai, ssbo_sets_.data());

            // Write each per-frame instance buffer into its descriptor set
            for (uint32_t i = 0; i < image_count; ++i)
            {
                VkDescriptorBufferInfo buf_info{};
                buf_info.buffer = instance_buffers_[i];
                buf_info.offset = 0;
                buf_info.range = sizeof(InstanceData) * kMaxInstances;

                VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                write.dstSet = ssbo_sets_[i];
                write.dstBinding = 0;
                write.descriptorCount = 1;
                write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                write.pBufferInfo = &buf_info;
                vkUpdateDescriptorSets(vk_->device(), 1, &write, 0, nullptr);
            }
        }

        // ---- Set 1: Bindless textures (fragment stage) ----
        shs::vk_create_bindless_descriptor_set_layout(vk_->device(), kMaxTextures, &bindless_layout_);
        shs::vk_create_bindless_descriptor_pool(vk_->device(), kMaxTextures, &bindless_pool_);

        VkDescriptorSetVariableDescriptorCountAllocateInfoEXT count_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT};
        uint32_t max_binding = kMaxTextures;
        count_info.descriptorSetCount = 1;
        count_info.pDescriptorCounts = &max_binding;

        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.pNext = &count_info;
        ai.descriptorPool = bindless_pool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &bindless_layout_;
        vkAllocateDescriptorSets(vk_->device(), &ai, &bindless_set_);

        // Update bindless set with our 4 textures
        for (uint32_t i = 0; i < 4; ++i)
        {
            shs::vk_update_bindless_texture(vk_->device(), bindless_set_, i, sampler_, texture_views_[i]);
        }
    }


    void create_pipeline()
    {
        const std::vector<char> vs_code = shs::vk_read_binary_file(SHS_VK_MODERN_VERT_SPV);
        const std::vector<char> fs_code = shs::vk_read_binary_file(SHS_VK_MODERN_FRAG_SPV);
        VkShaderModule vs = shs::vk_create_shader_module(vk_->device(), vs_code);
        VkShaderModule fs = shs::vk_create_shader_module(vk_->device(), fs_code);

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vs;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fs;
        stages[1].pName = "main";

        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = 1;
        vp.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_BACK_BIT;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS;

        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = 0xF;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 1;
        cb.pAttachments = &cba;

        VkDynamicState dyn_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dyn.dynamicStateCount = 2;
        dyn.pDynamicStates = dyn_states;

        // Two descriptor set layouts:
        //   set 0 -> SSBO for instance buffer (vertex stage)
        //   set 1 -> bindless texture array   (fragment stage)
        VkDescriptorSetLayout layouts[] = { ssbo_layout_, bindless_layout_ };
        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pcr.size = sizeof(glm::mat4);

        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl.setLayoutCount = 2;
        pl.pSetLayouts = layouts;
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &pcr;
        vkCreatePipelineLayout(vk_->device(), &pl, nullptr, &pipeline_layout_);

#if defined(VK_API_VERSION_1_3) || defined(VK_KHR_dynamic_rendering)
        VkPipelineRenderingCreateInfo pr{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        pr.colorAttachmentCount = 1;
        VkFormat color_fmt = vk_->swapchain_format();
        pr.pColorAttachmentFormats = &color_fmt;
        pr.depthAttachmentFormat = vk_->depth_format();

        VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gp.pNext = &pr;
        gp.stageCount = 2;
        gp.pStages = stages;
        gp.pVertexInputState = &vi;
        gp.pInputAssemblyState = &ia;
        gp.pViewportState = &vp;
        gp.pRasterizationState = &rs;
        gp.pMultisampleState = &ms;
        gp.pDepthStencilState = &ds;
        gp.pColorBlendState = &cb;
        gp.pDynamicState = &dyn;
        gp.layout = pipeline_layout_;
        vkCreateGraphicsPipelines(vk_->device(), VK_NULL_HANDLE, 1, &gp, nullptr, &pipeline_);
#endif

        vkDestroyShaderModule(vk_->device(), vs, nullptr);
        vkDestroyShaderModule(vk_->device(), fs, nullptr);
    }

    void main_loop()
    {
        bool running = true;
        while (running)
        {
            SDL_Event e;
            while (SDL_PollEvent(&e))
            {
                if (e.type == SDL_QUIT) running = false;
                if (e.type == SDL_WINDOWEVENT && e.window.event == SDL_WINDOWEVENT_RESIZED)
                    vk_->request_resize(e.window.data1, e.window.data2);
            }
            draw_frame();
        }
    }

    void draw_frame()
    {
        int w, h;
        SDL_Vulkan_GetDrawableSize(win_, &w, &h);
        if (w <= 0 || h <= 0) return;

        shs::RenderBackendFrameInfo frame{};
        frame.frame_index = ctx_.frame_index;
        frame.width = w;
        frame.height = h;

        shs::VulkanRenderBackend::FrameInfo fi;
        if (!vk_->begin_frame(ctx_, frame, fi)) return;

        // VMA mapping for per-frame instance data
        void* data;
        vmaMapMemory(vk_->allocator(), instance_allocations_[fi.image_index], &data);
        std::memcpy(data, instance_data_.data(), sizeof(InstanceData) * kMaxInstances);
        vmaUnmapMemory(vk_->allocator(), instance_allocations_[fi.image_index]);

        VkCommandBuffer cmd = fi.cmd;
        
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bi);

        VkClearColorValue clear = {{0.05f, 0.05f, 0.07f, 1.0f}};
        vk_->begin_rendering(cmd, fi.view, fi.depth_view, fi.extent, clear, 1.0f, true);
        
        shs::vk_cmd_set_viewport_scissor(cmd, fi.extent.width, fi.extent.height, true);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        
        // Update Push Constants
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)fi.extent.width / (float)fi.extent.height, 0.1f, 1000.0f);
        glm::mat4 view = glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, -25.0f));
        glm::mat4 viewProj = projection * view;
        vkCmdPushConstants(cmd, pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &viewProj);

        // Bind set 0: SSBO for per-frame instance data
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0, 1, &ssbo_sets_[fi.image_index], 0, nullptr);
        // Bind set 1: bindless texture array
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 1, 1, &bindless_set_, 0, nullptr);
        
        // Draw call: instances
        vkCmdDraw(cmd, 3, kMaxInstances, 0, 0);
        
        vk_->end_rendering(cmd);
        vkEndCommandBuffer(cmd);

        vk_->end_frame(fi);
        ctx_.frame_index++;
    }

    void cleanup()
    {
        if (vk_ && vk_->device() != VK_NULL_HANDLE)
        {
            vkDeviceWaitIdle(vk_->device());
            
            // Allocator check
            if (vk_->allocator() != VK_NULL_HANDLE)
            {
                for (size_t i = 0; i < instance_buffers_.size(); ++i)
                    shs::vma_destroy_buffer(vk_->allocator(), instance_buffers_[i], instance_allocations_[i]);
                for (int i = 0; i < 4; ++i)
                {
                    shs::vma_destroy_image(vk_->allocator(), textures_[i], texture_allocs_[i]);
                }
            }

            for (int i = 0; i < 4; ++i)
            {
                if (texture_views_[i] != VK_NULL_HANDLE)
                    vkDestroyImageView(vk_->device(), texture_views_[i], nullptr);
            }
            if (sampler_ != VK_NULL_HANDLE) vkDestroySampler(vk_->device(), sampler_, nullptr);
            if (ssbo_pool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(vk_->device(), ssbo_pool_, nullptr);
            if (ssbo_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(vk_->device(), ssbo_layout_, nullptr);
            if (bindless_pool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(vk_->device(), bindless_pool_, nullptr);
            if (bindless_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(vk_->device(), bindless_layout_, nullptr);
            if (pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(vk_->device(), pipeline_, nullptr);
            if (pipeline_layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(vk_->device(), pipeline_layout_, nullptr);
        }
        keep_.clear();
        if (win_) SDL_DestroyWindow(win_);
        if (sdl_ready_) SDL_Quit();
    }

    bool sdl_ready_ = false;
    SDL_Window* win_ = nullptr;
    shs::Context ctx_;
    std::vector<std::unique_ptr<shs::IRenderBackend>> keep_;
    shs::VulkanRenderBackend* vk_ = nullptr;
    
    std::vector<VkBuffer> instance_buffers_;
    std::vector<VmaAllocation> instance_allocations_;
    std::vector<InstanceData> instance_data_{kMaxInstances};

    VkImage textures_[4]{VK_NULL_HANDLE};
    VmaAllocation texture_allocs_[4]{VK_NULL_HANDLE};
    VkImageView texture_views_[4]{VK_NULL_HANDLE};
    VkSampler sampler_ = VK_NULL_HANDLE;

    VkDescriptorSetLayout ssbo_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool ssbo_pool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> ssbo_sets_;

    VkDescriptorSetLayout bindless_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool bindless_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet bindless_set_ = VK_NULL_HANDLE;

    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};
}

int main()
{
    try { HelloModernVulkanApp().run(); return 0; }
    catch (const std::exception& e) { std::fprintf(stderr, "Fatal: %s\n", e.what()); return 1; }
}
