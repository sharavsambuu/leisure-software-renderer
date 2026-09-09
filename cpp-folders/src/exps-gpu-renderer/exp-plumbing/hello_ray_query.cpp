#define SDL_MAIN_HANDLED

#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <array>

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <vulkan/vulkan.h>

#include <shs/core/context.hpp>
#include <shs/input/value_actions.hpp>
#include <shs/input/value_input_latch.hpp>
#include <shs/rhi/backend/backend_factory.hpp>
#include <shs/rhi/drivers/vulkan/vk_backend.hpp>
#include <shs/rhi/drivers/vulkan/vk_cmd_utils.hpp>
#include <shs/rhi/drivers/vulkan/vk_shader_utils.hpp>
#include <shs/rhi/drivers/vulkan/vk_memory_utils.hpp>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <shs/camera/convention.hpp>
namespace
{
constexpr int kDefaultW = 960;
constexpr int kDefaultH = 640;

struct Vertex {
    glm::vec3 pos;
};

struct PushConstants {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec3 lightPos;
};

class HelloRayQueryApp
{
public:
    ~HelloRayQueryApp()
    {
        cleanup();
    }

    void run()
    {
        init_sdl();
        init_backend();
        create_geometry();
        create_acceleration_structures();
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
        win_ = SDL_CreateWindow(
            "HelloRayQuery",
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
        if (!created.backend) throw std::runtime_error("Backend factory did not return a backend");
        keep_.push_back(std::move(created.backend));
        vk_ = dynamic_cast<shs::VulkanRenderBackend*>(keep_.back().get());
        
        shs::VulkanRenderBackend::InitDesc desc{};
        desc.window = win_;
        desc.width = kDefaultW;
        desc.height = kDefaultH;
        desc.enable_validation = true;
        desc.request_ray_bundle = true;
        
        if (!vk_->init(desc)) throw std::runtime_error("Vulkan backend init failed");

        if (!vk_->capabilities().features.ray_query)
        {
            throw std::runtime_error("Hardware Ray Tracing (Ray Query) is not supported on this device.");
        }
    }

    void create_geometry()
    {
        // Simple plane and a floating triangle
        std::vector<Vertex> vertices = {
            // Plane
            {{-5.0f, 0.0f, -5.0f}}, {{5.0f, 0.0f, -5.0f}}, {{5.0f, 0.0f, 5.0f}}, {{-5.0f, 0.0f, 5.0f}},
            // Triangle
            {{-1.0f, 2.0f, 0.0f}}, {{1.0f, 2.0f, 0.0f}}, {{0.0f, 4.0f, 0.0f}}
        };
        std::vector<uint32_t> indices = {
            0, 1, 2, 0, 2, 3, // plane
            4, 5, 6           // triangle
        };

        const VkDeviceSize vSize = vertices.size() * sizeof(Vertex);
        const VkDeviceSize iSize = indices.size() * sizeof(uint32_t);

        shs::vk_create_buffer(vk_->device(), vk_->physical_device(), vSize,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            vBuf_, vMem_);

        shs::vk_create_buffer(vk_->device(), vk_->physical_device(), iSize,
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            iBuf_, iMem_);

        void* data;
        vkMapMemory(vk_->device(), vMem_, 0, vSize, 0, &data);
        memcpy(data, vertices.data(), vSize);
        vkUnmapMemory(vk_->device(), vMem_);

        vkMapMemory(vk_->device(), iMem_, 0, iSize, 0, &data);
        memcpy(data, indices.data(), iSize);
        vkUnmapMemory(vk_->device(), iMem_);

        vAddr_ = vk_->get_buffer_device_address(vBuf_);
        iAddr_ = vk_->get_buffer_device_address(iBuf_);

        printf("[demo] Geometry addresses: Vertex=0x%llx, Index=0x%llx\n", (unsigned long long)vAddr_, (unsigned long long)iAddr_);
    }

    void create_acceleration_structures()
    {
        const VkDevice dev = vk_->device();

        // BLAS
        VkAccelerationStructureGeometryKHR geom{};
        geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        geom.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        geom.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        geom.geometry.triangles.vertexData.deviceAddress = vAddr_;
        geom.geometry.triangles.vertexStride = sizeof(Vertex);
        geom.geometry.triangles.maxVertex = 7;
        geom.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
        geom.geometry.triangles.indexData.deviceAddress = iAddr_;

        VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
        buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        buildInfo.geometryCount = 1;
        buildInfo.pGeometries = &geom;

        const uint32_t triangleCount = 3; // 2 for plane, 1 for tri
        VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
        sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        
        // Use the function pointer loaded in the backend
        auto get_sizes = (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(dev, "vkGetAccelerationStructureBuildSizesKHR");
        if (!get_sizes) throw std::runtime_error("vkGetAccelerationStructureBuildSizesKHR not found");
        get_sizes(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &triangleCount, &sizeInfo);

        if (!vk_->create_acceleration_structure(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, sizeInfo.accelerationStructureSize, blas_))
        {
            throw std::runtime_error("Failed to create BLAS");
        }
        printf("[demo] BLAS created, address=0x%llx\n", (unsigned long long)blas_.device_address);

        // Scratch buffer
        VkBuffer scratch; VkDeviceMemory sMem;
        shs::vk_create_buffer(dev, vk_->physical_device(), sizeInfo.buildScratchSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, scratch, sMem);

        // Build cmd
        // For simplicity in the demo, we use a one-time command buffer
        VkCommandPool pool;
        VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpci.queueFamilyIndex = vk_->graphics_queue_family_index();
        vkCreateCommandPool(dev, &cpci, nullptr, &pool);

        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = pool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cmd;
        vkAllocateCommandBuffers(dev, &cbai, &cmd);

        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &cbi);

        buildInfo.dstAccelerationStructure = blas_.handle;
        buildInfo.scratchData.deviceAddress = vk_->get_buffer_device_address(scratch);

        VkAccelerationStructureBuildRangeInfoKHR range{};
        range.primitiveCount = triangleCount;
        const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;

        auto build_as = (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(dev, "vkCmdBuildAccelerationStructuresKHR");
        build_as(cmd, 1, &buildInfo, &pRange);

        vkEndCommandBuffer(cmd);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        vkQueueSubmit(vk_->graphics_queue(), 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(vk_->graphics_queue());

        vkFreeCommandBuffers(dev, pool, 1, &cmd);
        vkDestroyCommandPool(dev, pool, nullptr);
        shs::vk_destroy_buffer(dev, scratch, sMem);

        // TLAS
        VkAccelerationStructureInstanceKHR instance{};
        instance.transform = { 1,0,0,0, 0,1,0,0, 0,0,1,0 };
        instance.instanceCustomIndex = 0;
        instance.mask = 0xFF;
        instance.instanceShaderBindingTableRecordOffset = 0;
        instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        instance.accelerationStructureReference = blas_.device_address;

        VkBuffer instBuf; VkDeviceMemory instMem;
        shs::vk_create_buffer(dev, vk_->physical_device(), sizeof(instance),
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            instBuf, instMem);
        void* instData;
        vkMapMemory(dev, instMem, 0, sizeof(instance), 0, &instData);
        memcpy(instData, &instance, sizeof(instance));
        vkUnmapMemory(dev, instMem);

        VkAccelerationStructureGeometryKHR tgeom{};
        tgeom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        tgeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        tgeom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        tgeom.geometry.instances.data.deviceAddress = vk_->get_buffer_device_address(instBuf);

        VkAccelerationStructureBuildGeometryInfoKHR tbuildInfo{};
        tbuildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        tbuildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        tbuildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        tbuildInfo.geometryCount = 1;
        tbuildInfo.pGeometries = &tgeom;

        const uint32_t instCount = 1;
        get_sizes(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tbuildInfo, &instCount, &sizeInfo);

        if (!vk_->create_acceleration_structure(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, sizeInfo.accelerationStructureSize, tlas_))
        {
            throw std::runtime_error("Failed to create TLAS");
        }
        printf("[demo] TLAS created, address=0x%llx\n", (unsigned long long)tlas_.device_address);

        shs::vk_create_buffer(dev, vk_->physical_device(), sizeInfo.buildScratchSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, scratch, sMem);

        vkCreateCommandPool(dev, &cpci, nullptr, &pool);
        vkAllocateCommandBuffers(dev, &cbai, &cmd);
        vkBeginCommandBuffer(cmd, &cbi);

        tbuildInfo.dstAccelerationStructure = tlas_.handle;
        tbuildInfo.scratchData.deviceAddress = vk_->get_buffer_device_address(scratch);
        
        range.primitiveCount = instCount;
        build_as(cmd, 1, &tbuildInfo, &pRange);

        vkEndCommandBuffer(cmd);
        vkQueueSubmit(vk_->graphics_queue(), 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(vk_->graphics_queue());

        vkFreeCommandBuffers(dev, pool, 1, &cmd);
        vkDestroyCommandPool(dev, pool, nullptr);
        shs::vk_destroy_buffer(dev, scratch, sMem);
        shs::vk_destroy_buffer(dev, instBuf, instMem);
    }

    void create_pipeline()
    {
        const VkDevice dev = vk_->device();
        
        auto vCode = shs::vk_read_binary_file(SHS_VK_RAY_QUERY_VERT_SPV);
        auto fCode = shs::vk_read_binary_file(SHS_VK_RAY_QUERY_FRAG_SPV);
        VkShaderModule vMod = shs::vk_create_shader_module(dev, vCode);
        VkShaderModule fMod = shs::vk_create_shader_module(dev, fCode);

        // Descriptors
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo dlci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dlci.bindingCount = 1;
        dlci.pBindings = &binding;
        vkCreateDescriptorSetLayout(dev, &dlci, nullptr, &dsLayout_);

        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1};
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets = 1;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &ps;
        vkCreateDescriptorPool(dev, &dpci, nullptr, &dsPool_);

        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = dsPool_;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &dsLayout_;
        vkAllocateDescriptorSets(dev, &dsai, &dsSet_);

        VkWriteDescriptorSetAccelerationStructureKHR wdas{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
        wdas.accelerationStructureCount = 1;
        wdas.pAccelerationStructures = &tlas_.handle;

        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = dsSet_;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        write.pNext = &wdas;
        vkUpdateDescriptorSets(dev, 1, &write, 0, nullptr);

        // Pipeline
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &dsLayout_;
        VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants)};
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        vkCreatePipelineLayout(dev, &plci, nullptr, &pLayout_);

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vMod;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fMod;
        stages[1].pName = "main";

        VkVertexInputBindingDescription bindDesc{};
        bindDesc.binding = 0;
        bindDesc.stride = sizeof(Vertex);
        bindDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription attrDesc{};
        attrDesc.binding = 0;
        attrDesc.location = 0;
        attrDesc.format = VK_FORMAT_R32G32B32_SFLOAT;
        attrDesc.offset = 0;

        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &bindDesc;
        vi.vertexAttributeDescriptionCount = 1;
        vi.pVertexAttributeDescriptions = &attrDesc;

        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = 1;
        vp.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rs.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS;

        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = 0xF;
        cba.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 1;
        cb.pAttachments = &cba;

        VkDynamicState dynStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dyn.dynamicStateCount = 2;
        dyn.pDynamicStates = dynStates;

        VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
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
        gp.layout = pLayout_;
        gp.renderPass = vk_->render_pass();

        if (vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, nullptr, &pipeline_) != VK_SUCCESS) {
             throw std::runtime_error("Failed to create ray query pipeline");
        }

        vkDestroyShaderModule(dev, vMod, nullptr);
        vkDestroyShaderModule(dev, fMod, nullptr);
    }

    void main_loop()
    {
        bool running = true;
        shs::RuntimeInputLatch input_latch{};
        std::vector<shs::RuntimeInputEvent> pending_input_events{};
        shs::RuntimeState runtime_state{};
        std::vector<shs::RuntimeAction> runtime_actions{};
        while (running)
        {
            SDL_Event e;
            while (SDL_PollEvent(&e))
            {
                if (e.type == SDL_QUIT)
                {
                    pending_input_events.push_back(shs::make_quit_input_event());
                }
                if (e.type == SDL_KEYDOWN)
                {
                    if (e.key.keysym.sym == SDLK_ESCAPE)
                    {
                        pending_input_events.push_back(shs::make_quit_input_event());
                    }
                }
            }
            input_latch = shs::reduce_runtime_input_latch(input_latch, pending_input_events);
            pending_input_events.clear();

            runtime_actions.clear();
            shs::InputState runtime_input{};
            runtime_input.quit = input_latch.quit_requested;
            shs::emit_human_actions(runtime_input, runtime_actions, 0.0f, 1.0f, 0.0f);
            runtime_state = shs::reduce_runtime_state(runtime_state, runtime_actions, 0.0f);
            if (runtime_state.quit_requested) break;

            draw_frame();
        }
    }

    void draw_frame()
    {
        shs::RenderBackendFrameInfo frame{};
        shs::VulkanRenderBackend::FrameInfo fi{};
        if (!vk_->begin_frame(ctx_, frame, fi)) return;

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(fi.cmd, &bi);

        VkClearValue cv[2]{};
        cv[0].color = {{0,0,0,1}}; cv[1].depthStencil = {1,0};
        VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rp.renderPass = fi.render_pass;
        rp.framebuffer = fi.framebuffer;
        rp.renderArea.extent = fi.extent;
        rp.clearValueCount = 2;
        rp.pClearValues = cv;
        
        vkCmdBeginRenderPass(fi.cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(fi.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        vkCmdBindDescriptorSets(fi.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pLayout_, 0, 1, &dsSet_, 0, nullptr);

        shs::vk_cmd_set_viewport_scissor(fi.cmd, fi.extent.width, fi.extent.height, true);

        PushConstants pcs{};
        pcs.model = glm::mat4(1.0f);
        pcs.view = shs::look_at_lh(glm::vec3(0, 5, 10), glm::vec3(0, 2, 0), glm::vec3(0, 1, 0));
        pcs.proj = shs::perspective_lh_no(
            glm::radians(45.0f),
            (float)fi.extent.width / fi.extent.height,
            0.1f,
            100.0f);
        pcs.lightPos = glm::vec3(std::sin(SDL_GetTicks() * 0.001f) * 5.0f, 6.0f, 2.0f);

        vkCmdPushConstants(fi.cmd, pLayout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pcs), &pcs);

        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(fi.cmd, 0, 1, &vBuf_, &offset);
        vkCmdBindIndexBuffer(fi.cmd, iBuf_, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(fi.cmd, 9, 1, 0, 0, 0);

        vkCmdEndRenderPass(fi.cmd);
        vkEndCommandBuffer(fi.cmd);

        vk_->end_frame(fi);
    }

    void cleanup()
    {
        if (vk_) {
            vkDeviceWaitIdle(vk_->device());
            vk_->destroy_acceleration_structure(blas_);
            vk_->destroy_acceleration_structure(tlas_);
            shs::vk_destroy_buffer(vk_->device(), vBuf_, vMem_);
            shs::vk_destroy_buffer(vk_->device(), iBuf_, iMem_);
            vkDestroyPipeline(vk_->device(), pipeline_, nullptr);
            vkDestroyPipelineLayout(vk_->device(), pLayout_, nullptr);
            vkDestroyDescriptorPool(vk_->device(), dsPool_, nullptr);
            vkDestroyDescriptorSetLayout(vk_->device(), dsLayout_, nullptr);
        }
        if (win_) SDL_DestroyWindow(win_);
        SDL_Quit();
    }

private:
    SDL_Window* win_ = nullptr;
    shs::Context ctx_{};
    std::vector<std::unique_ptr<shs::IRenderBackend>> keep_{};
    shs::VulkanRenderBackend* vk_ = nullptr;

    VkBuffer vBuf_ = VK_NULL_HANDLE, iBuf_ = VK_NULL_HANDLE;
    VkDeviceMemory vMem_ = VK_NULL_HANDLE, iMem_ = VK_NULL_HANDLE;
    VkDeviceAddress vAddr_ = 0, iAddr_ = 0;

    shs::VulkanRenderBackend::VulkanAccelerationStructure blas_, tlas_;
    VkDescriptorSetLayout dsLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool dsPool_ = VK_NULL_HANDLE;
    VkDescriptorSet dsSet_ = VK_NULL_HANDLE;
    VkPipelineLayout pLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};
}

int main() {
    try {
        HelloRayQueryApp app;
        app.run();
    } catch (const std::exception& e) {
        printf("Error: %s\n", e.what());
        return 1;
    }
    return 0;
}
