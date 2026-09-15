/*
    Offscreen Vulkan harness — implementation. See adventures_vk.hpp.
    Deliberately minimal: no validation layers, no swapchain, single graphics
    queue, single submit per render. Works on real GPUs and on lavapipe
    (headless CI).
*/

#include "adventures_vk.hpp"

#include <cstdio>
#include <cstring>

#include <stb_image_write.h>

#include "shs/rhi/drivers/vulkan/vk_shader_utils.hpp"

namespace adventures
{
    namespace
    {
        constexpr VkFormat kColorFormat = VK_FORMAT_R8G8B8A8_UNORM;
        constexpr VkFormat kDepthFormat = VK_FORMAT_D24_UNORM_S8_UINT;

        VkStencilOpState stencil_state(bool test, bool write, bool invert, uint8_t reference)
        {
            VkStencilOpState st{};
            st.compareMask = 0xff;
            st.writeMask   = write ? 0xffu : 0x00u;
            st.reference   = reference;
            st.compareOp = !test ? VK_COMPARE_OP_ALWAYS
                                 : (invert ? VK_COMPARE_OP_NOT_EQUAL : VK_COMPARE_OP_EQUAL);
            st.failOp      = VK_STENCIL_OP_KEEP;
            st.depthFailOp = VK_STENCIL_OP_KEEP;
            st.passOp      = write ? VK_STENCIL_OP_REPLACE : VK_STENCIL_OP_KEEP;
            return st;
        }
    }

    OffscreenVulkan::~OffscreenVulkan()
    {
        if (device_ == VK_NULL_HANDLE) return;
        vkDeviceWaitIdle(device_);
        for (const auto& ts : texture_sets_)
        {
            if (ts.sampler) vkDestroySampler(device_, ts.sampler, nullptr);
            if (ts.view) vkDestroyImageView(device_, ts.view, nullptr);
            if (ts.image) vkDestroyImage(device_, ts.image, nullptr);
            if (ts.memory) vkFreeMemory(device_, ts.memory, nullptr);
        }
        for (VkPipeline p : pipelines_) vkDestroyPipeline(device_, p, nullptr);
        if (framebuffer_) vkDestroyFramebuffer(device_, framebuffer_, nullptr);
        if (render_pass_) vkDestroyRenderPass(device_, render_pass_, nullptr);
        if (color_view_) vkDestroyImageView(device_, color_view_, nullptr);
        if (depth_view_) vkDestroyImageView(device_, depth_view_, nullptr);
        if (color_image_) vkDestroyImage(device_, color_image_, nullptr);
        if (depth_image_) vkDestroyImage(device_, depth_image_, nullptr);
        if (color_memory_) vkFreeMemory(device_, color_memory_, nullptr);
        if (depth_memory_) vkFreeMemory(device_, depth_memory_, nullptr);
        if (vertex_buffer_) vkDestroyBuffer(device_, vertex_buffer_, nullptr);
        if (vertex_memory_) vkFreeMemory(device_, vertex_memory_, nullptr);
        if (descriptor_pool_) vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
        if (texture_set_layout_) vkDestroyDescriptorSetLayout(device_, texture_set_layout_, nullptr);
        if (pipeline_layout_) vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
        if (cmd_pool_) vkDestroyCommandPool(device_, cmd_pool_, nullptr);
        if (device_) vkDestroyDevice(device_, nullptr);
        if (instance_) vkDestroyInstance(instance_, nullptr);
    }

    uint32_t OffscreenVulkan::find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags props) const
    {
        VkPhysicalDeviceMemoryProperties mem{};
        vkGetPhysicalDeviceMemoryProperties(physical_, &mem);
        for (uint32_t i = 0; i < mem.memoryTypeCount; ++i)
        {
            if ((type_bits & (1u << i)) && (mem.memoryTypes[i].propertyFlags & props) == props)
            {
                return i;
            }
        }
        return UINT32_MAX;
    }

    bool OffscreenVulkan::create_image(uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage,
                                       VkImage* image, VkDeviceMemory* memory) const
    {
        VkImageCreateInfo ii{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        ii.imageType   = VK_IMAGE_TYPE_2D;
        ii.format      = format;
        ii.extent      = { w, h, 1 };
        ii.mipLevels   = 1;
        ii.arrayLayers = 1;
        ii.samples     = VK_SAMPLE_COUNT_1_BIT;
        ii.tiling      = VK_IMAGE_TILING_OPTIMAL;
        ii.usage       = usage;
        if (vkCreateImage(device_, &ii, nullptr, image) != VK_SUCCESS) return false;

        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(device_, *image, &req);
        VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        ai.allocationSize  = req.size;
        ai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (ai.memoryTypeIndex == UINT32_MAX) return false;
        if (vkAllocateMemory(device_, &ai, nullptr, memory) != VK_SUCCESS) return false;
        return vkBindImageMemory(device_, *image, *memory, 0) == VK_SUCCESS;
    }

    bool OffscreenVulkan::init(uint32_t width, uint32_t height)
    {
        width_  = width;
        height_ = height;

        VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
        app.pApplicationName = "shs-adventures-tier0";
        app.apiVersion       = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ici{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
        ici.pApplicationInfo = &app;
        // optional validation (VVL from the SDK) for debugging driver oddities
        if (const char* want_validation = getenv("SHS_ADVENTURES_VALIDATION"))
        {
            if (want_validation[0] == '1')
            {
                const char* layer = "VK_LAYER_KHRONOS_validation";
                ici.enabledLayerCount = 1;
                ici.ppEnabledLayerNames = &layer;
            }
        }
        if (vkCreateInstance(&ici, nullptr, &instance_) != VK_SUCCESS)
        {
            std::fprintf(stderr, "adventures-vk: vkCreateInstance failed\n");
            return false;
        }

        uint32_t gpu_count = 0;
        vkEnumeratePhysicalDevices(instance_, &gpu_count, nullptr);
        if (gpu_count == 0)
        {
            std::fprintf(stderr, "adventures-vk: no physical devices found\n");
            return false;
        }
        std::vector<VkPhysicalDevice> gpus(gpu_count);
        vkEnumeratePhysicalDevices(instance_, &gpu_count, gpus.data());
        physical_ = gpus[0]; // first device (llvmpipe in headless CI)

        uint32_t q_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_, &q_count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(q_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_, &q_count, queues.data());
        bool found_queue = false;
        for (uint32_t i = 0; i < q_count; ++i)
        {
            if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            {
                queue_family_ = i;
                found_queue   = true;
                break;
            }
        }
        if (!found_queue)
        {
            std::fprintf(stderr, "adventures-vk: no graphics queue family\n");
            return false;
        }

        const float priority = 1.0f;
        VkDeviceQueueCreateInfo qci{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        qci.queueFamilyIndex = queue_family_;
        qci.queueCount       = 1;
        qci.pQueuePriorities = &priority;
        VkPhysicalDeviceFeatures features{};
        VkDeviceCreateInfo dci{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos    = &qci;
        dci.pEnabledFeatures     = &features;
        if (vkCreateDevice(physical_, &dci, nullptr, &device_) != VK_SUCCESS)
        {
            std::fprintf(stderr, "adventures-vk: vkCreateDevice failed\n");
            return false;
        }
        vkGetDeviceQueue(device_, queue_family_, 0, &queue_);

        VkCommandPoolCreateInfo pci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        pci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = queue_family_;
        if (vkCreateCommandPool(device_, &pci, nullptr, &cmd_pool_) != VK_SUCCESS) return false;
        VkCommandBufferAllocateInfo cai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        cai.commandPool        = cmd_pool_;
        cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(device_, &cai, &cmd_) != VK_SUCCESS)
        {
            std::fprintf(stderr, "adventures-vk: vkAllocateCommandBuffers failed\n");
            return false;
        }
        if (!init_render_targets())
        {
            std::fprintf(stderr, "adventures-vk: init_render_targets failed\n");
            return false;
        }
        return true;
    }

    // [PART3]
    bool OffscreenVulkan::init_render_targets()
    {
        // descriptor set layout for textured pipelines (set 0, binding 0)
        VkDescriptorSetLayoutBinding tex_binding{};
        tex_binding.binding         = 0;
        tex_binding.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        tex_binding.descriptorCount = 1;
        tex_binding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo dli{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        dli.bindingCount = 1;
        dli.pBindings    = &tex_binding;
        if (vkCreateDescriptorSetLayout(device_, &dli, nullptr, &texture_set_layout_) != VK_SUCCESS)
        {
            std::fprintf(stderr, "adventures-vk: texture set layout creation failed\n");
            return false;
        }
        // render pass: attachments[0] = color, attachments[1] = depth-stencil
        VkAttachmentDescription color_att{};
        color_att.format        = kColorFormat;
        color_att.samples       = VK_SAMPLE_COUNT_1_BIT;
        color_att.loadOp        = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_att.storeOp       = VK_ATTACHMENT_STORE_OP_STORE;
        color_att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color_att.finalLayout   = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; // ready for readback

        VkAttachmentDescription depth_att{};
        depth_att.format         = kDepthFormat;
        depth_att.samples        = VK_SAMPLE_COUNT_1_BIT;
        depth_att.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth_att.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth_att.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth_att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth_att.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        depth_att.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        const VkAttachmentDescription attachments[2] = { color_att, depth_att };
        const VkAttachmentReference color_ref{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
        const VkAttachmentReference depth_ref{ 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount    = 1;
        subpass.pColorAttachments       = &color_ref;
        subpass.pDepthStencilAttachment = &depth_ref;
        VkRenderPassCreateInfo rpi{ VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
        rpi.attachmentCount = 2;
        rpi.pAttachments    = attachments;
        rpi.subpassCount    = 1;
        rpi.pSubpasses      = &subpass;
        if (vkCreateRenderPass(device_, &rpi, nullptr, &render_pass_) != VK_SUCCESS)
        {
            std::fprintf(stderr, "adventures-vk: vkCreateRenderPass failed\n");
            return false;
        }

        if (!create_image(width_, height_, kColorFormat,
                          VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                          &color_image_, &color_memory_))
        {
            std::fprintf(stderr, "adventures-vk: color image creation failed\n");
            return false;
        }
        if (!create_image(width_, height_, kDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                          &depth_image_, &depth_memory_))
        {
            std::fprintf(stderr, "adventures-vk: depth image creation failed\n");
            return false;
        }

        VkImageViewCreateInfo cvi{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        cvi.image            = color_image_;
        cvi.viewType         = VK_IMAGE_VIEW_TYPE_2D;
        cvi.format           = kColorFormat;
        cvi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        if (vkCreateImageView(device_, &cvi, nullptr, &color_view_) != VK_SUCCESS) return false;

        VkImageViewCreateInfo dvi{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        dvi.image            = depth_image_;
        dvi.viewType         = VK_IMAGE_VIEW_TYPE_2D;
        dvi.format           = kDepthFormat;
        dvi.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1 };
        if (vkCreateImageView(device_, &dvi, nullptr, &depth_view_) != VK_SUCCESS) return false;

        const VkImageView fb_views[2] = { color_view_, depth_view_ };
        VkFramebufferCreateInfo fbi{ VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
        fbi.renderPass      = render_pass_;
        fbi.attachmentCount = 2;
        fbi.pAttachments    = fb_views;
        fbi.width           = width_;
        fbi.height          = height_;
        fbi.layers          = 1;
        if (vkCreateFramebuffer(device_, &fbi, nullptr, &framebuffer_) != VK_SUCCESS) return false;

        // shared pipeline layout: push constants (vertex+fragment) + optional texture set
        const VkPushConstantRange push_range{ VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                              0, kMaxPushConstants };
        VkPipelineLayoutCreateInfo pli{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges    = &push_range;
        pli.setLayoutCount         = 1;
        pli.pSetLayouts            = &texture_set_layout_;
        return vkCreatePipelineLayout(device_, &pli, nullptr, &pipeline_layout_) == VK_SUCCESS;
    }

    int OffscreenVulkan::add_pipeline(const VkPipelineSetup& setup)
    {
        std::vector<char> vs_bytes, fs_bytes;
        if (!shs::vk_try_read_binary_file(setup.vs_spv_path, vs_bytes) ||
            !shs::vk_try_read_binary_file(setup.fs_spv_path, fs_bytes))
        {
            std::fprintf(stderr, "adventures-vk: failed to read SPIR-V (%s / %s)\n",
                         setup.vs_spv_path, setup.fs_spv_path);
            return -1;
        }

        VkShaderModuleCreateInfo mi{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        mi.codeSize = vs_bytes.size();
        mi.pCode = reinterpret_cast<const uint32_t*>(vs_bytes.data());
        VkShaderModule vs = VK_NULL_HANDLE;
        if (vkCreateShaderModule(device_, &mi, nullptr, &vs) != VK_SUCCESS)
        {
            std::fprintf(stderr, "adventures-vk: vkCreateShaderModule(vs) failed\n");
            return -1;
        }
        mi.codeSize = fs_bytes.size();
        mi.pCode = reinterpret_cast<const uint32_t*>(fs_bytes.data());
        VkShaderModule fs = VK_NULL_HANDLE;
        if (vkCreateShaderModule(device_, &mi, nullptr, &fs) != VK_SUCCESS)
        {
            std::fprintf(stderr, "adventures-vk: vkCreateShaderModule(fs) failed\n");
            vkDestroyShaderModule(device_, vs, nullptr);
            return -1;
        }

        const VkPipelineShaderStageCreateInfo stages[2] = {
            { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
              VK_SHADER_STAGE_VERTEX_BIT, vs, "vs_main", nullptr },
            { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
              VK_SHADER_STAGE_FRAGMENT_BIT, fs, "fs_main", nullptr },
        };

        // vertex layout mirrors T0Vertex: pos(3f) col(4f) uv(2f) @ locations 0..2
        const VkVertexInputBindingDescription binding{ 0, sizeof(float) * 9, VK_VERTEX_INPUT_RATE_VERTEX };
        const VkVertexInputAttributeDescription attrs[3] = {
            { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0 },
            { 1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, sizeof(float) * 3 },
            { 2, 0, VK_FORMAT_R32G32_SFLOAT, sizeof(float) * 7 },
        };
        VkPipelineVertexInputStateCreateInfo vi{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
        vi.vertexBindingDescriptionCount   = 1;
        vi.pVertexBindingDescriptions      = &binding;
        vi.vertexAttributeDescriptionCount = 3;
        vi.pVertexAttributeDescriptions    = attrs;

        VkPipelineInputAssemblyStateCreateInfo ia{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vp{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
        vp.viewportCount = 1;
        vp.scissorCount  = 1;

        VkPipelineRasterizationStateCreateInfo rs{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode    = VK_CULL_MODE_NONE; // curriculum demos draw both orientations
        rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth   = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
        ds.depthTestEnable   = setup.depth_test ? VK_TRUE : VK_FALSE;
        ds.depthWriteEnable  = setup.depth_write ? VK_TRUE : VK_FALSE;
        ds.depthCompareOp    = VK_COMPARE_OP_LESS;
        ds.stencilTestEnable = (setup.stencil_test || setup.stencil_write) ? VK_TRUE : VK_FALSE;
        ds.front             = stencil_state(setup.stencil_test, setup.stencil_write, setup.stencil_invert, setup.stencil_ref);
        ds.back              = ds.front;

        VkPipelineColorBlendAttachmentState cb_att{};
        cb_att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        cb_att.blendEnable         = setup.blend ? VK_TRUE : VK_FALSE;
        cb_att.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        cb_att.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        cb_att.colorBlendOp        = VK_BLEND_OP_ADD;
        cb_att.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        cb_att.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        cb_att.alphaBlendOp        = VK_BLEND_OP_ADD;
        VkPipelineColorBlendStateCreateInfo cb{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
        cb.attachmentCount = 1;
        cb.pAttachments    = &cb_att;

        const VkDynamicState dynamics[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dyn{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
        dyn.dynamicStateCount = 2;
        dyn.pDynamicStates    = dynamics;

        VkGraphicsPipelineCreateInfo pi{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
        pi.stageCount          = 2;
        pi.pStages             = stages;
        pi.pVertexInputState   = &vi;
        pi.pInputAssemblyState = &ia;
        pi.pViewportState      = &vp;
        pi.pRasterizationState = &rs;
        pi.pMultisampleState   = &ms;
        pi.pDepthStencilState  = &ds;
        pi.pColorBlendState    = &cb;
        pi.pDynamicState       = &dyn;
        pi.layout              = pipeline_layout_;
        pi.renderPass          = render_pass_;
        pi.subpass             = 0;

        VkPipeline     pipeline = VK_NULL_HANDLE;
        const VkResult res      = vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &pipeline);
        vkDestroyShaderModule(device_, vs, nullptr);
        vkDestroyShaderModule(device_, fs, nullptr);
        if (res != VK_SUCCESS)
        {
            std::fprintf(stderr, "adventures-vk: vkCreateGraphicsPipelines failed (%d)\n", int(res));
            return -1;
        }
        pipelines_.push_back(pipeline);
        return int(pipelines_.size()) - 1;
    }

    bool OffscreenVulkan::upload_vertices(const void* data, size_t byte_size)
    {
        if (data == nullptr || byte_size == 0)
        {
            std::fprintf(stderr, "adventures-vk: upload_vertices invalid args\n");
            return false;
        }
        vertex_bytes_ = byte_size;

        VkBufferCreateInfo bi{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bi.size  = byte_size;
        bi.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        if (vkCreateBuffer(device_, &bi, nullptr, &vertex_buffer_) != VK_SUCCESS) return false;

        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device_, vertex_buffer_, &req);
        VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        ai.allocationSize = req.size;
        // host-visible coherent: simple, correct on llvmpipe and desktop drivers
        ai.memoryTypeIndex = find_memory_type(req.memoryTypeBits,
                                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (ai.memoryTypeIndex == UINT32_MAX)
        {
            std::fprintf(stderr, "adventures-vk: no HOST_VISIBLE memory type for vertex buffer\n");
            return false;
        }
        if (vkAllocateMemory(device_, &ai, nullptr, &vertex_memory_) != VK_SUCCESS)
        {
            std::fprintf(stderr, "adventures-vk: vkAllocateMemory(vertex) failed\n");
            return false;
        }
        if (vkBindBufferMemory(device_, vertex_buffer_, vertex_memory_, 0) != VK_SUCCESS) return false;

        void* mapped = nullptr;
        if (vkMapMemory(device_, vertex_memory_, 0, byte_size, 0, &mapped) != VK_SUCCESS) return false;
        std::memcpy(mapped, data, byte_size);
        vkUnmapMemory(device_, vertex_memory_);
        return true;
    }

    bool OffscreenVulkan::record_one_shot(const std::function<void(VkCommandBuffer)>& recorder) const
    {
        VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(cmd_, &bi) != VK_SUCCESS) return false;
        recorder(cmd_);
        return vkEndCommandBuffer(cmd_) == VK_SUCCESS;
    }

    bool OffscreenVulkan::submit_and_wait() const
    {
        VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cmd_;
        if (vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) return false;
        return vkQueueWaitIdle(queue_) == VK_SUCCESS;
    }

    int OffscreenVulkan::upload_texture_rgba(const uint8_t* rgba, uint32_t w, uint32_t h, bool bilinear)
    {
        // NOTE: descriptor pool + set are allocated BEFORE any transfer work —
        // llvmpipe segfaults when vkAllocateDescriptorSets follows a buffer->
        // image copy submission on this Mesa build.
        if (descriptor_pool_ == VK_NULL_HANDLE)
        {
            const VkDescriptorPoolSize pool_size{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8 };
            VkDescriptorPoolCreateInfo pi{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            pi.maxSets       = 8;
            pi.poolSizeCount = 1;
            pi.pPoolSizes    = &pool_size;
            if (vkCreateDescriptorPool(device_, &pi, nullptr, &descriptor_pool_) != VK_SUCCESS)
            {
                std::fprintf(stderr, "adventures-vk: descriptor pool creation failed\n");
                return -1;
            }
        }
        std::fprintf(stderr, "adventures-vk: descriptor pool ok\n");
        TextureSet ts{};
        VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        dai.descriptorPool     = descriptor_pool_;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts        = &texture_set_layout_;
        if (vkAllocateDescriptorSets(device_, &dai, &ts.set) != VK_SUCCESS)
        {
            std::fprintf(stderr, "adventures-vk: descriptor set alloc failed\n");
            return -1;
        }
        std::fprintf(stderr, "adventures-vk: descriptor set alloc ok\n");

        const VkDeviceSize tex_bytes = VkDeviceSize(w) * h * 4;

        // staging buffer (host visible)
        VkBuffer       staging     = VK_NULL_HANDLE;
        VkDeviceMemory staging_mem = VK_NULL_HANDLE;
        VkBufferCreateInfo bi{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bi.size  = tex_bytes;
        bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (vkCreateBuffer(device_, &bi, nullptr, &staging) != VK_SUCCESS) return -1;
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device_, staging, &req);
        VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = find_memory_type(req.memoryTypeBits,
                                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (ai.memoryTypeIndex == UINT32_MAX) return -1;
        if (vkAllocateMemory(device_, &ai, nullptr, &staging_mem) != VK_SUCCESS) return -1;
        vkBindBufferMemory(device_, staging, staging_mem, 0);
        void* mapped = nullptr;
        vkMapMemory(device_, staging_mem, 0, tex_bytes, 0, &mapped);
        std::memcpy(mapped, rgba, size_t(tex_bytes));
        vkUnmapMemory(device_, staging_mem);

        if (!create_image(w, h, kColorFormat,
                          VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                          &ts.image, &ts.memory))
        {
            return -1;
        }

        // upload + layout transitions in one shot
        VkImageMemoryBarrier to_dst{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        to_dst.image            = ts.image;
        to_dst.srcAccessMask    = 0;
        to_dst.dstAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_dst.oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED;
        to_dst.newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_dst.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VkBufferImageCopy region{};
        region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        region.imageExtent      = { w, h, 1 };
        VkImageMemoryBarrier to_shader = to_dst;
        to_shader.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_shader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        to_shader.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_shader.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        const bool uploaded = record_one_shot([&](VkCommandBuffer cb)
        {
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &to_dst);
            vkCmdCopyBufferToImage(cb, staging, ts.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &to_shader);
        }) && submit_and_wait();
        vkDestroyBuffer(device_, staging, nullptr);
        vkFreeMemory(device_, staging_mem, nullptr);
        if (!uploaded)
        {
            std::fprintf(stderr, "adventures-vk: texture upload commands failed\n");
            return -1;
        }

        VkImageViewCreateInfo vi{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        vi.image            = ts.image;
        vi.viewType         = VK_IMAGE_VIEW_TYPE_2D;
        vi.format           = kColorFormat;
        vi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        if (vkCreateImageView(device_, &vi, nullptr, &ts.view) != VK_SUCCESS)
        {
            std::fprintf(stderr, "adventures-vk: texture view creation failed\n");
            return -1;
        }

        VkSamplerCreateInfo si{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        si.magFilter    = bilinear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        si.minFilter    = si.magFilter;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        if (vkCreateSampler(device_, &si, nullptr, &ts.sampler) != VK_SUCCESS)
        {
            std::fprintf(stderr, "adventures-vk: sampler creation failed\n");
            return -1;
        }

        VkDescriptorImageInfo img_info{ ts.sampler, ts.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        write.dstSet          = ts.set;
        write.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.descriptorCount = 1;
        write.pImageInfo      = &img_info;
        vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
        std::fprintf(stderr, "adventures-vk: descriptor write ok\n");

        texture_sets_.push_back(ts);
        return int(texture_sets_.size()) - 1;
    }

    bool OffscreenVulkan::render(const std::vector<VkDraw>& draws, const VkRect2D* scissor_override)
    {
        VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(cmd_, &bi) != VK_SUCCESS) return false;

        const VkClearValue clears[2] = {
            { { { 12.0f / 255.0f, 12.0f / 255.0f, 16.0f / 255.0f, 1.0f } } }, // matches _sw clears
            { { { 1.0f, 0u } } },                                            // depth 1, stencil 0
        };
        VkRenderPassBeginInfo rbi{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
        rbi.renderPass      = render_pass_;
        rbi.framebuffer     = framebuffer_;
        rbi.renderArea      = { { 0, 0 }, { width_, height_ } };
        rbi.clearValueCount = 2;
        rbi.pClearValues    = clears;
        vkCmdBeginRenderPass(cmd_, &rbi, VK_SUBPASS_CONTENTS_INLINE);

        // NEGATIVE viewport height pin: NDC +Y at the top (matches _sw y-flip)
        const VkViewport viewport{ 0.0f, float(height_), float(width_), -float(height_), 0.0f, 1.0f };
        const VkRect2D scissor = scissor_override ? *scissor_override
                                                  : VkRect2D{ { 0, 0 }, { width_, height_ } };
        vkCmdSetViewport(cmd_, 0, 1, &viewport);
        vkCmdSetScissor(cmd_, 0, 1, &scissor);

        const VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd_, 0, 1, &vertex_buffer_, &offset);

        uint32_t bound_pipeline = UINT32_MAX;
        int      bound_set      = -1;
        for (const VkDraw& d : draws)
        {
            if (d.pipeline >= pipelines_.size()) continue;
            if (bound_pipeline != d.pipeline)
            {
                vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines_[d.pipeline]);
                bound_pipeline = d.pipeline;
            }
            const bool textured = d.texture_set >= 0 &&
                                  d.texture_set < int(texture_sets_.size());
            const int set_idx = textured ? d.texture_set : -1;
            if (set_idx != bound_set)
            {
                if (set_idx >= 0)
                {
                    const VkDescriptorSet set = texture_sets_[size_t(set_idx)].set;
                    vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_,
                                            0, 1, &set, 0, nullptr);
                }
                bound_set = set_idx;
            }
            if (d.push != nullptr && d.push_size > 0)
            {
                vkCmdPushConstants(cmd_, pipeline_layout_,
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   0, d.push_size, d.push);
            }
            vkCmdDraw(cmd_, d.vertex_count, 1, d.first_vertex, 0);
        }

        vkCmdEndRenderPass(cmd_);
        if (vkEndCommandBuffer(cmd_) != VK_SUCCESS) return false;

        VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cmd_;
        if (vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) return false;
        if (vkQueueWaitIdle(queue_) != VK_SUCCESS) return false;

        // color image is left in TRANSFER_SRC_OPTIMAL by the render pass: read back
        color_readback_.resize(size_t(width_) * height_ * 4);
        VkBuffer       readback     = VK_NULL_HANDLE;
        VkDeviceMemory readback_mem = VK_NULL_HANDLE;
        VkBufferCreateInfo bi2{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bi2.size  = color_readback_.size();
        bi2.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (vkCreateBuffer(device_, &bi2, nullptr, &readback) != VK_SUCCESS) return false;
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device_, readback, &req);
        VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = find_memory_type(req.memoryTypeBits,
                                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        bool ok = false;
        if (ai.memoryTypeIndex != UINT32_MAX &&
            vkAllocateMemory(device_, &ai, nullptr, &readback_mem) == VK_SUCCESS &&
            vkBindBufferMemory(device_, readback, readback_mem, 0) == VK_SUCCESS)
        {
            ok = record_one_shot([&](VkCommandBuffer cb)
            {
                VkBufferImageCopy region{};
                region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
                region.imageExtent      = { width_, height_, 1 };
                vkCmdCopyImageToBuffer(cb, color_image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                       readback, 1, &region);
            }) && submit_and_wait();
            if (ok)
            {
                void* mapped = nullptr;
                if (vkMapMemory(device_, readback_mem, 0, color_readback_.size(), 0, &mapped) == VK_SUCCESS)
                {
                    std::memcpy(color_readback_.data(), mapped, color_readback_.size());
                    vkUnmapMemory(device_, readback_mem);
                }
                else
                {
                    ok = false;
                }
            }
        }
        vkDestroyBuffer(device_, readback, nullptr);
        vkFreeMemory(device_, readback_mem, nullptr);
        return ok;
    }

    bool OffscreenVulkan::save_png(const char* path) const
    {
        if (color_readback_.empty() || path == nullptr) return false;
        const int stride = int(width_) * 4;
        return stbi_write_png(path, int(width_), int(height_), 4,
                              color_readback_.data(), stride) != 0;
    }
}
