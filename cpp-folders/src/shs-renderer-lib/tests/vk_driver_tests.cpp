#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

#include <vulkan/vulkan.h>

#include "shs/core/context.hpp"
#include "shs/execution/rhi/drivers/vulkan/vk_backend.hpp"
#include "shs/execution/rhi/drivers/vulkan/vk_commands.hpp"

// GPU-free value tests for the P2 Vulkan driver pod (roadmap P2 ctest gate:
// shs_renderer_vop_vk_driver_tests). Every layer tested here is pure
// translation/bookkeeping — no VkDevice is created and no libvulkan symbol
// is called: the binary links only Vulkan headers. Device-dependent behavior
// is covered by the headless-mode contract tests.
namespace
{
    // --- pure translation: mappers ------------------------------------------

    bool test_format_mapping()
    {
        using shs::RHIFormat;
        if (shs::vk_format_of(RHIFormat::RGBA8_UNorm) != VK_FORMAT_R8G8B8A8_UNORM) return false;
        if (shs::vk_format_of(RHIFormat::BGRA8_UNorm) != VK_FORMAT_B8G8R8A8_UNORM) return false;
        if (shs::vk_format_of(RHIFormat::RGBA16F) != VK_FORMAT_R16G16B16A16_SFLOAT) return false;
        if (shs::vk_format_of(RHIFormat::RGBA32F) != VK_FORMAT_R32G32B32A32_SFLOAT) return false;
        if (shs::vk_format_of(RHIFormat::D24S8) != VK_FORMAT_D24_UNORM_S8_UINT) return false;
        if (shs::vk_format_of(RHIFormat::D32F) != VK_FORMAT_D32_SFLOAT) return false;
        if (shs::vk_format_of(RHIFormat::Unknown) != VK_FORMAT_UNDEFINED) return false;
        return true;
    }

    bool test_usage_and_memory_mapping()
    {
        const VkBufferUsageFlags bu = shs::vk_buffer_usage_of(
            shs::RHIBufferUsage_Vertex | shs::RHIBufferUsage_TransferDst);
        if (!(bu & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) || !(bu & VK_BUFFER_USAGE_TRANSFER_DST_BIT)) return false;
        if (bu & VK_BUFFER_USAGE_INDEX_BUFFER_BIT) return false;

        const VkImageUsageFlags iu = shs::vk_image_usage_of(
            shs::RHIImageUsage_ColorAttachment | shs::RHIImageUsage_Sampled);
        if (!(iu & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) || !(iu & VK_IMAGE_USAGE_SAMPLED_BIT)) return false;

        if (shs::vk_memory_props_of(shs::RHIMemoryClass::GPUOnly) != VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) return false;
        if (!(shs::vk_memory_props_of(shs::RHIMemoryClass::CPUVisible) & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) return false;
        return true;
    }

    bool test_stage_access_mapping()
    {
        if (shs::vk_stage_of(shs::RHIPipelineStage::ColorOutput) != VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT) return false;
        if (shs::vk_stage_of(shs::RHIPipelineStage::ComputeShader) != VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) return false;
        if (shs::vk_access_of(shs::RHIAccess::None) != 0) return false;
        if (!(shs::vk_access_of(shs::RHIAccess::Write) & VK_ACCESS_SHADER_WRITE_BIT)) return false;
        return true;
    }

    bool test_memory_type_picker()
    {
        VkPhysicalDeviceMemoryProperties props{};
        props.memoryTypeCount = 3;
        props.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        props.memoryTypes[1].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        props.memoryTypes[2].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        if (shs::vk_pick_memory_type(props, 0b111, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 1) return false;
        if (shs::vk_pick_memory_type(props, 0b001, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) return false;
        if (shs::vk_pick_memory_type(props, 0b100, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 2) return false;
        if (shs::vk_pick_memory_type(props, 0b001, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != UINT32_MAX) return false;
        return true;
    }

    // --- descriptor hashing: explicit cache keys ------------------------------

    bool test_desc_hash_stability()
    {
        shs::RHIBufferDesc a{};
        a.size_bytes = 1024;
        a.usage = shs::RHIBufferUsage_Vertex | shs::RHIBufferUsage_TransferDst;
        a.memory = shs::RHIMemoryClass::GPUOnly;

        if (shs::hash_buffer_desc(a) != shs::hash_buffer_desc(a)) return false;

        shs::RHIBufferDesc b = a;
        b.usage |= shs::RHIBufferUsage_Storage;
        if (shs::hash_buffer_desc(a) == shs::hash_buffer_desc(b)) return false;

        shs::RHIBufferDesc c = a;
        c.memory = shs::RHIMemoryClass::CPUVisible;
        if (shs::hash_buffer_desc(a) == shs::hash_buffer_desc(c)) return false;

        shs::RHIImageDesc i1{};
        i1.width = 64; i1.height = 64; i1.format = shs::RHIFormat::RGBA8_UNorm;
        shs::RHIImageDesc i2 = i1;
        if (shs::hash_image_desc(i1) != shs::hash_image_desc(i2)) return false;
        i2.layers = 6;
        if (shs::hash_image_desc(i1) == shs::hash_image_desc(i2)) return false;
        return true;
    }

    bool test_shader_module_hash_content()
    {
        static const uint32_t code_a[] = {0x07230203, 0x00010000, 0x0008000a, 0xdeadbeef};
        static const uint32_t code_b[] = {0x07230203, 0x00010000, 0x0008000a, 0xfeedface};

        shs::RHIShaderModuleDesc m1{};
        m1.stage = shs::RHIShaderStage::Vertex;
        m1.bytecode = code_a;
        m1.bytecode_size = sizeof(code_a);
        shs::RHIShaderModuleDesc m2 = m1;
        if (shs::hash_shader_module_desc(m1) != shs::hash_shader_module_desc(m2)) return false;

        m2.bytecode = code_b; // same size, different content
        if (shs::hash_shader_module_desc(m1) == shs::hash_shader_module_desc(m2)) return false;

        m2.bytecode = code_a;
        m2.stage = shs::RHIShaderStage::Fragment;
        if (shs::hash_shader_module_desc(m1) == shs::hash_shader_module_desc(m2)) return false;
        return true;
    }

    // --- resource registry: hash-keyed explicit cache --------------------------

    bool test_resource_registry_dedupe()
    {
        shs::VulkanResourceRegistry reg;

        shs::RHIBufferDesc d{};
        d.size_bytes = 4096;
        d.usage = shs::RHIBufferUsage_Vertex | shs::RHIBufferUsage_Index;
        d.memory = shs::RHIMemoryClass::GPUOnly;

        uint32_t creates = 0;
        const uint64_t id1 = reg.intern_buffer(d, [&](uint64_t, const shs::RHIBufferDesc&) {
            creates++;
            return true;
        });
        const uint64_t id2 = reg.intern_buffer(d, [&](uint64_t, const shs::RHIBufferDesc&) {
            creates++;
            return true;
        });
        if (id1 == 0 || id1 != id2) return false;   // identical desc → identical stable ID
        if (creates != 1) return false;             // explicit cache: one creation call

        const shs::VulkanResourceStats& st = reg.stats();
        if (st.create_calls != 1 || st.cache_hits != 1) return false;
        if (reg.find_buffer(id1) == nullptr) return false;
        if (reg.find_buffer(shs::VulkanResourceRegistry::kBufferIdBase + 99) != nullptr) return false;
        return true;
    }

    bool test_resource_registry_failure_semantics()
    {
        shs::VulkanResourceRegistry reg;
        shs::RHIBufferDesc d{};
        d.size_bytes = 128;
        d.usage = shs::RHIBufferUsage_Uniform;

        const uint64_t failed = reg.intern_buffer(d, [](uint64_t, const shs::RHIBufferDesc&) {
            return false; // device creation failure path
        });
        if (failed != 0) return false;                   // no ID on failure
        if (reg.stats().create_calls != 0) return false; // no create counted

        // Retry after failure succeeds and mints a fresh ID.
        const uint64_t ok = reg.intern_buffer(d, [](uint64_t, const shs::RHIBufferDesc&) { return true; });
        if (ok == 0 || reg.find_buffer(ok) == nullptr) return false;

        // Image namespace is separate from buffer namespace.
        shs::RHIImageDesc img{};
        img.width = 8; img.height = 8; img.format = shs::RHIFormat::RGBA8_UNorm;
        img.usage = shs::RHIImageUsage_ColorAttachment;
        const uint64_t iid = reg.intern_image(img, [](uint64_t, const shs::RHIImageDesc&) { return true; });
        if (iid == 0 || reg.find_image(iid) == nullptr) return false;
        if ((iid >> 56) != (shs::VulkanResourceRegistry::kImageIdBase >> 56)) return false;
        return true;
    }

    // --- pipeline cache: explicit, hash-keyed ----------------------------------

    bool test_pipeline_cache_explicit()
    {
        static const uint32_t vs_code[] = {0x07230203, 0x1};
        static const uint32_t fs_code[] = {0x07230203, 0x2};

        shs::RHIGraphicsPipelineDesc d{};
        d.vs.stage = shs::RHIShaderStage::Vertex;
        d.vs.bytecode = vs_code;
        d.vs.bytecode_size = sizeof(vs_code);
        d.fs.stage = shs::RHIShaderStage::Fragment;
        d.fs.bytecode = fs_code;
        d.fs.bytecode_size = sizeof(fs_code);
        d.rt.color_format = shs::RHIFormat::BGRA8_UNorm;

        shs::VulkanPipelineCache cache;
        uint32_t creates = 0;
        const uint64_t p1 = cache.intern_graphics(d, [&](uint64_t, const shs::RHIGraphicsPipelineDesc&) {
            creates++;
            return true;
        });
        const uint64_t p2 = cache.intern_graphics(d, [&](uint64_t, const shs::RHIGraphicsPipelineDesc&) {
            creates++;
            return true;
        });
        if (p1 == 0 || p1 != p2 || creates != 1) return false;
        if (cache.stats().cache_hits != 1) return false;

        const shs::VulkanPipelineRecord* record = cache.find_graphics(p1);
        if (!record || record->id != p1 ||
            record->desc_hash != shs::hash_graphics_pipeline_desc(d)) return false;

        // A state field change must produce a distinct slot (no accidental reuse).
        shs::RHIGraphicsPipelineDesc d2 = d;
        d2.depth.enable_write = !d2.depth.enable_write;
        const uint64_t p3 = cache.intern_graphics(d2, [](uint64_t, const shs::RHIGraphicsPipelineDesc&) { return true; });
        if (p3 == 0 || p3 == p1) return false;
        record = cache.find_graphics(p3);
        if (!record || record->id != p3 ||
            record->desc_hash != shs::hash_graphics_pipeline_desc(d2)) return false;
        record = cache.find_graphics(p1);
        if (!record || record->id != p1) return false;
        if (cache.find_graphics(0) || cache.find_graphics(UINT64_MAX)) return false;

        auto failed_desc = d;
        failed_desc.blend.enable = !failed_desc.blend.enable;
        uint64_t failed_id = 0;
        if (cache.intern_graphics(failed_desc, [&](uint64_t id, const shs::RHIGraphicsPipelineDesc&) {
                failed_id = id;
                return false;
            }) != 0) return false;
        if (failed_id == 0 || cache.find_graphics(failed_id)) return false;
        const uint64_t retried = cache.intern_graphics(failed_desc, [](uint64_t, const shs::RHIGraphicsPipelineDesc&) { return true; });
        record = cache.find_graphics(retried);
        if (!record || record->id != retried || cache.find_graphics(failed_id)) return false;

        // Shader modules intern independently and dedupe by content hash.
        uint32_t module_creates = 0;
        const shs::RHIShaderModuleDesc vs = d.vs;
        const uint64_t m1 = cache.intern_shader_module(vs, [&](uint64_t, const shs::RHIShaderModuleDesc&) {
            module_creates++;
            return true;
        });
        const uint64_t m2 = cache.intern_shader_module(vs, [&](uint64_t, const shs::RHIShaderModuleDesc&) {
            module_creates++;
            return true;
        });
        if (m1 == 0 || m1 != m2 || module_creates != 1) return false;
        return true;
    }

    // --- command stream translation (spy sink, exact ordering) ------------------

    struct SpySink
    {
        std::vector<uint32_t> calls{};
        std::vector<uint64_t> ids{};

        void begin_pass(const shs::RHICmdBeginPassDesc& d) { calls.push_back(1); ids.push_back(d.color_target); }
        void end_pass(const shs::RHICmdEndPassDesc&) { calls.push_back(2); ids.push_back(0); }
        void bind_pipeline(const shs::RHICmdBindPipelineDesc& d) { calls.push_back(3); ids.push_back(d.pipeline); }
        void bind_vertex_buffer(const shs::RHICmdBindVertexBufferDesc& d) { calls.push_back(4); ids.push_back(d.buffer); }
        void bind_index_buffer(const shs::RHICmdBindIndexBufferDesc& d) { calls.push_back(5); ids.push_back(d.buffer); }
        void draw_indexed(const shs::RHICmdDrawIndexedDesc& d) { calls.push_back(6); ids.push_back(d.index_count); }
        void dispatch(const shs::RHICmdDispatchDesc& d) { calls.push_back(7); ids.push_back(d.group_x); }
        void barrier(const shs::RHICmdBarrierDesc& d) { calls.push_back(8); ids.push_back((uint64_t)d.memory.src_stage); }
    };

    bool test_command_stream_translation()
    {
        std::vector<shs::RHICmd> stream;
        shs::RHICmdBeginPassDesc bp{};
        bp.color_target = 0x53ull;
        bp.clear_color = true;
        stream.push_back(shs::rhi_cmd_begin_pass(bp));
        stream.push_back(shs::rhi_cmd_bind_pipeline(0x47ull));
        stream.push_back(shs::rhi_cmd_bind_vertex_buffer(0x52ull, 0));
        stream.push_back(shs::rhi_cmd_bind_index_buffer(0x52ull, 16, true));
        shs::RHICmdDrawIndexedDesc di{};
        di.index_count = 3;
        stream.push_back(shs::rhi_cmd_draw_indexed(di));
        stream.push_back(shs::rhi_cmd_end_pass());

        SpySink sink;
        shs::record_commands(std::span<const shs::RHICmd>(stream.data(), stream.size()), sink);

        if (sink.calls.size() != 6) return false;
        const uint32_t expected[] = {1, 3, 4, 5, 6, 2};
        for (size_t i = 0; i < 6; ++i)
        {
            if (sink.calls[i] != expected[i]) return false;
        }
        if (sink.ids[0] != 0x53ull || sink.ids[1] != 0x47ull || sink.ids[2] != 0x52ull) return false;
        if (sink.ids[3] != 0x52ull || sink.ids[4] != 3) return false;
        return true;
    }

    bool test_command_stream_dispatch_and_barrier()
    {
        std::vector<shs::RHICmd> stream;
        stream.push_back(shs::rhi_cmd_dispatch(4, 4, 1));
        shs::RHIMemoryBarrierDesc mb{};
        mb.src_stage = shs::RHIPipelineStage::ComputeShader;
        mb.dst_stage = shs::RHIPipelineStage::FragmentShader;
        mb.src_access = shs::RHIAccess::Write;
        mb.dst_access = shs::RHIAccess::Read;
        stream.push_back(shs::rhi_cmd_barrier(mb));

        SpySink sink;
        shs::record_commands(std::span<const shs::RHICmd>(stream.data(), stream.size()), sink);
        if (sink.calls.size() != 2) return false;
        if (sink.calls[0] != 7 || sink.ids[0] != 4) return false;
        if (sink.calls[1] != 8 || sink.ids[1] != (uint64_t)shs::RHIPipelineStage::ComputeShader) return false;
        return true;
    }

    // --- frame sync: slot rotation, timeline semantics ---------------------------

    bool test_frame_sync_slots()
    {
        shs::VulkanFrameSync sync;
        sync.configure(2);

        (void)sync.begin_frame(0);
        sync.end_frame(0);
        if (!sync.frame_in_flight(0)) return false;
        if (sync.frame_in_flight(1)) return false;

        // Re-entering the in-flight slot accounts a fence wait.
        const uint64_t waits_before = sync.stats().waits_issued;
        (void)sync.begin_frame(2); // frame 2 → slot 0 again
        if (sync.stats().waits_issued != waits_before + 1) return false;
        if (sync.frame_in_flight(0)) return false;

        // Timeline advanced to last submitted frame + 1.
        const uint64_t tl = sync.graphics_timeline_id();
        if (sync.timeline_value(tl) != 1) return false;
        return true;
    }

    bool test_frame_sync_triple_buffer()
    {
        shs::VulkanFrameSync sync;
        sync.configure(3);
        for (uint64_t f = 0; f < 3; ++f)
        {
            (void)sync.begin_frame(f);
            sync.end_frame(f);
        }
        if (sync.stats().waits_issued != 0) return false; // no slot revisited yet
        (void)sync.begin_frame(3);                        // slot 0 revisited
        if (sync.stats().waits_issued != 1) return false;
        if (sync.stats().begin_frames != 4 || sync.stats().end_frames != 3) return false;
        return true;
    }

    // --- backend: headless contracts ---------------------------------------------

    bool test_backend_headless_contract()
    {
        shs::VulkanRenderBackend backend;
        if (backend.type() != shs::RenderBackendType::Vulkan) return false;
        if (backend.device_ready()) return false; // no device by default

        const shs::BackendCapabilities caps = backend.capabilities();
        if (!caps.supports_offscreen) return false;
        if (caps.limits.max_frames_in_flight != 2) return false;

        // Headless mode: GPU entry points no-op, frame bookkeeping works.
        shs::Context ctx{};
        shs::RenderBackendFrameInfo frame{};
        frame.frame_index = 0;
        frame.width = 64;
        frame.height = 64;
        backend.begin_frame(ctx, frame);
        backend.end_frame(ctx, frame);
        if (backend.frame_sync_stats().begin_frames != 1) return false;

        std::vector<shs::RHICmd> stream;
        stream.push_back(shs::rhi_cmd_bind_pipeline(0x47ull));
        backend.record_frame_commands(std::span<const shs::RHICmd>(stream.data(), stream.size()));

        backend.shutdown();
        return true;
    }

    bool test_backend_create_info_purity()
    {
        shs::RHIBufferDesc bd{};
        bd.size_bytes = 256;
        bd.usage = shs::RHIBufferUsage_Vertex;
        const VkBufferCreateInfo bci = shs::vk_buffer_create_info(bd);
        if (bci.size != 256 || bci.usage != VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) return false;

        shs::RHIImageDesc id{};
        id.width = 32; id.height = 16; id.format = shs::RHIFormat::RGBA8_UNorm;
        id.usage = shs::RHIImageUsage_ColorAttachment | shs::RHIImageUsage_Sampled;
        const VkImageCreateInfo ici = shs::vk_image_create_info(id);
        if (ici.extent.width != 32 || ici.extent.height != 16) return false;
        if (ici.format != VK_FORMAT_R8G8B8A8_UNORM) return false;
        if (ici.mipLevels != 1 || ici.arrayLayers != 1) return false;

        shs::RHISamplerDesc sd{};
        const VkSamplerCreateInfo sci = shs::vk_sampler_create_info(sd);
        if (sci.magFilter != VK_FILTER_LINEAR || sci.addressModeU != VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE) return false;

        shs::RHIRasterStateDesc rs{};
        const VkPipelineRasterizationStateCreateInfo rci = shs::vk_raster_state(rs);
        if (rci.cullMode != VK_CULL_MODE_BACK_BIT || rci.frontFace != VK_FRONT_FACE_COUNTER_CLOCKWISE) return false;

        shs::RHIDepthStateDesc ds{};
        const VkPipelineDepthStencilStateCreateInfo dci = shs::vk_depth_state(ds);
        if (dci.depthTestEnable != VK_TRUE || dci.depthWriteEnable != VK_TRUE) return false;

        shs::RHIBlendStateDesc bs{};
        VkPipelineColorBlendAttachmentState att{};
        const VkPipelineColorBlendStateCreateInfo bsci = shs::vk_blend_state(bs, att);
        if (bsci.attachmentCount != 1 || att.blendEnable != VK_FALSE) return false;
        if ((att.colorWriteMask & VK_COLOR_COMPONENT_A_BIT) == 0) return false;
        return true;
    }
}

int main()
{
    struct Case { const char* name; bool (*fn)(); };
    const Case cases[] = {
        {"vk_format_mapping", test_format_mapping},
        {"vk_usage_memory_mapping", test_usage_and_memory_mapping},
        {"vk_stage_access_mapping", test_stage_access_mapping},
        {"vk_memory_type_picker", test_memory_type_picker},
        {"vk_desc_hash_stability", test_desc_hash_stability},
        {"vk_shader_module_hash_content", test_shader_module_hash_content},
        {"vk_resource_registry_dedupe", test_resource_registry_dedupe},
        {"vk_resource_registry_failure", test_resource_registry_failure_semantics},
        {"vk_pipeline_cache_explicit", test_pipeline_cache_explicit},
        {"vk_command_stream_translation", test_command_stream_translation},
        {"vk_command_stream_dispatch_barrier", test_command_stream_dispatch_and_barrier},
        {"vk_frame_sync_slots", test_frame_sync_slots},
        {"vk_frame_sync_triple_buffer", test_frame_sync_triple_buffer},
        {"vk_backend_headless_contract", test_backend_headless_contract},
        {"vk_backend_create_info_purity", test_backend_create_info_purity},
    };

    bool ok = true;
    for (const Case& c : cases)
    {
        const bool passed = c.fn();
        std::fprintf(stderr, "[vk-driver-tests] %-36s %s\n", c.name,
                     passed ? "ok" : "FAIL");
        ok = ok && passed;
    }

    if (ok)
    {
        std::fprintf(stderr, "[vk-driver-tests] all %zu cases passed\n",
                     sizeof(cases) / sizeof(cases[0]));
        return 0;
    }
    std::fprintf(stderr, "[vk-driver-tests] FAILURES detected\n");
    return 1;
}
