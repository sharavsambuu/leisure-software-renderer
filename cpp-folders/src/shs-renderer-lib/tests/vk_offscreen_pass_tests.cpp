#include <cstdio>
#include <cstdint>
#include <fstream>
#include <vector>

#ifdef SHS_OFFSCREEN_SHADER_DIR
static std::vector<uint32_t> read_spirv(const char* path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const auto bytes = file.tellg();
    if (bytes <= 0 || bytes % 4 != 0) return {};
    std::vector<uint32_t> words(static_cast<size_t>(bytes) / 4);
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(words.data()), bytes)) return {};
    return words;
}
#endif
#include "shs/execution/rhi/drivers/vulkan/vk_backend.hpp"

#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); return 1; } } while (false)

// Recording evidence only. Submission/readback and known pixels belong to G3.
int main()
{
    using namespace shs;
    struct Device
    {
        VulkanDeviceManager value;
        ~Device() { value.shutdown(); }
    } device;
    if (!device.value.initialize({}))
    {
        std::fprintf(stderr, "SKIP: Vulkan device unavailable\n");
        return 77;
    }
    const VkDevice vk = device.value.device();
    std::fprintf(stderr, "Device: %s\n", device.value.info().device_name);
    struct Objects
    {
        VkDevice device;
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkCommandPool pool = VK_NULL_HANDLE;
        ~Objects()
        {
            if (pool) vkDestroyCommandPool(device, pool, nullptr);
            if (image) vkDestroyImage(device, image, nullptr);
            if (memory) vkFreeMemory(device, memory, nullptr);
        }
    } objects{vk};
    RHIImageDesc desc{};
    desc.width = 32;
    desc.height = 32;
    desc.format = RHIFormat::RGBA8_UNorm;
    desc.usage = RHIImageUsage_ColorAttachment;
    CHECK(VulkanOffscreenPass::supports(desc));
    for (int invalid = 0; invalid < 6; ++invalid)
    {
        auto d = desc;
        if (invalid == 0) d.width = 0;
        if (invalid == 1) d.format = RHIFormat::D32F;
        if (invalid == 2) d.layers = 2;
        if (invalid == 3) d.mip_levels = 2;
        if (invalid == 4) d.usage = RHIImageUsage_Sampled;
        if (invalid == 5) d.type = RHIImageType::TexCube;
        CHECK(!VulkanOffscreenPass::supports(d));
    }
    auto ci = vk_image_create_info(desc);
    CHECK(vkCreateImage(vk, &ci, nullptr, &objects.image) == VK_SUCCESS);
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(vk, objects.image, &requirements);
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(device.value.physical(), &properties);
    VkMemoryAllocateInfo allocation{};
    allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = vk_pick_memory_type(properties, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    CHECK(allocation.memoryTypeIndex != UINT32_MAX);
    CHECK(vkAllocateMemory(vk, &allocation, nullptr, &objects.memory) == VK_SUCCESS);
    CHECK(vkBindImageMemory(vk, objects.image, objects.memory, 0) == VK_SUCCESS);
    VulkanOffscreenPass pass;
    CHECK(!pass.initialize(vk, 0, objects.image, desc));
    CHECK(!pass.framebuffer());
    CHECK(pass.initialize(vk, 1, objects.image, desc));
    CHECK(!pass.initialize(vk, 1, objects.image, desc));
    CHECK(pass.framebuffer()); // failed re-prepare leaves live objects intact
    CHECK(pass.accepts({1, 0, true, false}));
    CHECK(!pass.accepts({1, 0, false, false}));
    CHECK(!pass.accepts({1, 2, true, true}));

    VkCommandPoolCreateInfo pool{};
    pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool.queueFamilyIndex = device.value.info().graphics_queue_family;
    CHECK(vkCreateCommandPool(vk, &pool, nullptr, &objects.pool) == VK_SUCCESS);
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = objects.pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    CHECK(vkAllocateCommandBuffers(vk, &ai, &cmd) == VK_SUCCESS);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    CHECK(vkBeginCommandBuffer(cmd, &bi) == VK_SUCCESS);
    VulkanBufferPool buffers{std::pmr::get_default_resource()};
    VulkanImagePool images{std::pmr::get_default_resource()};
    images.insert_or_assign(1, objects.image);
    VulkanCommandRecorder recorder{vk, cmd, buffers, images, nullptr, &pass};
    const RHICmd rejected[] = {rhi_cmd_barrier({}), rhi_cmd_begin_pass({1, 0, false, false}), rhi_cmd_end_pass()};
    auto result = record_commands(rejected, recorder);
    CHECK(!result);
    CHECK(result.error().code == VulkanRecordingError::UnsupportedCommand);
    CHECK(result.error().stage == VulkanRecordingStage::Validation);
    CHECK(result.error().command_index == 1);
    CHECK(!recorder.end_pass({}));
    const RHICmd stream[] = {rhi_cmd_begin_pass({1, 0, true, false}), rhi_cmd_end_pass()};
    CHECK(record_commands(stream, recorder));
    CHECK(record_commands(stream, recorder));
    CHECK(!recorder.end_pass({}));
#ifdef SHS_OFFSCREEN_SHADER_DIR
    auto vs = read_spirv(SHS_OFFSCREEN_SHADER_DIR "/offscreen_vs.spv");
    auto fs = read_spirv(SHS_OFFSCREEN_SHADER_DIR "/offscreen_fs.spv");
    CHECK(!vs.empty() && !fs.empty());
    RHIGraphicsPipelineDesc pd{};
    pd.vs = {RHIShaderStage::Vertex, vs.data(), vs.size() * 4, "vs_main"};
    pd.fs = {RHIShaderStage::Fragment, fs.data(), fs.size() * 4, "fs_main"};
    pd.rt.has_depth = false;
    pd.depth = {false, false};
    pd.raster.cull = RHICullMode::None;
    CHECK(VulkanOffscreenPipeline::supports(pd));
    for (int invalid = 0; invalid < 13; ++invalid)
    {
        auto d = pd;
        if (invalid == 0) d.vs.bytecode = nullptr;
        if (invalid == 1) d.fs.bytecode_size = 3;
        if (invalid == 2) d.vs.stage = RHIShaderStage::Compute;
        if (invalid == 3) d.fs.entry = "";
        if (invalid == 4) d.rt.has_depth = true;
        if (invalid == 5) d.rt.color_format = RHIFormat::D32F;
        if (invalid == 6) d.depth.enable_test = true;
        if (invalid == 7) d.depth.enable_write = true;
        if (invalid == 8) d.blend.enable = true;
        if (invalid == 9) d.raster.depth_clamp = true;
        if (invalid == 10) d.raster.cull = static_cast<RHICullMode>(255);
        if (invalid == 11) d.raster.front_face = static_cast<RHIFrontFace>(255);
        if (invalid == 12) d.fs.entry = nullptr;
        CHECK(!VulkanOffscreenPipeline::supports(d));
        VulkanOffscreenPipeline rejected_pipeline;
        CHECK(!rejected_pipeline.initialize(1, d, pass));
        CHECK(!rejected_pipeline.pipeline() && !rejected_pipeline.layout());
    }
    auto newer_spirv = vs;
    newer_spirv[1] = 0x00010500;
    auto unsupported = pd;
    unsupported.vs.bytecode = newer_spirv.data();
    CHECK(!VulkanOffscreenPipeline::supports(unsupported));
    VulkanPipelineCache cache;
    VulkanOffscreenPipeline graphics;
    const auto id = cache.intern_graphics(pd, [&](uint64_t key, const auto& d) {
        return graphics.initialize(key, d, pass);
    });
    CHECK(id && graphics.pipeline() && graphics.layout());
    CHECK(!graphics.initialize(id, pd, pass));
    CHECK(graphics.accepts(*cache.find_graphics(id), pass));
    bool created_again = false;
    CHECK(cache.intern_graphics(pd, [&](uint64_t, const auto&) { created_again = true; return false; }) == id);
    CHECK(!created_again);
    VulkanCommandRecorder bound{vk, cmd, buffers, images, &cache, &pass, &graphics};
    CHECK(!bound.bind_pipeline({id})); // direct bind outside a pass
    const RHICmd bind_stream[] = {rhi_cmd_begin_pass({1, 0, true, false}),
        rhi_cmd_bind_pipeline(id), rhi_cmd_end_pass()};
    VulkanCommandRecorder bookkeeping{vk, cmd, buffers, images, &cache, &pass};
    auto unavailable = record_commands(bind_stream, bookkeeping);
    CHECK(!unavailable && unavailable.error().code == VulkanRecordingError::UnsupportedCommand);
    CHECK(unavailable.error().stage == VulkanRecordingStage::Validation);
    CHECK(unavailable.error().command_index == 1);
    CHECK(!bookkeeping.end_pass({}));
    CHECK(record_commands(bind_stream, bound));
    CHECK(record_commands(bind_stream, bound));
    std::fprintf(stderr, "PASS: real graphics pipeline creation and value-stream bind recording\n");
#else
    std::fprintf(stderr, "NOTE: pipeline coverage unavailable (slangc not configured); attachments only\n");
#endif
    CHECK(vkEndCommandBuffer(cmd) == VK_SUCCESS);
    // No submitted work: discard references before exercising explicit reset/recreate.
    CHECK(vkResetCommandPool(vk, objects.pool, 0) == VK_SUCCESS);
#ifdef SHS_OFFSCREEN_SHADER_DIR
    graphics.reset();
    graphics.reset();
    CHECK(!graphics.pipeline() && !graphics.layout());
    CHECK(!graphics.accepts(*cache.find_graphics(id), pass));
    auto retired = bound.validate_command(rhi_cmd_bind_pipeline(id), true);
    CHECK(!retired && retired.error().code == VulkanRecordingError::UnsupportedCommand);
    CHECK(graphics.initialize(id, pd, pass)); // explicit realization after retirement
    CHECK(graphics.accepts(*cache.find_graphics(id), pass));
    auto wrong = *cache.find_graphics(id);
    ++wrong.id;
    CHECK(!graphics.accepts(wrong, pass));
    wrong = *cache.find_graphics(id);
    ++wrong.desc_hash;
    CHECK(!graphics.accepts(wrong, pass));
    graphics.reset();
#endif
    pass.reset();
    pass.reset();
    CHECK(!pass.framebuffer());
    CHECK(pass.initialize(vk, 1, objects.image, desc));
    std::fprintf(stderr, "PASS: explicit offscreen attachments and value-stream begin/end recording (no submission)\n");
    return 0;
}
