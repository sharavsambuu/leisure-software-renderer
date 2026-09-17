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

// G2 draw recording plus focused G3 submission/readback evidence.
// Factory-facing execution and buffer upload remain separate acceptance items.
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
        VkBuffer readback = VK_NULL_HANDLE;
        VkDeviceMemory readback_memory = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        ~Objects()
        {
            // Also protects failure exits after submission.
            vkDeviceWaitIdle(device);
            if (fence) vkDestroyFence(device, fence, nullptr);
            if (readback) vkDestroyBuffer(device, readback, nullptr);
            if (readback_memory) vkFreeMemory(device, readback_memory, nullptr);
            if (pool) vkDestroyCommandPool(device, pool, nullptr);
            if (image) vkDestroyImage(device, image, nullptr);
            if (memory) vkFreeMemory(device, memory, nullptr);
        }
    } objects{vk};
    RHIImageDesc desc{};
    desc.width = 32;
    desc.height = 32;
    desc.format = RHIFormat::RGBA8_UNorm;
    desc.usage = RHIImageUsage_ColorAttachment | RHIImageUsage_TransferSrc;
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
        rhi_cmd_bind_pipeline(id), rhi_cmd_draw({3}), rhi_cmd_end_pass()};
    VulkanCommandRecorder bookkeeping{vk, cmd, buffers, images, &cache, &pass};
    auto unavailable = record_commands(bind_stream, bookkeeping);
    CHECK(!unavailable && unavailable.error().code == VulkanRecordingError::UnsupportedCommand);
    CHECK(unavailable.error().stage == VulkanRecordingStage::Validation);
    CHECK(unavailable.error().command_index == 1);
    CHECK(!bookkeeping.end_pass({}));
    const RHICmd outside[] = {rhi_cmd_draw({3})};
    auto bad_draw = record_commands(outside, bound);
    CHECK(!bad_draw && bad_draw.error().code == VulkanRecordingError::InvalidRecordingOrder);
    CHECK(bad_draw.error().command == VulkanCommandKind::Draw);
    const RHICmd unbound[] = {rhi_cmd_begin_pass({1, 0, true, false}),
        rhi_cmd_draw({3}), rhi_cmd_end_pass()};
    bad_draw = record_commands(unbound, bound);
    CHECK(!bad_draw && bad_draw.error().code == VulkanRecordingError::MissingBinding);
    CHECK(bad_draw.error().command_index == 1);
    CHECK(!bound.end_pass({})); // rejection was preflight, not partial recording
    CHECK(!bound.draw({3}));
    CHECK(bound.begin_pass({1, 0, true, false}));
    auto direct = bound.draw({3});
    CHECK(!direct && direct.error() == VulkanRecordingError::MissingBinding);
    CHECK(bound.end_pass({}));
    CHECK(record_commands(bind_stream, bound));
    CHECK(record_commands(bind_stream, bound));
    std::fprintf(stderr, "PASS: real graphics pipeline creation and value-stream bind/draw recording\n");
#else
    std::fprintf(stderr, "NOTE: pipeline coverage unavailable (slangc not configured); attachments only\n");
#endif
    VkBufferCreateInfo read_ci{};
    read_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    read_ci.size = 32 * 32 * 4;
    read_ci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    CHECK(vkCreateBuffer(vk, &read_ci, nullptr, &objects.readback) == VK_SUCCESS);
    vkGetBufferMemoryRequirements(vk, objects.readback, &requirements);
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = vk_pick_memory_type(properties, requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    CHECK(allocation.memoryTypeIndex != UINT32_MAX);
    CHECK(vkAllocateMemory(vk, &allocation, nullptr, &objects.readback_memory) == VK_SUCCESS);
    CHECK(vkBindBufferMemory(vk, objects.readback, objects.readback_memory, 0) == VK_SUCCESS);
    VkImageMemoryBarrier image_barrier{};
    image_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    image_barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    image_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    image_barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    image_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    image_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_barrier.image = objects.image;
    image_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &image_barrier);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {32, 32, 1};
    vkCmdCopyImageToBuffer(cmd, objects.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        objects.readback, 1, &copy);
    VkMemoryBarrier host_barrier{};
    host_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    host_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    host_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
        0, 1, &host_barrier, 0, nullptr, 0, nullptr);
    CHECK(vkEndCommandBuffer(cmd) == VK_SUCCESS);
    const auto missing_command = vulkan_submit_sync(vk, device.value.graphics_queue(), VK_NULL_HANDLE);
    CHECK(!missing_command && missing_command.error().stage == VulkanSubmitStage::Prerequisite);
    VulkanRenderBackend uninitialized;
    const auto missing_device = uninitialized.submit_commands_sync();
    CHECK(!missing_device && missing_device.error().stage == VulkanSubmitStage::Prerequisite);
    CHECK(vulkan_submit_sync(vk, device.value.graphics_queue(), cmd));
    void* mapped = nullptr;
    CHECK(vkMapMemory(vk, objects.readback_memory, 0, VK_WHOLE_SIZE, 0, &mapped) == VK_SUCCESS);
    std::vector<uint8_t> pixels(static_cast<uint8_t*>(mapped), static_cast<uint8_t*>(mapped) + 32 * 32 * 4);
    vkUnmapMemory(vk, objects.readback_memory);
    const auto pixel_is = [&](int x, int y, int r, int g, int b, int a) {
        const auto* p = pixels.data() + (y * 32 + x) * 4;
        return p[0] == r && p[1] == g && p[2] == b && p[3] == a;
    };
    CHECK(pixel_is(1, 1, 0, 0, 0, 0));
#ifdef SHS_OFFSCREEN_SHADER_DIR
    CHECK(pixel_is(16, 12, 255, 64, 0, 255));
    CHECK(pixel_is(16, 28, 0, 0, 0, 0));
    VulkanRenderBackend backend;
    CHECK(backend.initialize_device());
    auto prepared = backend.prepare_offscreen(desc, pd);
    CHECK(prepared);
    CHECK(!backend.prepare_offscreen(desc, pd));
    const RHICmd backend_stream[] = {rhi_cmd_begin_pass({backend.offscreen_target(), 0, true, false}),
        rhi_cmd_bind_pipeline(*prepared), rhi_cmd_draw({3}), rhi_cmd_end_pass()};
    std::vector<uint8_t> backend_pixels(pixels.size(), 123);
    CHECK(backend.execute_offscreen(backend_stream, backend_pixels));
    CHECK(backend_pixels == pixels);
    CHECK(backend.execute_offscreen(backend_stream, backend_pixels));
    CHECK(backend_pixels == pixels);
    const RHICmd malformed[] = {backend_stream[0], rhi_cmd_draw({3}), rhi_cmd_end_pass()};
    auto failed = backend.execute_offscreen(malformed, backend_pixels);
    CHECK(!failed && failed.error().code == VulkanExecutionError::RecordingFailed);
    CHECK(failed.error().recording.code == VulkanRecordingError::MissingBinding);
    CHECK(failed.error().recording.command_index == 1);
    CHECK(backend_pixels == pixels);
    CHECK(!backend.execute_offscreen({}, backend_pixels));
    CHECK(!backend.execute_offscreen(backend_stream, std::span<uint8_t>(backend_pixels).first(1)));
    CHECK(backend.execute_offscreen(backend_stream, backend_pixels));
    backend.reset_offscreen();
    CHECK(!backend.execute_offscreen(backend_stream, backend_pixels));
    auto resized = desc;
    resized.width = 64;
    resized.height = 64;
    prepared = backend.prepare_offscreen(resized, pd);
    CHECK(prepared);
    const RHICmd resized_stream[] = {rhi_cmd_begin_pass({backend.offscreen_target(), 0, true, false}),
        rhi_cmd_bind_pipeline(*prepared), rhi_cmd_draw({3}), rhi_cmd_end_pass()};
    backend_pixels.resize(64 * 64 * 4);
    CHECK(backend.execute_offscreen(resized_stream, backend_pixels));
    CHECK(backend_pixels[(24 * 64 + 32) * 4] == 255);
    CHECK(backend_pixels[(24 * 64 + 32) * 4 + 1] == 64);
    CHECK(backend_pixels[0] == 0 && backend_pixels[3] == 0);
    const auto retired_target = backend.offscreen_target();
    backend.shutdown();
    CHECK(backend.resource_stats().live_images == 0);
    CHECK(backend.create_image(resized) == 0); // never return a retired cache hit
    CHECK(backend.initialize_device());
    prepared = backend.prepare_offscreen(resized, pd);
    CHECK(prepared && backend.offscreen_target() != retired_target);
    const RHICmd recreated_stream[] = {rhi_cmd_begin_pass({backend.offscreen_target(), 0, true, false}),
        rhi_cmd_bind_pipeline(*prepared), rhi_cmd_draw({3}), rhi_cmd_end_pass()};
    CHECK(backend.execute_offscreen(recreated_stream, backend_pixels));
    CHECK(!backend.execute_offscreen(resized_stream, backend_pixels));
    backend.shutdown();
    std::fprintf(stderr, "PASS: backend synchronous execution, repeated readback, rejection recovery and resize\n");
    std::fprintf(stderr, "PASS: submitted triangle, fence completion, RGBA8 known pixels\n");
#else
    CHECK(pixel_is(16, 12, 0, 0, 0, 0));
#endif
    // Completed work: discard references before explicit reset/recreate.
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
    std::fprintf(stderr, "PASS: explicit offscreen attachments, submission/readback and reset/recreation\n");
    return 0;
}
