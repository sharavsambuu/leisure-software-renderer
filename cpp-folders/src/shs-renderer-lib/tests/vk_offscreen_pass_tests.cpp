#include <cstdio>
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
    CHECK(vkEndCommandBuffer(cmd) == VK_SUCCESS);
    // No submitted work: discard references before exercising explicit reset/recreate.
    CHECK(vkResetCommandPool(vk, objects.pool, 0) == VK_SUCCESS);
    pass.reset();
    pass.reset();
    CHECK(!pass.framebuffer());
    CHECK(pass.initialize(vk, 1, objects.image, desc));
    std::fprintf(stderr, "PASS: explicit offscreen attachments and value-stream begin/end recording (no submission)\n");
    return 0;
}
