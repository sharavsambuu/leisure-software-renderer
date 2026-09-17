#include <cstdio>
#include "shs/execution/rhi/drivers/vulkan/vk_backend.hpp"

// Real handle/prerequisite coverage only: no submission and no pixel evidence.
// Keep ICD-dependent checks separate from the GPU-free driver suite.
int main()
{
    using namespace shs;
    using E = VulkanRecordingError;
    VulkanRenderBackend backend;
    if (!backend.initialize_device())
    {
        std::fprintf(stderr, "SKIP: Vulkan device initialization unavailable\n");
        return 77;
    }
    const RHICmd command[] = {rhi_cmd_barrier({})};
    for (auto stream : {std::span<const RHICmd>{}, std::span<const RHICmd>{command}})
    {
        const auto result = backend.record_frame_commands(stream);
        if (result || result.error().code != E::CommandBufferUnavailable ||
            result.error().command_index != SIZE_MAX ||
            result.error().stage != VulkanRecordingStage::Prerequisite ||
            result.error().command != VulkanCommandKind::None) return 1;
    }
    backend.shutdown();

    VulkanDeviceManager device;
    if (!device.initialize({})) return 1; // device was available above; do not mask a failure
    VulkanBufferPool buffers{std::pmr::get_default_resource()};
    VulkanImagePool images{std::pmr::get_default_resource()};
    VulkanCommandRecorder unavailable{device.device(), VK_NULL_HANDLE, buffers, images};
    const std::expected<void, E> missing_cmd[] = {
        unavailable.bind_vertex_buffer({123, 0}), unavailable.bind_index_buffer({123, 0, false}),
        unavailable.barrier({}), unavailable.recording_ready()
    };
    for (const auto& result : missing_cmd)
        if (result || result.error() != E::CommandBufferUnavailable) return 1;
    const auto empty = record_commands({}, unavailable);
    if (empty || empty.error().code != E::CommandBufferUnavailable ||
        empty.error().stage != VulkanRecordingStage::Prerequisite ||
        empty.error().command_index != SIZE_MAX) return 1;

    // RAII pool is destroyed before its device, including on assertion failure.
    struct Pool
    {
        VkDevice device;
        VkCommandPool handle = VK_NULL_HANDLE;
        ~Pool() { if (handle) vkDestroyCommandPool(device, handle, nullptr); }
    } pool{device.device()};
    VkCommandPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.queueFamilyIndex = device.info().graphics_queue_family;
    if (vkCreateCommandPool(device.device(), &ci, nullptr, &pool.handle) != VK_SUCCESS) return 1;
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = pool.handle;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device.device(), &ai, &cmd) != VK_SUCCESS) return 1;
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) return 1;
    VulkanCommandRecorder recorder{device.device(), cmd, buffers, images};
    buffers.insert_or_assign(52, VK_NULL_HANDLE);
    for (uint64_t id : {uint64_t{0}, uint64_t{51}, uint64_t{52}})
    {
        const auto vertex = recorder.bind_vertex_buffer({id, 0});
        const auto index = recorder.bind_index_buffer({id, 0, true});
        if (vertex || vertex.error() != E::MissingBuffer || index || index.error() != E::MissingBuffer) return 1;
    }
    const RHICmd rejected[] = {rhi_cmd_barrier({}), rhi_cmd_bind_vertex_buffer(52, 0)};
    const auto result = record_commands(rejected, recorder);
    if (result || result.error().code != E::MissingBuffer || result.error().command_index != 1 ||
        result.error().resource_id != 52 || result.error().stage != VulkanRecordingStage::Validation) return 1;
    if (!record_commands({}, recorder)) return 1;
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) return 1;
    std::fprintf(stderr, "PASS: real device/command-buffer prerequisites and missing/null buffer rejection (no submission)\n");
    return 0;
}
