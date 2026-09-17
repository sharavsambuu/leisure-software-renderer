#pragma once

/*
    SHS RENDERER SAN

    FILE: vk_backend.hpp
    MODULE: rhi/drivers/vulkan
    PURPOSE: Vulkan backend — the IRenderBackend implementation the factory
            (backend_factory.hpp) has referenced aspirationally since P0.5.
            Pod-first construction per arch doc §4: this class owns the driver
            pods (device / resources / pipelines / commands / sync) and exposes
            only value descs + stable IDs upward. No Vk* type appears in the
            public API; GPU object creation happens only through the explicit
            event entry points (create_buffer/create_image over the plan),
            never lazily.
*/

#include <cstdint>
#include <expected>
#include <memory_resource>

#include <vulkan/vulkan.h>

#include "shs/containers/flat_map.hpp"
#include "shs/execution/rhi/core/backend.hpp"
#include "shs/execution/rhi/core/capabilities.hpp"
#include "shs/execution/rhi/drivers/vulkan/vk_device.hpp"
#include "shs/execution/rhi/drivers/vulkan/vk_resources.hpp"
#include "shs/execution/rhi/drivers/vulkan/vk_pipelines.hpp"
#include "shs/execution/rhi/drivers/vulkan/vk_commands.hpp"
#include "shs/execution/rhi/drivers/vulkan/vk_sync.hpp"

namespace shs
{
    using VulkanBufferPool = containers::FlatMap<uint64_t, VkBuffer>;
    using VulkanImagePool = containers::FlatMap<uint64_t, VkImage>;
    using VulkanMemoryPool = containers::FlatMap<uint64_t, VkDeviceMemory>;

    // Device-bound command sink: fulfills the record_commands sink concept by
    // translating stable IDs → Vk* handles and issuing vkCmd* calls.
    class VulkanCommandRecorder
    {
    public:
        VulkanCommandRecorder(VkDevice device, VkCommandBuffer cmd,
                              const VulkanBufferPool& buffers,
                              const VulkanImagePool& images)
            : device_(device), cmd_(cmd), buffers_(buffers), images_(images) {}

        std::expected<void, VulkanRecordingError> begin_pass(const RHICmdBeginPassDesc& d)
        {
            (void)d;
            // Renderpass dynamic rendering comes with the resource-plan wiring;
            // recording contract established here (P2).
            return std::unexpected(VulkanRecordingError::UnsupportedCommand);
        }

        std::expected<void, VulkanRecordingError> end_pass(const RHICmdEndPassDesc&)
        {
            return std::unexpected(VulkanRecordingError::UnsupportedCommand);
        }

        std::expected<void, VulkanRecordingError> bind_pipeline(const RHICmdBindPipelineDesc& d)
        {
            (void)d;
            return std::unexpected(VulkanRecordingError::UnsupportedCommand);
        }

        std::expected<void, VulkanRecordingError> bind_vertex_buffer(const RHICmdBindVertexBufferDesc& d)
        {
            if (auto ready = recording_ready(); !ready) return ready;
            if (const VkBuffer* b = buffers_.find(d.buffer); b && *b != VK_NULL_HANDLE)
            {
                const VkDeviceSize offset = (VkDeviceSize)d.offset;
                vkCmdBindVertexBuffers(cmd_, 0, 1, b, &offset);
                return {};
            }
            return std::unexpected(VulkanRecordingError::MissingBuffer);
        }

        std::expected<void, VulkanRecordingError> bind_index_buffer(const RHICmdBindIndexBufferDesc& d)
        {
            if (auto ready = recording_ready(); !ready) return ready;
            if (const VkBuffer* b = buffers_.find(d.buffer); b && *b != VK_NULL_HANDLE)
            {
                vkCmdBindIndexBuffer(cmd_, *b, (VkDeviceSize)d.offset,
                                     d.index_u32 ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16);
                return {};
            }
            return std::unexpected(VulkanRecordingError::MissingBuffer);
        }

        std::expected<void, VulkanRecordingError> draw_indexed(const RHICmdDrawIndexedDesc& d)
        {
            (void)d;
            // Pipeline binding must exist before draws are legal (G2).
            return std::unexpected(VulkanRecordingError::UnsupportedCommand);
        }

        std::expected<void, VulkanRecordingError> dispatch(const RHICmdDispatchDesc& d)
        {
            (void)d;
            // Compute pipeline binding must exist before dispatch is legal (G2).
            return std::unexpected(VulkanRecordingError::UnsupportedCommand);
        }

        std::expected<void, VulkanRecordingError> barrier(const RHICmdBarrierDesc& d)
        {
            if (auto ready = recording_ready(); !ready) return ready;
            VkMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier.srcAccessMask = vk_access_of(d.memory.src_access);
            barrier.dstAccessMask = vk_access_of(d.memory.dst_access);
            vkCmdPipelineBarrier(cmd_, vk_stage_of(d.memory.src_stage), vk_stage_of(d.memory.dst_stage),
                                 0, 1, &barrier, 0, nullptr, 0, nullptr);
            return {};
        }

    private:
        std::expected<void, VulkanRecordingError> recording_ready() const
        {
            if (device_ == VK_NULL_HANDLE)
                return std::unexpected(VulkanRecordingError::DeviceUnavailable);
            if (cmd_ == VK_NULL_HANDLE)
                return std::unexpected(VulkanRecordingError::CommandBufferUnavailable);
            return {};
        }

        VkDevice device_;
        VkCommandBuffer cmd_;
        const VulkanBufferPool& buffers_;
        const VulkanImagePool& images_;
    };

    // ------------------------------------------------------------------
    // The backend (factory-facing; default-constructible per backend_factory).
    // ------------------------------------------------------------------

    class VulkanRenderBackend final : public IRenderBackend
    {
    public:
        VulkanRenderBackend() = default;
        ~VulkanRenderBackend() override { shutdown(); }

        VulkanRenderBackend(const VulkanRenderBackend&) = delete;
        VulkanRenderBackend& operator=(const VulkanRenderBackend&) = delete;

        [[nodiscard]] RenderBackendType type() const override { return RenderBackendType::Vulkan; }

        [[nodiscard]] BackendCapabilities capabilities() const override
        {
            BackendCapabilities c{};
            c.queues.graphics_count = 1;
            c.queues.compute_count = 1;
            c.queues.transfer_count = 1;
            c.queues.present_count = 1;
            c.features.validation_layers = false;
            c.features.multithread_command_recording = false;
            c.limits.max_frames_in_flight = frame_sync_slots_;
            c.limits.max_color_attachments = 1;
            c.supports_present = false;   // offscreen pod; WSI joins via a later edge
            c.supports_offscreen = true;
            c.depth_attachment_known = true;
            c.supports_depth_attachment = true;
            return c;
        }

        // Explicit, once-only device bootstrap. Returns false when no Vulkan
        // loader/ICD exists. Headless bookkeeping remains usable, but command
        // recording reports DeviceUnavailable rather than silently succeeding.
        [[nodiscard]] bool initialize_device(const VulkanDeviceDesc& desc = {})
        {
            if (device_.device_available()) return true;
            return device_.initialize(desc);
        }

        void shutdown() { destroy_gpu_objects(); device_.shutdown(); }

        // ---- event-driven GPU object creation (arch §4 rule 3) ------------
        // Called only from PATH_COMPILED / resource-plan handling. Identical
        // descs dedupe through the hash-keyed registry (explicit cache).

        [[nodiscard]] uint64_t create_buffer(const RHIBufferDesc& d)
        {
            return resources_.intern_buffer(d, [this](uint64_t id, const RHIBufferDesc& bd) {
                return create_buffer_gpu(id, bd);
            });
        }

        [[nodiscard]] uint64_t create_image(const RHIImageDesc& d)
        {
            return resources_.intern_image(d, [this](uint64_t id, const RHIImageDesc& im) {
                return create_image_gpu(id, im);
            });
        }

        // ---- CommandDesc stream recording (arch §4 rule 2) ----------------

        [[nodiscard]] std::expected<void, VulkanRecordingFailure> record_frame_commands(
            std::span<const RHICmd> stream)
        {
            if (!device_.device_available())
                return std::unexpected(VulkanRecordingFailure{VulkanRecordingError::DeviceUnavailable});
            if (command_buffer_ == VK_NULL_HANDLE)
                return std::unexpected(VulkanRecordingFailure{VulkanRecordingError::CommandBufferUnavailable});
            VulkanCommandRecorder recorder{device_.device(), command_buffer_, buffers_, images_};
            return record_commands(stream, recorder);
        }

        void set_command_buffer(VkCommandBuffer cmd) { command_buffer_ = cmd; }

        // ---- IRenderBackend ------------------------------------------------

        void begin_frame(Context& ctx, const RenderBackendFrameInfo& frame) override
        {
            (void)ctx;
            frame_sync_.begin_frame(frame.frame_index);
        }

        void end_frame(Context& ctx, const RenderBackendFrameInfo& frame) override
        {
            (void)ctx;
            frame_sync_.end_frame(frame.frame_index);
        }

        [[nodiscard]] bool device_ready() const { return device_.device_available(); }
        [[nodiscard]] const VulkanDeviceInfo& device_info() const { return device_.info(); }
        [[nodiscard]] const VulkanResourceStats& resource_stats() const { return resources_.stats(); }
        [[nodiscard]] const VulkanPipelineStats& pipeline_stats() const { return pipelines_.stats(); }
        [[nodiscard]] const VulkanFrameSyncStats& frame_sync_stats() const { return frame_sync_.stats(); }

        void configure_frame_sync(uint32_t frames_in_flight)
        {
            frame_sync_slots_ = frames_in_flight == 0 ? 1 : frames_in_flight;
            frame_sync_.configure(frame_sync_slots_);
        }

    private:
        [[nodiscard]] bool create_buffer_gpu(uint64_t id, const RHIBufferDesc& d)
        {
            if (!device_.device_available()) return false;
            const VkBufferCreateInfo ci = vk_buffer_create_info(d);
            VkBuffer buffer = VK_NULL_HANDLE;
            if (vkCreateBuffer(device_.device(), &ci, nullptr, &buffer) != VK_SUCCESS) return false;

            VkMemoryRequirements reqs{};
            vkGetBufferMemoryRequirements(device_.device(), buffer, &reqs);
            VkPhysicalDeviceMemoryProperties mem_props{};
            vkGetPhysicalDeviceMemoryProperties(device_.physical(), &mem_props);
            const uint32_t type = vk_pick_memory_type(mem_props, reqs.memoryTypeBits, vk_memory_props_of(d.memory));
            if (type == UINT32_MAX) { vkDestroyBuffer(device_.device(), buffer, nullptr); return false; }

            VkMemoryAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            ai.allocationSize = reqs.size;
            ai.memoryTypeIndex = type;
            VkDeviceMemory memory = VK_NULL_HANDLE;
            if (vkAllocateMemory(device_.device(), &ai, nullptr, &memory) != VK_SUCCESS)
            {
                vkDestroyBuffer(device_.device(), buffer, nullptr);
                return false;
            }
            if (vkBindBufferMemory(device_.device(), buffer, memory, 0) != VK_SUCCESS)
            {
                vkFreeMemory(device_.device(), memory, nullptr);
                vkDestroyBuffer(device_.device(), buffer, nullptr);
                return false;
            }
            buffers_.insert_or_assign(id, buffer);
            buffer_memory_.insert_or_assign(id, memory);
            live_buffer_ids_.push_back(id);
            return true;
        }

        [[nodiscard]] bool create_image_gpu(uint64_t id, const RHIImageDesc& d)
        {
            if (!device_.device_available()) return false;
            const VkImageCreateInfo ci = vk_image_create_info(d);
            VkImage image = VK_NULL_HANDLE;
            if (vkCreateImage(device_.device(), &ci, nullptr, &image) != VK_SUCCESS) return false;

            VkMemoryRequirements reqs{};
            vkGetImageMemoryRequirements(device_.device(), image, &reqs);
            VkPhysicalDeviceMemoryProperties mem_props{};
            vkGetPhysicalDeviceMemoryProperties(device_.physical(), &mem_props);
            const uint32_t type = vk_pick_memory_type(mem_props, reqs.memoryTypeBits, vk_memory_props_of(d.memory));
            if (type == UINT32_MAX) { vkDestroyImage(device_.device(), image, nullptr); return false; }

            VkMemoryAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            ai.allocationSize = reqs.size;
            ai.memoryTypeIndex = type;
            VkDeviceMemory memory = VK_NULL_HANDLE;
            if (vkAllocateMemory(device_.device(), &ai, nullptr, &memory) != VK_SUCCESS)
            {
                vkDestroyImage(device_.device(), image, nullptr);
                return false;
            }
            if (vkBindImageMemory(device_.device(), image, memory, 0) != VK_SUCCESS)
            {
                vkFreeMemory(device_.device(), memory, nullptr);
                vkDestroyImage(device_.device(), image, nullptr);
                return false;
            }
            images_.insert_or_assign(id, image);
            image_memory_.insert_or_assign(id, memory);
            live_image_ids_.push_back(id);
            return true;
        }

        void destroy_gpu_objects()
        {
            if (device_.device_available())
            {
                for (const uint64_t id : live_buffer_ids_)
                {
                    if (VkBuffer* b = buffers_.find(id)) vkDestroyBuffer(device_.device(), *b, nullptr);
                    if (VkDeviceMemory* m = buffer_memory_.find(id)) vkFreeMemory(device_.device(), *m, nullptr);
                }
                for (const uint64_t id : live_image_ids_)
                {
                    if (VkImage* i = images_.find(id)) vkDestroyImage(device_.device(), *i, nullptr);
                    if (VkDeviceMemory* m = image_memory_.find(id)) vkFreeMemory(device_.device(), *m, nullptr);
                }
            }
            live_buffer_ids_.clear();
            live_image_ids_.clear();
            command_buffer_ = VK_NULL_HANDLE;
        }

        VulkanDeviceManager device_;
        VulkanResourceRegistry resources_;
        VulkanPipelineCache pipelines_;
        VulkanFrameSync frame_sync_;
        VulkanBufferPool buffers_{std::pmr::get_default_resource()};
        VulkanImagePool images_{std::pmr::get_default_resource()};
        VulkanMemoryPool buffer_memory_{std::pmr::get_default_resource()};
        VulkanMemoryPool image_memory_{std::pmr::get_default_resource()};
        std::pmr::vector<uint64_t> live_buffer_ids_{};
        std::pmr::vector<uint64_t> live_image_ids_{};
        VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
        uint32_t frame_sync_slots_ = 2;
    };
}
