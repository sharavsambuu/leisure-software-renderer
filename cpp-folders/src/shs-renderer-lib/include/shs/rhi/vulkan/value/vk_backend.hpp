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
#include "shs/rhi/core/backend.hpp"
#include "shs/rhi/core/capabilities.hpp"
#include "shs/rhi/vulkan/value/vk_device.hpp"
#include "shs/rhi/vulkan/value/vk_resources.hpp"
#include "shs/rhi/vulkan/value/vk_pipelines.hpp"
#include "shs/rhi/vulkan/value/vk_commands.hpp"
#include "shs/rhi/vulkan/value/vk_sync.hpp"
#include "shs/rhi/vulkan/value/vk_submit.hpp"
#include "shs/rhi/vulkan/value/vk_readback.hpp"
#include "shs/rhi/vulkan/value/vk_offscreen.hpp"
#include "shs/rhi/vulkan/value/vk_offscreen_pipeline.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace rhi
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
                              const VulkanImagePool& images,
                              const VulkanPipelineCache* pipelines = nullptr,
                              const VulkanOffscreenPass* offscreen = nullptr,
                              const VulkanOffscreenPipeline* graphics = nullptr)
            : device_(device), cmd_(cmd), buffers_(buffers), images_(images), pipelines_(pipelines),
              offscreen_(offscreen), graphics_(graphics) {}

        // Pure resource/capability checks, also usable without a device in tests.
        [[nodiscard]] std::expected<void, VulkanRecordingFailure> validate_command(
            const RHICmd& cmd, bool inside_pass) const
        {
            const auto fail = [](VulkanRecordingError code, uint64_t id = 0) {
                return std::unexpected(VulkanRecordingFailure{code, SIZE_MAX,
                    VulkanRecordingStage::Validation, VulkanCommandKind::None, id});
            };
            if (const auto* d = std::get_if<RHICmdBeginPassDesc>(&cmd.payload))
            {
                for (uint64_t id : {d->color_target, d->depth_target})
                {
                    if (!id) continue; // absent attachment, not an invalid reference
                    const auto* image = images_.find(id);
                    if (!image || *image == VK_NULL_HANDLE) return fail(VulkanRecordingError::MissingImage, id);
                }
                if (offscreen_ && offscreen_->device() == device_ && offscreen_->accepts(*d)) return {};
                return fail(VulkanRecordingError::UnsupportedCommand);
            }
            if (std::holds_alternative<RHICmdEndPassDesc>(cmd.payload) && offscreen_ && inside_pass)
                return {};
            if (const auto* d = std::get_if<RHICmdBindPipelineDesc>(&cmd.payload))
            {
                const bool found = pipelines_ && (inside_pass ? pipelines_->find_graphics(d->pipeline) :
                    pipelines_->find_compute(d->pipeline));
                if (!found) return fail(VulkanRecordingError::MissingPipeline, d->pipeline);
                // A cache record alone is not a realized VkPipeline.
                if (inside_pass && graphics_ && offscreen_ && offscreen_->device() == device_ &&
                    graphics_->accepts(*pipelines_->find_graphics(d->pipeline), *offscreen_)) return {};
                return fail(VulkanRecordingError::UnsupportedCommand, d->pipeline);
            }
            if (std::holds_alternative<RHICmdDrawDesc>(cmd.payload) ||
                std::holds_alternative<RHICmdDrawIndexedDesc>(cmd.payload))
            {
                if (inside_pass && graphics_ && graphics_->pipeline()) return {};
                return fail(VulkanRecordingError::UnsupportedCommand);
            }
            uint64_t buffer = 0;
            if (const auto* d = std::get_if<RHICmdBindVertexBufferDesc>(&cmd.payload)) buffer = d->buffer;
            else if (const auto* d = std::get_if<RHICmdBindIndexBufferDesc>(&cmd.payload)) buffer = d->buffer;
            else if (std::holds_alternative<RHICmdBarrierDesc>(cmd.payload)) return {};
            else return fail(VulkanRecordingError::UnsupportedCommand);
            const auto* handle = buffers_.find(buffer);
            if (!buffer || !handle || *handle == VK_NULL_HANDLE) return fail(VulkanRecordingError::MissingBuffer, buffer);
            return {};
        }

        std::expected<void, VulkanRecordingError> begin_pass(const RHICmdBeginPassDesc& d)
        {
            if (!offscreen_) return std::unexpected(VulkanRecordingError::UnsupportedCommand);
            if (auto ready = recording_ready(); !ready) return ready;
            if (inside_pass_) return std::unexpected(VulkanRecordingError::InvalidRecordingOrder);
            if (auto valid = validate_command(rhi_cmd_begin_pass(d), false); !valid)
                return std::unexpected(valid.error().code);
            VkClearValue clear{}; // transparent black; no configurable clear value in the RHI yet
            VkRenderPassBeginInfo info{};
            info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            info.renderPass = offscreen_->render_pass();
            info.framebuffer = offscreen_->framebuffer();
            info.renderArea.extent = offscreen_->extent();
            info.clearValueCount = 1;
            info.pClearValues = &clear;
            vkCmdBeginRenderPass(cmd_, &info, VK_SUBPASS_CONTENTS_INLINE);
            inside_pass_ = true;
            bound_pipeline_ = 0;
            bound_index_ = false;
            return {};
        }

        std::expected<void, VulkanRecordingError> end_pass(const RHICmdEndPassDesc&)
        {
            if (!offscreen_) return std::unexpected(VulkanRecordingError::UnsupportedCommand);
            if (auto ready = recording_ready(); !ready) return ready;
            if (!inside_pass_) return std::unexpected(VulkanRecordingError::InvalidRecordingOrder);
            vkCmdEndRenderPass(cmd_);
            inside_pass_ = false;
            bound_pipeline_ = 0;
            return {};
        }

        std::expected<void, VulkanRecordingError> bind_pipeline(const RHICmdBindPipelineDesc& d)
        {
            if (!graphics_) return std::unexpected(VulkanRecordingError::UnsupportedCommand);
            if (auto ready = recording_ready(); !ready) return ready;
            if (!inside_pass_) return std::unexpected(VulkanRecordingError::InvalidRecordingOrder);
            if (auto valid = validate_command(rhi_cmd_bind_pipeline(d.pipeline), true); !valid)
                return std::unexpected(valid.error().code);
            vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_->pipeline());
            bound_pipeline_ = d.pipeline;
            return {};
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
                bound_index_ = true;
                return {};
            }
            return std::unexpected(VulkanRecordingError::MissingBuffer);
        }

        std::expected<void, VulkanRecordingError> draw(const RHICmdDrawDesc& d)
        {
            if (auto ready = recording_ready(); !ready) return ready;
            if (!inside_pass_) return std::unexpected(VulkanRecordingError::InvalidRecordingOrder);
            if (!bound_pipeline_) return std::unexpected(VulkanRecordingError::MissingBinding);
            if (auto valid = validate_command(rhi_cmd_bind_pipeline(bound_pipeline_), true); !valid)
                return std::unexpected(valid.error().code);
            vkCmdDraw(cmd_, d.vertex_count, d.instance_count, d.first_vertex, d.first_instance);
            return {};
        }

        std::expected<void, VulkanRecordingError> draw_indexed(const RHICmdDrawIndexedDesc& d)
        {
            if (!graphics_) return std::unexpected(VulkanRecordingError::UnsupportedCommand);
            if (auto ready = recording_ready(); !ready) return ready;
            if (!inside_pass_) return std::unexpected(VulkanRecordingError::InvalidRecordingOrder);
            if (!bound_pipeline_ || !bound_index_) return std::unexpected(VulkanRecordingError::MissingBinding);
            vkCmdDrawIndexed(cmd_, d.index_count, d.instance_count, d.first_index, d.vertex_offset, d.first_instance);
            return {};
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

        [[nodiscard]] std::expected<void, VulkanRecordingError> recording_ready() const
        {
            if (device_ == VK_NULL_HANDLE)
                return std::unexpected(VulkanRecordingError::DeviceUnavailable);
            if (cmd_ == VK_NULL_HANDLE)
                return std::unexpected(VulkanRecordingError::CommandBufferUnavailable);
            return {};
        }

    private:
        VkDevice device_;
        VkCommandBuffer cmd_;
        const VulkanBufferPool& buffers_;
        const VulkanImagePool& images_;
        const VulkanPipelineCache* pipelines_;
        const VulkanOffscreenPass* offscreen_;
        const VulkanOffscreenPipeline* graphics_;
        bool inside_pass_ = false;
        bool bound_index_ = false;
        uint64_t bound_pipeline_ = 0;
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

        void shutdown() { reset_offscreen(); destroy_gpu_objects(); device_.shutdown(); }

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

        // CPU-visible coherent upload, or a staging→device-local copy for
        // GPUOnly buffers carrying TransferDst usage. Caller retires any
        // borrowed submissions before updating.
        [[nodiscard]] std::expected<void, VulkanExecutionFailure> upload_buffer(
            uint64_t id, std::span<const uint8_t> bytes)
        {
            const auto* desc = buffer_descs_.find(id);
            const auto* memory = buffer_memory_.find(id);
            if (!device_ready() || !desc || !memory ||
                (desc->memory != RHIMemoryClass::CPUVisible && desc->memory != RHIMemoryClass::GPUOnly) ||
                bytes.size() != desc->size_bytes || bytes.empty())
                return std::unexpected(VulkanExecutionFailure{VulkanExecutionError::InvalidDescriptor});
            std::vector<uint8_t> shadow(bytes.begin(), bytes.end());
            if (desc->memory == RHIMemoryClass::GPUOnly)
            {
                // vkCmdCopyBuffer demands TRANSFER_DST; demand it up front so the
                // rejection is typed and the recording stays validation-clean.
                if (!(desc->usage & RHIBufferUsage_TransferDst))
                    return std::unexpected(VulkanExecutionFailure{VulkanExecutionError::InvalidDescriptor});
                const auto* gpu_buffer = buffers_.find(id);
                if (!gpu_buffer)
                    return std::unexpected(VulkanExecutionFailure{VulkanExecutionError::InvalidDescriptor});
                const auto copied = vulkan_buffer_upload_sync(device_.device(), device_.physical(),
                    device_.graphics_queue(), device_.info().graphics_queue_family, *gpu_buffer, 0, bytes);
                if (!copied)
                    return std::unexpected(VulkanExecutionFailure{VulkanExecutionError::UploadFailed, copied.error()});
            }
            else
            {
                void* mapped = nullptr;
                const auto result = vkMapMemory(device_.device(), *memory, 0, bytes.size(), 0, &mapped);
                if (result != VK_SUCCESS)
                    return std::unexpected(VulkanExecutionFailure{VulkanExecutionError::UploadFailed, result});
                std::memcpy(mapped, bytes.data(), bytes.size());
                vkUnmapMemory(device_.device(), *memory);
            }
            uploaded_.insert_or_assign(id, std::move(shadow));
            return {};
        }

        // Explicit single-target preparation. Reset before resize; callers must
        // retire borrowed command buffers before reset/shutdown. Synchronous
        // execute_offscreen itself never leaves successful work in flight.
        [[nodiscard]] std::expected<uint64_t, VulkanExecutionFailure> prepare_offscreen(
            const RHIImageDesc& image, const RHIGraphicsPipelineDesc& pipeline)
        {
            if (!device_ready() || offscreen_.framebuffer())
                return std::unexpected(VulkanExecutionFailure{VulkanExecutionError::NotPrepared});
            if (!VulkanOffscreenPass::supports(image) || !VulkanOffscreenPipeline::supports(pipeline) ||
                !(image.usage & RHIImageUsage_TransferSrc))
                return std::unexpected(VulkanExecutionFailure{VulkanExecutionError::InvalidDescriptor});
            offscreen_target_ = create_image(image);
            if (!offscreen_target_ || !offscreen_.initialize(device_.device(), offscreen_target_,
                    *images_.find(offscreen_target_), image))
            {
                reset_offscreen();
                return std::unexpected(VulkanExecutionFailure{VulkanExecutionError::CreationFailed});
            }
            const auto id = pipelines_.intern_graphics(pipeline, [&](uint64_t key, const auto& desc) {
                return graphics_.initialize(key, desc, offscreen_);
            });
            if (!id || (!graphics_.pipeline() && !graphics_.initialize(id, pipeline, offscreen_)) ||
                !readback_.initialize(device_, offscreen_.extent()))
            {
                reset_offscreen();
                return std::unexpected(VulkanExecutionFailure{VulkanExecutionError::CreationFailed});
            }
            vertex_layout_ = pipeline.vertex_layout;
            return id;
        }

        void reset_offscreen()
        {
            readback_.reset();
            graphics_.reset();
            offscreen_.reset();
            offscreen_target_ = 0;
        }
        [[nodiscard]] uint64_t offscreen_target() const { return offscreen_target_; }

        [[nodiscard]] std::expected<void, VulkanExecutionFailure> execute_offscreen(
            std::span<const RHICmd> stream, std::span<uint8_t> rgba)
        {
            if (!offscreen_.framebuffer())
                return std::unexpected(VulkanExecutionFailure{VulkanExecutionError::NotPrepared});
            if (stream.empty() || !std::holds_alternative<RHICmdBeginPassDesc>(stream.front().payload) ||
                !std::holds_alternative<RHICmdEndPassDesc>(stream.back().payload))
                return std::unexpected(VulkanExecutionFailure{VulkanExecutionError::InvalidDescriptor});
            if (auto order = validate_command_order(stream); !order)
                return std::unexpected(VulkanExecutionFailure{VulkanExecutionError::RecordingFailed, VK_SUCCESS, order.error()});
            if (auto geometry = validate_geometry(stream); !geometry)
                return std::unexpected(VulkanExecutionFailure{VulkanExecutionError::RecordingFailed, VK_SUCCESS, geometry.error()});
            return readback_.execute(*images_.find(offscreen_target_), rgba, [&](VkCommandBuffer command) {
                VulkanCommandRecorder recorder{device_.device(), command, buffers_, images_,
                    &pipelines_, &offscreen_, &graphics_};
                return record_commands(stream, recorder);
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
            VulkanCommandRecorder recorder{device_.device(), command_buffer_, buffers_, images_, &pipelines_};
            return record_commands(stream, recorder);
        }

        // Borrowed handle: caller owns lifetime, vkBegin/vkEndCommandBuffer,
        // queue compatibility and external synchronization. A non-null handle
        // alone cannot establish Vulkan recording state or device ownership.
        void set_command_buffer(VkCommandBuffer cmd) { command_buffer_ = cmd; }

        // Explicit submission, separate from frame-slot bookkeeping. The caller
        // must have ended recording on the borrowed command buffer.
        [[nodiscard]] std::expected<void, VulkanSubmitFailure> submit_commands_sync()
        {
            return vulkan_submit_sync(device_.device(), device_.graphics_queue(), command_buffer_);
        }

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
        [[nodiscard]] std::expected<void, VulkanRecordingFailure> validate_geometry(std::span<const RHICmd> stream) const
        {
            RHICmdBindVertexBufferDesc vertex{};
            RHICmdBindIndexBufferDesc index{};
            for (size_t i = 0; i < stream.size(); ++i)
            {
                const auto fail = [&](VulkanRecordingError code, uint64_t id = 0) {
                    return std::unexpected(VulkanRecordingFailure{code, i, VulkanRecordingStage::Validation,
                        vulkan_command_kind(stream[i]), id});
                };
                const auto& p = stream[i].payload;
                if (std::holds_alternative<RHICmdBeginPassDesc>(p)) { vertex = {}; index = {}; }
                if (const auto* d = std::get_if<RHICmdBindVertexBufferDesc>(&p)) vertex = *d;
                if (const auto* d = std::get_if<RHICmdBindIndexBufferDesc>(&p)) index = *d;
                if (const auto* d = std::get_if<RHICmdBindVertexBufferDesc>(&p))
                {
                    const auto* desc = buffer_descs_.find(d->buffer);
                    if (!desc) return fail(VulkanRecordingError::MissingBuffer, d->buffer);
                    if (!(desc->usage & RHIBufferUsage_Vertex) || d->offset >= desc->size_bytes || d->offset % 4)
                        return fail(VulkanRecordingError::InvalidCommand, d->buffer);
                }
                if (const auto* d = std::get_if<RHICmdBindIndexBufferDesc>(&p))
                {
                    const auto* desc = buffer_descs_.find(d->buffer);
                    if (!desc) return fail(VulkanRecordingError::MissingBuffer, d->buffer);
                    if (!(desc->usage & RHIBufferUsage_Index) || d->offset >= desc->size_bytes)
                        return fail(VulkanRecordingError::InvalidCommand, d->buffer);
                }
                const auto* draw = std::get_if<RHICmdDrawDesc>(&p);
                const auto* indexed = std::get_if<RHICmdDrawIndexedDesc>(&p);
                if (!draw && !indexed) continue;
                const auto* vertices = uploaded_.find(vertex.buffer);
                if (vertex_layout_ == RHIVertexLayout::Position2F && !vertices)
                    return fail(VulkanRecordingError::MissingBinding, vertex.buffer);
                const uint64_t count = vertices && vertex.offset <= vertices->size() ? (vertices->size() - vertex.offset) / 8 : 0;
                if (draw && vertex_layout_ == RHIVertexLayout::Position2F &&
                    uint64_t(draw->first_vertex) + draw->vertex_count > count)
                    return fail(VulkanRecordingError::InvalidCommand, vertex.buffer);
                if (indexed)
                {
                    const auto* indices = uploaded_.find(index.buffer);
                    if (!indices) return fail(VulkanRecordingError::MissingBinding, index.buffer);
                    const uint64_t stride = index.index_u32 ? 4 : 2;
                    const uint64_t end = index.offset + (uint64_t(indexed->first_index) + indexed->index_count) * stride;
                    if (end > indices->size()) return fail(VulkanRecordingError::InvalidCommand, index.buffer);
                    if (vertex_layout_ == RHIVertexLayout::Position2F)
                    {
                        for (uint64_t n = 0; n < indexed->index_count; ++n)
                        {
                            uint32_t value = 0;
                            const auto* source = indices->data() + index.offset + (indexed->first_index + n) * stride;
                            if (index.index_u32) std::memcpy(&value, source, 4);
                            else { uint16_t small; std::memcpy(&small, source, 2); value = small; }
                            const int64_t effective = int64_t(value) + indexed->vertex_offset;
                            if (effective < 0 || uint64_t(effective) >= count)
                                return fail(VulkanRecordingError::InvalidCommand, vertex.buffer);
                        }
                    }
                }
            }
            return {};
        }

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
            buffer_descs_.insert_or_assign(id, d);
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
            buffers_.clear(); images_.clear();
            buffer_memory_.clear(); image_memory_.clear();
            resources_.clear(); pipelines_.clear();
            buffer_descs_.clear(); uploaded_.clear();
            live_buffer_ids_.clear();
            live_image_ids_.clear();
            command_buffer_ = VK_NULL_HANDLE;
        }

        VulkanDeviceManager device_;
        VulkanResourceRegistry resources_;
        VulkanPipelineCache pipelines_;
        VulkanFrameSync frame_sync_;
        VulkanOffscreenPass offscreen_;
        VulkanOffscreenPipeline graphics_;
        VulkanReadback readback_;
        uint64_t offscreen_target_ = 0;
        RHIVertexLayout vertex_layout_ = RHIVertexLayout::Procedural;
        containers::FlatMap<uint64_t, RHIBufferDesc> buffer_descs_{std::pmr::get_default_resource()};
        containers::FlatMap<uint64_t, std::vector<uint8_t>> uploaded_{std::pmr::get_default_resource()};
        VulkanBufferPool buffers_{std::pmr::get_default_resource()};
        VulkanImagePool images_{std::pmr::get_default_resource()};
        VulkanMemoryPool buffer_memory_{std::pmr::get_default_resource()};
        VulkanMemoryPool image_memory_{std::pmr::get_default_resource()};
        std::pmr::vector<uint64_t> live_buffer_ids_{};
        std::pmr::vector<uint64_t> live_image_ids_{};
        VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
        uint32_t frame_sync_slots_ = 2;
    };

    } // inline namespace rhi
}
