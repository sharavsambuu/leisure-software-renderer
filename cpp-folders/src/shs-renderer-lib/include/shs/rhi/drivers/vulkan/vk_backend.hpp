#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: vk_backend.hpp
    МОДУЛЬ: rhi/drivers/vulkan
    ЗОРИЛГО: Vulkan backend-ийн суурь класс.
            Одоогийн байдлаар lifecycle ба capability contract-уудыг хангана.
*/


#include "shs/rhi/core/backend.hpp"
#include "shs/rhi/drivers/vulkan/vk_component_notes.hpp"

namespace shs
{
    class VulkanRenderBackend final : public IRenderBackend
    {
    public:
        RenderBackendType type() const override { return RenderBackendType::Vulkan; }
        BackendCapabilities capabilities() const override
        {
            BackendCapabilities c{};
            c.queues.graphics_count = 1;
            c.queues.compute_count = 1;
            c.queues.transfer_count = 1;
            c.queues.present_count = 1;
            c.features.validation_layers = true;
            c.features.timeline_semaphore = true;
            c.features.descriptor_indexing = true;
            c.features.dynamic_rendering = true;
            c.features.push_constants = true;
            c.features.multithread_command_recording = true;
            c.features.async_compute = true;
            c.limits.max_frames_in_flight = 3;
            c.limits.max_color_attachments = 8;
            c.limits.max_descriptor_sets_per_pipeline = 8;
            c.limits.max_push_constant_bytes = 128;
            c.supports_present = true;
            c.supports_offscreen = true;
            return c;
        }

        void begin_frame(Context& ctx, const RenderBackendFrameInfo& frame) override
        {
            (void)ctx;
            (void)frame;
        }

        void end_frame(Context& ctx, const RenderBackendFrameInfo& frame) override
        {
            (void)ctx;
            (void)frame;
        }
    };
}

