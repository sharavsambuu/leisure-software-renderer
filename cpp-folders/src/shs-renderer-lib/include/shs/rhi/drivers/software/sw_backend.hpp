#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: sw_backend.hpp
    МОДУЛЬ: rhi/drivers/software
    ЗОРИЛГО: Software renderer backend-ийн default хэрэгжүүлэлт.
            Одоогийн pass/RT pipeline энэ backend дээр ажиллана.
*/


#include "shs/rhi/core/backend.hpp"

namespace shs
{
    class SoftwareRenderBackend final : public IRenderBackend
    {
    public:
        RenderBackendType type() const override { return RenderBackendType::Software; }
        BackendCapabilities capabilities() const override
        {
            BackendCapabilities c{};
            c.queues.graphics_count = 1;
            c.queues.compute_count = 1;
            c.queues.transfer_count = 1;
            c.queues.present_count = 1;
            c.features.validation_layers = false;
            c.features.multithread_command_recording = true;
            c.limits.max_frames_in_flight = 2;
            c.limits.max_color_attachments = 1;
            c.supports_present = true;
            c.supports_offscreen = true;
            return c;
        }
        void begin_frame(Context& ctx, const RenderBackendFrameInfo& frame) override { (void)ctx; (void)frame; }
        void end_frame(Context& ctx, const RenderBackendFrameInfo& frame) override { (void)ctx; (void)frame; }
    };
}
