#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: gl_backend.hpp
    МОДУЛЬ: rhi/drivers/opengl
    ЗОРИЛГО: OpenGL backend-ийн суурь класс.
            Одоогоор lifecycle hook-уудыг л хангаж, pass executor-уудыг дараагийн шатанд нэмнэ.
*/


#include "shs/rhi/core/backend.hpp"

namespace shs
{
    class OpenGLRenderBackend final : public IRenderBackend
    {
    public:
        RenderBackendType type() const override { return RenderBackendType::OpenGL; }
        BackendCapabilities capabilities() const override
        {
            BackendCapabilities c{};
            c.queues.graphics_count = 1;
            c.queues.compute_count = 1;
            c.queues.transfer_count = 1;
            c.queues.present_count = 1;
            c.features.validation_layers = false;
            c.features.push_constants = false;
            c.features.dynamic_rendering = true;
            c.features.multithread_command_recording = false;
            c.limits.max_frames_in_flight = 2;
            c.limits.max_color_attachments = 8;
            c.supports_present = true;
            c.supports_offscreen = true;
            return c;
        }
        void begin_frame(Context& ctx, const RenderBackendFrameInfo& frame) override { (void)ctx; (void)frame; }
        void end_frame(Context& ctx, const RenderBackendFrameInfo& frame) override { (void)ctx; (void)frame; }
    };
}
