#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: sw_backend.hpp
    МОДУЛЬ: rhi/drivers/software
    ЗОРИЛГО: Software renderer backend-ийн default хэрэгжүүлэлт.
            Одоогийн pass/RT pipeline энэ backend дээр ажиллана.
*/


#include "shs/rhi/core/backend.hpp"
#include "shs/rhi/software/sw_offscreen.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace rhi
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
            c.depth_attachment_known = true;
            c.supports_depth_attachment = true;
            return c;
        }
        void begin_frame(Context& ctx, const RenderBackendFrameInfo& frame) override { (void)ctx; (void)frame; }
        void end_frame(Context& ctx, const RenderBackendFrameInfo& frame) override { (void)ctx; (void)frame; }

        // The CPU rasterizer needs no device to open and is always available, so
        // this backend realizes the generic offscreen contract instead of
        // declining it: it is the software side of the SW/Vulkan equivalence
        // gate. Failures inside it are reported as 0 / false, never approximated.
        [[nodiscard]] IOffscreenExecution* offscreen_execution() override
        {
            return &offscreen_execution_;
        }

    private:
        SoftwareOffscreenExecution offscreen_execution_{};
    };

    } // inline namespace rhi
}
