#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: offscreen_execution.hpp
    МОДУЛЬ: rhi/core
    ЗОРИЛГО: Backend-ийн offscreen execution-ийг vendor-free contract-оор
            илэрхийлнэ. IRenderBackend::offscreen_execution() үүнийг буцаана.

            Гэрээний дүрэм:
              * null буюу 0 гэдэг нь "боломжгүй" (fallback хий эсвэл skip) —
                хэзээ ч "амжилттай" гэсэн үг биш.
              * Энэ header-т Vk* болон SDK төрөл байхгүй; GPU объект үүсгэх эрх нь
                зөвхөн backend-ийн explicit entry point-ууд.
              * Execute нь синхрон: буцах үед in-flight ажил үлдэхгүй.
*/

#include <cstdint>
#include <span>

#include "shs/rhi/command/command_desc.hpp"
#include "shs/rhi/pipeline/pipeline_desc.hpp"
#include "shs/rhi/resource/resource_desc.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace rhi
    {
    // Minimal, backend-owned offscreen surface. A backend that cannot drive a
    // self-owned offscreen target simply exposes no instance of this contract
    // (IRenderBackend::offscreen_execution() returns nullptr); callers must
    // fall back, never assume support.
    class IOffscreenExecution
    {
    public:
        virtual ~IOffscreenExecution() = default;

        // Open the backend's device for offscreen work. The factory only
        // constructs the backend; nothing is usable before this succeeds. This
        // is the portable replacement for the dynamic_cast-then-init idiom
        // consumers use today. A headless-capable opening is expected; a
        // windowed/surface opening is not part of this contract.
        // Returns false when no device is available: the caller skips, never
        // passes.
        [[nodiscard]] virtual bool initialize_device() = 0;

        // Prepare one owned color target plus its single-pass graphics pipeline.
        // Returns the backend-owned graphics-pipeline ID on success, or 0 on
        // rejection (unsupported descriptor, unavailable device). The prepared
        // color target is reported separately by offscreen_target(). A 0 return
        // is a skip signal, never a pass.
        [[nodiscard]] virtual uint64_t prepare_offscreen(
            const RHIImageDesc& target, const RHIGraphicsPipelineDesc& pipeline) = 0;

        // Target ID prepared by the last successful prepare_offscreen(); 0 when
        // nothing is prepared.
        [[nodiscard]] virtual uint64_t offscreen_target() const = 0;

        // Record, submit and read the prepared target back into |pixels|
        // (RGBA8, width * height * 4 bytes). Returns false on a rejected stream,
        // a recording/validation failure or an unavailable target. Callers must
        // treat |pixels| as valid only when this returns true.
        [[nodiscard]] virtual bool execute_offscreen(
            std::span<const RHICmd> commands, std::span<uint8_t> pixels) = 0;

        // Release the target, pipeline and readback resources. After this call
        // execute_offscreen() must fail until prepare_offscreen() succeeds again.
        virtual void reset_offscreen() = 0;
    };

    } // inline namespace rhi
}
