#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: backend.hpp
    МОДУЛЬ: rhi/core
    ЗОРИЛГО: Render backend-ийн ерөнхий интерфэйс.
            Одоогоор software backend ашиглана, цаашид OpenGL/Vulkan нэмэх суурь болно.
*/


#include <cstdint>
#include "shs/render/frame/backend_type.hpp"
#include "shs/rhi/core/capabilities.hpp"

namespace shs
{
    // The context passed through the backend interface is the app-owned
    // Context (definition in shs/app/context.hpp); forward-declared in
    // shs::app (re-pointed by the app cutover slice, step 7).
    namespace app
    {
    struct Context;
    }

    // Root spelling compatibility (step 7): shs::Context denotes the
    // app-owned shs::app::Context.
    using app::Context;

// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace rhi
    {
    struct RenderBackendFrameInfo
    {
        uint64_t frame_index = 0;
        int width = 0;
        int height = 0;
    };

    // Forward declaration only: this contract header stays free of desc-tier
    // includes. The full offscreen contract lives in
    // shs/rhi/core/offscreen_execution.hpp — include that header before calling
    // through the pointer returned by offscreen_execution().
    class IOffscreenExecution;

    class IRenderBackend
    {
    public:
        virtual ~IRenderBackend() = default;

        virtual RenderBackendType type() const = 0;
        virtual const char* name() const { return render_backend_type_name(type()); }
        virtual BackendCapabilities capabilities() const { return BackendCapabilities{}; }

        virtual void on_resize(Context& ctx, int w, int h) { (void)ctx; (void)w; (void)h; }
        virtual void begin_frame(Context& ctx, const RenderBackendFrameInfo& frame) = 0;
        virtual void end_frame(Context& ctx, const RenderBackendFrameInfo& frame) = 0;

        // Optional, backend-owned offscreen execution surface. The default is
        // nullptr: a backend without a self-owned offscreen path declines.
        // Null means "fall back" (or skip), never "passed"; the returned
        // contract is defined in shs/rhi/core/offscreen_execution.hpp.
        [[nodiscard]] virtual IOffscreenExecution* offscreen_execution() { return nullptr; }
    };

    } // inline namespace rhi
}
