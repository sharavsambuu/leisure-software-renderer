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
    // Context (definition in shs/app/context.hpp); forward-declared here at
    // the root namespace so it denotes the same entity while the app module
    // is still un-namespaced. Re-pointed to shs::app by the app cutover slice.
    struct Context;

// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace rhi
    {
    struct RenderBackendFrameInfo
    {
        uint64_t frame_index = 0;
        int width = 0;
        int height = 0;
    };

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
    };

    } // inline namespace rhi
}
