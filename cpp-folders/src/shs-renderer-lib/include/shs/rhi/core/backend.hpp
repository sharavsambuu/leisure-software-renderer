#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: backend.hpp
    МОДУЛЬ: rhi/core
    ЗОРИЛГО: Render backend-ийн ерөнхий интерфэйс.
            Одоогоор software backend ашиглана, цаашид OpenGL/Vulkan нэмэх суурь болно.
*/


#include <cstdint>
#include "shs/rhi/core/capabilities.hpp"

namespace shs
{
    struct Context;

    enum class RenderBackendType : uint8_t
    {
        Software = 0,
        OpenGL = 1,
        Vulkan = 2
    };

    inline const char* render_backend_type_name(RenderBackendType type)
    {
        switch (type)
        {
            case RenderBackendType::Software: return "software";
            case RenderBackendType::OpenGL: return "opengl";
            case RenderBackendType::Vulkan: return "vulkan";
        }
        return "unknown";
    }

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
}
