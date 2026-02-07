#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: backend_factory.hpp
    МОДУЛЬ: rhi/backend
    ЗОРИЛГО: Render backend-ийг нэр/type-ээр үүсгэх helper.
            Одоогийн software pass-уудтай нийцэх баталгааг эндээс хийнэ.
*/


#include <algorithm>
#include <cctype>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "shs/rhi/drivers/opengl/gl_backend.hpp"
#include "shs/rhi/drivers/software/sw_backend.hpp"
#include "shs/rhi/drivers/vulkan/vk_backend.hpp"
#include "shs/rhi/core/backend.hpp"

namespace shs
{
    struct RenderBackendCreateResult
    {
        std::unique_ptr<IRenderBackend> backend{};
        std::vector<std::unique_ptr<IRenderBackend>> auxiliary_backends{};
        RenderBackendType requested = RenderBackendType::Software;
        RenderBackendType active = RenderBackendType::Software;
        std::string note{};
    };

    inline std::string to_lower_ascii(std::string_view s)
    {
        std::string out{};
        out.reserve(s.size());
        for (const char c : s)
        {
            out.push_back((char)std::tolower((unsigned char)c));
        }
        return out;
    }

    inline RenderBackendType parse_render_backend_type(std::string_view text, RenderBackendType fallback = RenderBackendType::Software)
    {
        const std::string v = to_lower_ascii(text);
        if (v == "software" || v == "sw" || v == "cpu") return RenderBackendType::Software;
        if (v == "opengl" || v == "gl") return RenderBackendType::OpenGL;
        if (v == "vulkan" || v == "vk") return RenderBackendType::Vulkan;
        return fallback;
    }

    inline RenderBackendCreateResult create_render_backend(RenderBackendType requested)
    {
        RenderBackendCreateResult out{};
        out.requested = requested;

        switch (requested)
        {
            case RenderBackendType::Software:
            {
                out.backend = std::make_unique<SoftwareRenderBackend>();
                out.active = RenderBackendType::Software;
                return out;
            }
            case RenderBackendType::OpenGL:
            {
                out.backend = std::make_unique<OpenGLRenderBackend>();
                out.auxiliary_backends.push_back(std::make_unique<SoftwareRenderBackend>());
                out.active = RenderBackendType::OpenGL;
                out.note = "OpenGL backend selected. Software backend is registered as hybrid fallback for unported passes.";
                return out;
            }
            case RenderBackendType::Vulkan:
            {
                out.backend = std::make_unique<VulkanRenderBackend>();
                out.auxiliary_backends.push_back(std::make_unique<SoftwareRenderBackend>());
                out.active = RenderBackendType::Vulkan;
                out.note = "Vulkan backend selected. Software backend is registered as hybrid fallback for unported passes.";
                return out;
            }
        }

        out.backend = std::make_unique<SoftwareRenderBackend>();
        out.active = RenderBackendType::Software;
        return out;
    }

    inline RenderBackendCreateResult create_render_backend(std::string_view requested_text)
    {
        return create_render_backend(parse_render_backend_type(requested_text, RenderBackendType::Software));
    }
}
