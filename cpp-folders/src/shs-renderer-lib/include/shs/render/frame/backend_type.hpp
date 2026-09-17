#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: backend_type.hpp
    МОДУЛЬ: render/frame
    ЗОРИЛГО: Render backend-ийн төрлийн хувьсагч (value-tier enum).
            renderpath/planning болон rhi хоёулаа ашигладаг тул
            OK түвшний (value) саармаг байрлалд байна — planning → rhi
            cycle-ийг эвдэхийн тулд rhi/core/backend.hpp-ээс гаргаж ирсэн.
*/

#include <cstdint>

namespace shs
{
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
}
