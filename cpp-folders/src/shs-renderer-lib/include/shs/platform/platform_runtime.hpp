#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: platform_runtime.hpp
    МОДУЛЬ: platform
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн platform модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <cstdint>
#include <string>

#include "shs/platform/platform_input.hpp"

namespace shs
{
    struct WindowDesc
    {
        std::string title{};
        int width = 1280;
        int height = 720;
    };

    struct SurfaceDesc
    {
        int width = 800;
        int height = 600;
    };

    class IPlatformRuntime
    {
    public:
        virtual ~IPlatformRuntime() = default;

        virtual bool valid() const = 0;
        virtual bool pump_input(PlatformInputState& out) = 0;
        virtual void set_relative_mouse_mode(bool enabled) = 0;
        virtual void set_title(const std::string& title) = 0;
        virtual void upload_rgba8(const uint8_t* src, int width, int height, int src_pitch_bytes) = 0;
        virtual void present() = 0;
    };
}

