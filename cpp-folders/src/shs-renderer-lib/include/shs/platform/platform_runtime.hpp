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
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace platform
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

    // Platform-agnostic Vulkan window interop (R4 platform-seam ruling): the
    // RHI backend must not include any windowing SDK. A windowing backend
    // (SDL2, SDL3, future SFML/GLFW) exposes surface creation, required
    // instance extensions and the drawable pixel size through this seam.
    class IVulkanWindowInterop
    {
    public:
        virtual ~IVulkanWindowInterop() = default;

        // Returns a pointer to backend-owned static/durable storage of
        // required Vulkan instance extensions (nullptr on failure).
        virtual const char* const* vulkan_instance_extensions(uint32_t* count) = 0;

        // Creates a Vulkan surface for the backend window.
        // vk_instance: VkInstance (type-erased to keep this header SDK-free),
        // out_surface: receives VkSurfaceKHR (type-erased).
        virtual bool vulkan_create_surface(void* vk_instance, void** out_surface) = 0;

        // Window-backbuffer size in pixels (physical size, may differ from
        // logical size on HiDPI).
        virtual void drawable_size_pixels(int* w, int* h) = 0;
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

        // Vulkan interop for this runtime's window; nullptr when the backend
        // does not support Vulkan window integration. Defaulted (not pure)
        // so headless/other backends are not forced to implement it.
        virtual IVulkanWindowInterop* window_vulkan_interop() { return nullptr; }
    };

    } // inline namespace platform
}

