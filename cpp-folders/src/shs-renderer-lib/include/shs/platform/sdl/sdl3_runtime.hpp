#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: sdl3_runtime.hpp
    МОДУЛЬ: platform (SDL3 windowing backend adapter)
    ЗОРИЛГО: IPlatformRuntime implementation over SDL3/SDL3_image, plus the
             SDL3 IVulkanWindowInterop (surface creation lives behind the
             platform seam — the RHI never sees SDL). SDL2 and SDL3 headers
             must never share a translation unit.
*/

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <SDL3_image/SDL_image.h>

#include "shs/platform/platform_runtime.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace platform
    {
    class Sdl3VulkanInterop final : public IVulkanWindowInterop
    {
    public:
        explicit Sdl3VulkanInterop(SDL_Window* window) : window_(window) {}

        const char* const* vulkan_instance_extensions(uint32_t* count) override
        {
            if (!window_ || !count) return nullptr;
            uint32_t n = 0;
            const char* const* exts = SDL_Vulkan_GetInstanceExtensions(&n);
            if (!exts || n == 0)
            {
                *count = 0;
                return nullptr;
            }
            *count = n;
            return exts; // SDL3-owned static storage; durable for the process
        }

        bool vulkan_create_surface(void* vk_instance, void** out_surface) override
        {
            if (!window_ || !out_surface) return false;
            return SDL_Vulkan_CreateSurface(
                window_,
                reinterpret_cast<VkInstance>(vk_instance),
                nullptr,
                reinterpret_cast<VkSurfaceKHR*>(out_surface));
        }

        void drawable_size_pixels(int* w, int* h) override
        {
            if (window_ && w && h) SDL_GetWindowSizeInPixels(window_, w, h);
        }

    private:
        SDL_Window* window_ = nullptr;
    };

    class Sdl3Runtime final : public IPlatformRuntime
    {
    public:
        Sdl3Runtime(const WindowDesc& win, const SurfaceDesc& surface)
        {
            if (!SDL_Init(SDL_INIT_VIDEO)) return;

            // SDL3_image removed IMG_Init/IMG_Quit — format support is
            // compiled in and initialized on demand by IMG_Load.
            window_ = SDL_CreateWindow(
                win.title.c_str(),
                win.width,
                win.height,
                0 // SDL3: windows are shown by default; SDL_WINDOW_SHOWN retired
            );
            if (!window_) return;

            // SDL3: renderer flags are gone from SDL_CreateRenderer (2-arg).
            // Accelerated is the default; vsync is set post-creation.
            renderer_ = SDL_CreateRenderer(window_, nullptr);
            if (!renderer_) return;
            SDL_SetRenderVSync(renderer_, 1);

            texture_ = SDL_CreateTexture(
                renderer_,
                SDL_PIXELFORMAT_RGBA32,
                SDL_TEXTUREACCESS_STREAMING,
                surface.width,
                surface.height
            );
            if (!texture_) return;

            vulkan_interop_ = std::make_unique<Sdl3VulkanInterop>(window_);
            valid_ = true;
        }

        ~Sdl3Runtime() override
        {
            vulkan_interop_.reset();
            if (texture_) SDL_DestroyTexture(texture_);
            if (renderer_) SDL_DestroyRenderer(renderer_);
            if (window_) SDL_DestroyWindow(window_);
            SDL_Quit();
        }

        bool valid() const override { return valid_; }

        bool pump_input(PlatformInputState& out) override
        {
            out = PlatformInputState{};

            SDL_Event e;
            while (SDL_PollEvent(&e))
            {
                if (e.type == SDL_EVENT_QUIT) out.quit = true;
                if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_ESCAPE) out.quit = true;
                if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_L) out.toggle_light_shafts = true;
                if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_B) out.toggle_bot = true;
                if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_F1) out.cycle_debug_view = true;
                if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_F2) out.cycle_cull_mode = true;
                if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_F3) out.toggle_front_face = true;
                if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_F4) out.toggle_shading_model = true;
                if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_F5) out.toggle_sky_mode = true;
                if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_F6) out.toggle_follow_camera = true;
                if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_M) out.toggle_motion_blur = true;
                if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_F12) out.save_screenshot = true;

                if (e.type == SDL_EVENT_MOUSE_MOTION)
                {
                    const bool capture_mouse =
                        right_mouse_held_ ||
                        left_mouse_held_ ||
                        (SDL_GetWindowRelativeMouseMode(window_));
                    if (capture_mouse && !ignore_next_mouse_dt_)
                    {
                        out.mouse_dx += (float)e.motion.xrel;
                        out.mouse_dy += (float)e.motion.yrel;
                    }
                }
                if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN && e.button.button == SDL_BUTTON_RIGHT)
                {
                    right_mouse_held_ = true;
                }
                if (e.type == SDL_EVENT_MOUSE_BUTTON_UP && e.button.button == SDL_BUTTON_RIGHT)
                {
                    right_mouse_held_ = false;
                    out.right_mouse_up = true;
                }
                if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN && e.button.button == SDL_BUTTON_LEFT)
                {
                    left_mouse_held_ = true;
                }
                if (e.type == SDL_EVENT_MOUSE_BUTTON_UP && e.button.button == SDL_BUTTON_LEFT)
                {
                    left_mouse_held_ = false;
                    out.left_mouse_up = true;
                }
                if (e.type == SDL_EVENT_WINDOW_FOCUS_LOST)
                {
                    right_mouse_held_ = false;
                    left_mouse_held_ = false;
                }
            }

            uint32_t ms = SDL_GetMouseState(nullptr, nullptr);
            const bool relative_mode = SDL_GetWindowRelativeMouseMode(window_);
            if ((ms & SDL_BUTTON_MASK(SDL_BUTTON_RIGHT)) != 0)
            {
                right_mouse_held_ = true;
            }
            else if (!relative_mode)
            {
                right_mouse_held_ = false;
            }
            if ((ms & SDL_BUTTON_MASK(SDL_BUTTON_LEFT)) != 0)
            {
                left_mouse_held_ = true;
            }
            else if (!relative_mode)
            {
                left_mouse_held_ = false;
            }
            out.right_mouse_down = right_mouse_held_;
            out.left_mouse_down = left_mouse_held_;

            const bool* ks = SDL_GetKeyboardState(nullptr);
            out.forward = ks[SDL_SCANCODE_W] != 0;
            out.backward = ks[SDL_SCANCODE_S] != 0;
            out.left = ks[SDL_SCANCODE_A] != 0;
            out.right = ks[SDL_SCANCODE_D] != 0;
            out.descend = ks[SDL_SCANCODE_Q] != 0;
            out.ascend = ks[SDL_SCANCODE_E] != 0;
            out.boost = ks[SDL_SCANCODE_LSHIFT] != 0;

            if (ignore_next_mouse_dt_) ignore_next_mouse_dt_ = false;

            return !out.quit;
        }

        void set_relative_mouse_mode(bool enabled) override
        {
            SDL_SetWindowRelativeMouseMode(window_, enabled);
            if (enabled) ignore_next_mouse_dt_ = true;
        }

        void set_title(const std::string& title) override
        {
            if (window_) SDL_SetWindowTitle(window_, title.c_str());
        }

        SDL_Window* window() const { return window_; }

        IVulkanWindowInterop* window_vulkan_interop() override
        {
            return vulkan_interop_.get();
        }

        void upload_rgba8(const uint8_t* src, int width, int height, int src_pitch_bytes) override
        {
            if (!texture_ || !src) return;
            void* dst = nullptr;
            int dst_pitch = 0;
            if (SDL_LockTexture(texture_, nullptr, &dst, &dst_pitch) != 0) return;

            const int rows = height;
            const int copy_bytes = width * 4;
            auto* d = static_cast<uint8_t*>(dst);
            for (int y = 0; y < rows; ++y)
            {
                std::memcpy(d + y * dst_pitch, src + y * src_pitch_bytes, (size_t)copy_bytes);
            }
            SDL_UnlockTexture(texture_);
        }

        void present() override
        {
            if (!renderer_ || !texture_) return;
            SDL_SetRenderDrawColor(renderer_, 10, 10, 14, 255);
            SDL_RenderClear(renderer_);
            SDL_RenderTexture(renderer_, texture_, nullptr, nullptr);
            SDL_RenderPresent(renderer_);
        }

    private:
        bool valid_ = false;
        SDL_Window* window_ = nullptr;
        SDL_Renderer* renderer_ = nullptr;
        SDL_Texture* texture_ = nullptr;
        std::unique_ptr<Sdl3VulkanInterop> vulkan_interop_;
        bool right_mouse_held_ = false;
        bool left_mouse_held_ = false;
        bool ignore_next_mouse_dt_ = false;
    };

    } // inline namespace platform
}
