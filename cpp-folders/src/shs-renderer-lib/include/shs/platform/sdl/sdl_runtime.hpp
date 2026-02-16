#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: sdl_runtime.hpp
    МОДУЛЬ: platform
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн platform модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <cstdint>
#include <cstring>
#include <string>

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>

#include "shs/platform/platform_runtime.hpp"

namespace shs
{
    class SdlRuntime final : public IPlatformRuntime
    {
    public:
        SdlRuntime(const WindowDesc& win, const SurfaceDesc& surface)
        {
            if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) return;

            const int img_flags = IMG_INIT_PNG | IMG_INIT_JPG;
            if ((IMG_Init(img_flags) & img_flags) == 0) return;

            window_ = SDL_CreateWindow(
                win.title.c_str(),
                SDL_WINDOWPOS_CENTERED,
                SDL_WINDOWPOS_CENTERED,
                win.width,
                win.height,
                SDL_WINDOW_SHOWN
            );
            if (!window_) return;

            renderer_ = SDL_CreateRenderer(window_, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
            if (!renderer_) return;

            texture_ = SDL_CreateTexture(
                renderer_,
                SDL_PIXELFORMAT_RGBA32,
                SDL_TEXTUREACCESS_STREAMING,
                surface.width,
                surface.height
            );
            if (!texture_) return;

            valid_ = true;
        }

        ~SdlRuntime() override
        {
            if (texture_) SDL_DestroyTexture(texture_);
            if (renderer_) SDL_DestroyRenderer(renderer_);
            if (window_) SDL_DestroyWindow(window_);
            IMG_Quit();
            SDL_Quit();
        }

        bool valid() const override { return valid_; }

        bool pump_input(PlatformInputState& out) override
        {
            out = PlatformInputState{};

            SDL_Event e;
            while (SDL_PollEvent(&e))
            {
                if (e.type == SDL_QUIT) out.quit = true;
                if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE) out.quit = true;
                if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_l) out.toggle_light_shafts = true;
                if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_b) out.toggle_bot = true;
                if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_F1) out.cycle_debug_view = true;
                if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_F2) out.cycle_cull_mode = true;
                if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_F3) out.toggle_front_face = true;
                if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_F4) out.toggle_shading_model = true;
                if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_F5) out.toggle_sky_mode = true;
                if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_F6) out.toggle_follow_camera = true;
                if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_m) out.toggle_motion_blur = true;

                if (e.type == SDL_MOUSEMOTION)
                {
                    out.mouse_dx += (float)e.motion.xrel;
                    out.mouse_dy += (float)e.motion.yrel;
                }
                if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_RIGHT)
                {
                    right_mouse_held_ = true;
                }
                if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_RIGHT)
                {
                    right_mouse_held_ = false;
                    out.right_mouse_up = true;
                }
                if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT)
                {
                    left_mouse_held_ = true;
                }
                if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_LEFT)
                {
                    left_mouse_held_ = false;
                    out.left_mouse_up = true;
                }
                if (e.type == SDL_WINDOWEVENT && e.window.event == SDL_WINDOWEVENT_FOCUS_LOST)
                {
                    right_mouse_held_ = false;
                    left_mouse_held_ = false;
                }
            }

            uint32_t ms = SDL_GetMouseState(nullptr, nullptr);
            const bool relative_mode = SDL_GetRelativeMouseMode() == SDL_TRUE;
            if ((ms & SDL_BUTTON(SDL_BUTTON_RIGHT)) != 0)
            {
                right_mouse_held_ = true;
            }
            else if (!relative_mode)
            {
                right_mouse_held_ = false;
            }
            if ((ms & SDL_BUTTON(SDL_BUTTON_LEFT)) != 0)
            {
                left_mouse_held_ = true;
            }
            else if (!relative_mode)
            {
                left_mouse_held_ = false;
            }
            out.right_mouse_down = right_mouse_held_;
            out.left_mouse_down = left_mouse_held_;

            const uint8_t* ks = SDL_GetKeyboardState(nullptr);
            out.forward = ks[SDL_SCANCODE_W] != 0;
            out.backward = ks[SDL_SCANCODE_S] != 0;
            out.left = ks[SDL_SCANCODE_A] != 0;
            out.right = ks[SDL_SCANCODE_D] != 0;
            out.descend = ks[SDL_SCANCODE_Q] != 0;
            out.ascend = ks[SDL_SCANCODE_E] != 0;
            out.boost = ks[SDL_SCANCODE_LSHIFT] != 0;
            return !out.quit;
        }

        void set_relative_mouse_mode(bool enabled) override
        {
            SDL_SetRelativeMouseMode(enabled ? SDL_TRUE : SDL_FALSE);
        }

        void set_title(const std::string& title) override
        {
            if (window_) SDL_SetWindowTitle(window_, title.c_str());
        }

        SDL_Window* window() const { return window_; }

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
            SDL_RenderCopy(renderer_, texture_, nullptr, nullptr);
            SDL_RenderPresent(renderer_);
        }

    private:
        bool valid_ = false;
        SDL_Window* window_ = nullptr;
        SDL_Renderer* renderer_ = nullptr;
        SDL_Texture* texture_ = nullptr;
        bool right_mouse_held_ = false;
        bool left_mouse_held_ = false;
    };
}
