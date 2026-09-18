#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: sdl2_runtime.hpp
    МОДУЛЬ: platform (SDL2 windowing backend adapter)
    ЗОРИЛГО: IPlatformRuntime implementation over SDL2/SDL2_image, plus the
             SDL2 IVulkanWindowInterop (surface creation lives behind the
             platform seam — the RHI never sees SDL). SDL2 and SDL3 headers
             must never share a translation unit.
*/

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#if __has_include(<SDL2/SDL.h>)
#include <SDL2/SDL.h>
#else
#include <SDL.h>
#endif
#if __has_include(<SDL2/SDL_vulkan.h>)
#include <SDL2/SDL_vulkan.h>
#else
#include <SDL_vulkan.h>
#endif
#if __has_include(<SDL2_image/SDL_image.h>)
#include <SDL2_image/SDL_image.h>
#elif __has_include(<SDL2/SDL_image.h>)
#include <SDL2/SDL_image.h> // system (Debian-style) dev layout: /usr/include/SDL2
#elif __has_include(<SDL_image.h>)
#include <SDL_image.h>
#endif

#include "shs/platform/platform_runtime.hpp"

#if !defined(_WIN32)
#include <dlfcn.h> // dlopen-based SDL2 dispatch (see Sdl2Api below)
#endif

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace platform
    {
    // ------------------------------------------------------------------
    // dlopen-based SDL2 dispatch — the dual-link hijack shield.
    //
    // Demo binaries link STATIC libSDL3.a/SDL3_image side by side with
    // SHARED libSDL2-2.0.so/SDL2_image. Both export identical C symbol
    // names, and the flat ELF namespace serves every SDL_*/IMG_* call in
    // the binary from whichever definition the linker consumed first —
    // SDL3's static archive. Verified with gdb: this backend's SDL_Init
    // bound to SDL3's SDL_dynapi_procs.h inside the demo executable, so
    // SDL2 window creation received SDL3 ABI arguments and failed.
    //
    // Contract: this TU owns ZERO undefined SDL_*/IMG_* symbols. The
    // shared libraries are opened with dlopen(RTLD_LOCAL) and every call
    // goes through the table below. Linking libSDL2 remains legal (it
    // keeps the soname resolvable); nothing references it statically.
    // See sdl3_cutover_runbook.md §7.
    // ------------------------------------------------------------------
    struct Sdl2Api
    {
        decltype(&SDL_Init) Init = nullptr;
        decltype(&SDL_Quit) Quit = nullptr;
        decltype(&SDL_CreateWindow) CreateWindow = nullptr;
        decltype(&SDL_DestroyWindow) DestroyWindow = nullptr;
        decltype(&SDL_SetWindowTitle) SetWindowTitle = nullptr;
        decltype(&SDL_CreateRenderer) CreateRenderer = nullptr;
        decltype(&SDL_DestroyRenderer) DestroyRenderer = nullptr;
        decltype(&SDL_RenderSetVSync) RenderSetVSync = nullptr;
        decltype(&SDL_CreateTexture) CreateTexture = nullptr;
        decltype(&SDL_DestroyTexture) DestroyTexture = nullptr;
        decltype(&SDL_LockTexture) LockTexture = nullptr;
        decltype(&SDL_UnlockTexture) UnlockTexture = nullptr;
        decltype(&SDL_SetRenderDrawColor) SetRenderDrawColor = nullptr;
        decltype(&SDL_RenderClear) RenderClear = nullptr;
        decltype(&SDL_RenderCopy) RenderCopy = nullptr;
        decltype(&SDL_RenderPresent) RenderPresent = nullptr;
        decltype(&SDL_PollEvent) PollEvent = nullptr;
        decltype(&SDL_GetMouseState) GetMouseState = nullptr;
        decltype(&SDL_GetRelativeMouseMode) GetRelativeMouseMode = nullptr;
        decltype(&SDL_SetRelativeMouseMode) SetRelativeMouseMode = nullptr;
        decltype(&SDL_GetKeyboardState) GetKeyboardState = nullptr;
        decltype(&SDL_GetWindowSizeInPixels) GetWindowSizeInPixels = nullptr;
        decltype(&SDL_Vulkan_GetInstanceExtensions) Vulkan_GetInstanceExtensions = nullptr;
        decltype(&SDL_Vulkan_CreateSurface) Vulkan_CreateSurface = nullptr;
        decltype(&SDL_ConvertSurfaceFormat) ConvertSurfaceFormat = nullptr;
        decltype(&SDL_FreeSurface) FreeSurface = nullptr;
        decltype(&SDL_GetRGBA) GetRGBA = nullptr;
        decltype(&IMG_Init) ImgInit = nullptr;
        decltype(&IMG_Load) ImgLoad = nullptr;
        decltype(&IMG_Quit) ImgQuit = nullptr;
    };

#if defined(_WIN32)
    // No dlopen dispatch on this platform yet — the SDL2 backend honestly
    // reports unavailability (same contract as "SDK not discovered").
    inline const Sdl2Api* shs_sdl2_api(std::string* error_out = nullptr)
    {
        if (error_out) *error_out = "SDL2 dlopen dispatch not supported on this platform";
        return nullptr;
    }
#else
    template <typename Fn>
    inline Fn shs_sdl2_dlsym(void* lib, const char* name)
    {
        void* sym = dlsym(lib, name);
        Fn fn = nullptr;
        static_assert(sizeof(fn) == sizeof(sym), "dlsym result must fit a function pointer");
        std::memcpy(&fn, &sym, sizeof(fn));
        return fn;
    }

    inline Sdl2Api shs_sdl2_load_api(std::string& error)
    {
        Sdl2Api t{};
        void* lib = nullptr;
        for (const char* name : {"libSDL2-2.0.so.0", "libSDL2-2.0.so", "libSDL2-2.0.0.dylib", "SDL2"})
        {
            lib = dlopen(name, RTLD_NOW | RTLD_LOCAL);
            if (lib) break;
        }
        if (!lib)
        {
            const char* why = dlerror();
            error = std::string("dlopen(libSDL2) failed") + (why ? (std::string(": ") + why) : "");
            return t;
        }

        // SDL2_image is optional (only the texture adapter needs it).
        void* img = nullptr;
        for (const char* name : {"libSDL2_image-2.0.so.0", "libSDL2_image-2.0.so", "libSDL2_image-2.0.0.dylib", "SDL2_image"})
        {
            img = dlopen(name, RTLD_NOW | RTLD_LOCAL);
            if (img) break;
        }

#define SHS_SDL2_SYM(member, sym) \
    t.member = shs_sdl2_dlsym<decltype(&sym)>(lib, #sym)
        SHS_SDL2_SYM(Init, SDL_Init);
        SHS_SDL2_SYM(Quit, SDL_Quit);
        SHS_SDL2_SYM(CreateWindow, SDL_CreateWindow);
        SHS_SDL2_SYM(DestroyWindow, SDL_DestroyWindow);
        SHS_SDL2_SYM(SetWindowTitle, SDL_SetWindowTitle);
        SHS_SDL2_SYM(CreateRenderer, SDL_CreateRenderer);
        SHS_SDL2_SYM(DestroyRenderer, SDL_DestroyRenderer);
        SHS_SDL2_SYM(RenderSetVSync, SDL_RenderSetVSync);
        SHS_SDL2_SYM(CreateTexture, SDL_CreateTexture);
        SHS_SDL2_SYM(DestroyTexture, SDL_DestroyTexture);
        SHS_SDL2_SYM(LockTexture, SDL_LockTexture);
        SHS_SDL2_SYM(UnlockTexture, SDL_UnlockTexture);
        SHS_SDL2_SYM(SetRenderDrawColor, SDL_SetRenderDrawColor);
        SHS_SDL2_SYM(RenderClear, SDL_RenderClear);
        SHS_SDL2_SYM(RenderCopy, SDL_RenderCopy);
        SHS_SDL2_SYM(RenderPresent, SDL_RenderPresent);
        SHS_SDL2_SYM(PollEvent, SDL_PollEvent);
        SHS_SDL2_SYM(GetMouseState, SDL_GetMouseState);
        SHS_SDL2_SYM(GetRelativeMouseMode, SDL_GetRelativeMouseMode);
        SHS_SDL2_SYM(SetRelativeMouseMode, SDL_SetRelativeMouseMode);
        SHS_SDL2_SYM(GetKeyboardState, SDL_GetKeyboardState);
        SHS_SDL2_SYM(GetWindowSizeInPixels, SDL_GetWindowSizeInPixels);
        SHS_SDL2_SYM(Vulkan_GetInstanceExtensions, SDL_Vulkan_GetInstanceExtensions);
        SHS_SDL2_SYM(Vulkan_CreateSurface, SDL_Vulkan_CreateSurface);
        SHS_SDL2_SYM(ConvertSurfaceFormat, SDL_ConvertSurfaceFormat);
        SHS_SDL2_SYM(FreeSurface, SDL_FreeSurface);
        SHS_SDL2_SYM(GetRGBA, SDL_GetRGBA);
#undef SHS_SDL2_SYM
        if (img)
        {
#define SHS_SDL2_IMG_SYM(member, sym) \
    t.member = shs_sdl2_dlsym<decltype(&sym)>(img, #sym)
            SHS_SDL2_IMG_SYM(ImgInit, IMG_Init);
            SHS_SDL2_IMG_SYM(ImgLoad, IMG_Load);
            SHS_SDL2_IMG_SYM(ImgQuit, IMG_Quit);
#undef SHS_SDL2_IMG_SYM
        }

        if (!(t.Init && t.CreateWindow && t.CreateRenderer && t.PollEvent))
        {
            error = "libSDL2 loaded but required symbols missing";
            return Sdl2Api{};
        }
        return t;
    }

    inline Sdl2Api& shs_sdl2_api_table() { static Sdl2Api t{}; return t; }
    inline std::string& shs_sdl2_api_error() { static std::string s; return s; }

    // Loads once (magic static). Returns nullptr with `error_out` filled when
    // the SDL2 shared library is unavailable — the honest "backend not
    // available" path, never a silent swap.
    inline const Sdl2Api* shs_sdl2_api(std::string* error_out = nullptr)
    {
        static const bool shs_sdl2_api_attempted = []() {
            shs_sdl2_api_table() = shs_sdl2_load_api(shs_sdl2_api_error());
            return true;
        }();
        (void)shs_sdl2_api_attempted;
        if (error_out) *error_out = shs_sdl2_api_error();
        const Sdl2Api& t = shs_sdl2_api_table();
        return t.Init ? &t : nullptr;
    }
#endif // !defined(_WIN32)

    class Sdl2VulkanInterop final : public IVulkanWindowInterop
    {
    public:
        explicit Sdl2VulkanInterop(SDL_Window* window) : window_(window) {}

        const char* const* vulkan_instance_extensions(uint32_t* count) override
        {
            if (!window_ || !count) return nullptr;
            const Sdl2Api* s2 = shs_sdl2_api();
            if (!s2 || !s2->Vulkan_GetInstanceExtensions) return nullptr;
            unsigned n = 0;
            // SDL2 two-call form: pNames == nullptr fills the count only;
            // the extension list is then copied into backend-owned storage.
            if (s2->Vulkan_GetInstanceExtensions(window_, &n, nullptr) != SDL_TRUE || n == 0)
            {
                *count = 0;
                return nullptr;
            }
            vk_exts_.resize(n);
            if (s2->Vulkan_GetInstanceExtensions(window_, &n, vk_exts_.data()) != SDL_TRUE)
            {
                *count = 0;
                return nullptr;
            }
            *count = static_cast<uint32_t>(n);
            return vk_exts_.data();
        }

        bool vulkan_create_surface(void* vk_instance, void** out_surface) override
        {
            if (!window_ || !out_surface) return false;
            const Sdl2Api* s2 = shs_sdl2_api();
            if (!s2 || !s2->Vulkan_CreateSurface) return false;
            return s2->Vulkan_CreateSurface(
                window_,
                reinterpret_cast<VkInstance>(vk_instance),
                reinterpret_cast<VkSurfaceKHR*>(out_surface)) == SDL_TRUE;
        }

        void drawable_size_pixels(int* w, int* h) override
        {
            if (window_ && w && h)
            {
                if (const Sdl2Api* s2 = shs_sdl2_api())
                    if (s2->GetWindowSizeInPixels) s2->GetWindowSizeInPixels(window_, w, h);
            }
        }

    private:
        SDL_Window* window_ = nullptr;
        std::vector<const char*> vk_exts_;
    };

    class Sdl2Runtime final : public IPlatformRuntime
    {
    public:
        Sdl2Runtime(const WindowDesc& win, const SurfaceDesc& surface)
        {
            const Sdl2Api* s2 = shs_sdl2_api();
            if (!s2) return; // honest failure: SDL2 shared library not loadable

            if (s2->Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) return;

            window_ = s2->CreateWindow(
                win.title.c_str(),
                SDL_WINDOWPOS_CENTERED,
                SDL_WINDOWPOS_CENTERED,
                win.width,
                win.height,
                SDL_WINDOW_SHOWN
            );
            if (!window_) return;

            renderer_ = s2->CreateRenderer(
                window_, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
            if (!renderer_)
            {
                // honest fallback: any renderer, vsync attempted post-hoc
                renderer_ = s2->CreateRenderer(window_, -1, 0);
            }
            if (!renderer_) return;
            if (s2->RenderSetVSync) s2->RenderSetVSync(renderer_, 1); // SDL2 >= 2.0.18; ignored otherwise

            texture_ = s2->CreateTexture(
                renderer_,
                SDL_PIXELFORMAT_RGBA32,
                SDL_TEXTUREACCESS_STREAMING,
                surface.width,
                surface.height
            );
            if (!texture_) return;

            vulkan_interop_ = std::make_unique<Sdl2VulkanInterop>(window_);
            valid_ = true;
        }

        ~Sdl2Runtime() override
        {
            const Sdl2Api* s2 = shs_sdl2_api();
            if (!s2) return;
            vulkan_interop_.reset();
            if (texture_) s2->DestroyTexture(texture_);
            if (renderer_) s2->DestroyRenderer(renderer_);
            if (window_) s2->DestroyWindow(window_);
            s2->Quit();
        }

        bool valid() const override { return valid_; }

        bool pump_input(PlatformInputState& out) override
        {
            out = PlatformInputState{};
            const Sdl2Api* s2 = shs_sdl2_api();
            if (!s2) return false; // honest failure: SDL2 API unavailable

            SDL_Event e;
            while (s2->PollEvent(&e))
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
                if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_F12) out.save_screenshot = true;

                if (e.type == SDL_MOUSEMOTION)
                {
                    const bool capture_mouse =
                        right_mouse_held_ ||
                        left_mouse_held_ ||
                        (s2->GetRelativeMouseMode() == SDL_TRUE);
                    if (capture_mouse && !ignore_next_mouse_dt_)
                    {
                        out.mouse_dx += (float)e.motion.xrel;
                        out.mouse_dy += (float)e.motion.yrel;
                    }
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

            uint32_t ms = s2->GetMouseState(nullptr, nullptr);
            const bool relative_mode = s2->GetRelativeMouseMode() == SDL_TRUE;
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

            const uint8_t* ks = s2->GetKeyboardState(nullptr);
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
            if (const Sdl2Api* s2 = shs_sdl2_api())
                s2->SetRelativeMouseMode(enabled ? SDL_TRUE : SDL_FALSE);
            if (enabled) ignore_next_mouse_dt_ = true;
        }

        void set_title(const std::string& title) override
        {
            if (window_)
                if (const Sdl2Api* s2 = shs_sdl2_api())
                    s2->SetWindowTitle(window_, title.c_str());
        }

        SDL_Window* window() const { return window_; }

        IVulkanWindowInterop* window_vulkan_interop() override
        {
            return vulkan_interop_.get();
        }

        void upload_rgba8(const uint8_t* src, int width, int height, int src_pitch_bytes) override
        {
            if (!texture_ || !src) return;
            const Sdl2Api* s2 = shs_sdl2_api();
            if (!s2) return;
            void* dst = nullptr;
            int dst_pitch = 0;
            if (s2->LockTexture(texture_, nullptr, &dst, &dst_pitch) != 0) return;

            const int rows = height;
            const int copy_bytes = width * 4;
            auto* d = static_cast<uint8_t*>(dst);
            for (int y = 0; y < rows; ++y)
            {
                std::memcpy(d + y * dst_pitch, src + y * src_pitch_bytes, (size_t)copy_bytes);
            }
            s2->UnlockTexture(texture_);
        }

        void present() override
        {
            if (!renderer_ || !texture_) return;
            const Sdl2Api* s2 = shs_sdl2_api();
            if (!s2) return;
            s2->SetRenderDrawColor(renderer_, 10, 10, 14, 255);
            s2->RenderClear(renderer_);
            s2->RenderCopy(renderer_, texture_, nullptr, nullptr);
            s2->RenderPresent(renderer_);
        }

    private:
        bool valid_ = false;
        SDL_Window* window_ = nullptr;
        SDL_Renderer* renderer_ = nullptr;
        SDL_Texture* texture_ = nullptr;
        std::unique_ptr<Sdl2VulkanInterop> vulkan_interop_;
        bool right_mouse_held_ = false;
        bool left_mouse_held_ = false;
        bool ignore_next_mouse_dt_ = false;
    };

    } // inline namespace platform
}
