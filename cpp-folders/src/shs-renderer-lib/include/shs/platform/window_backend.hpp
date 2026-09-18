#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: window_backend.hpp
    МОДУЛЬ: platform
    ЗОРИЛГО: Platform-agnostic windowing backend selection. Backends (SDL3,
             SDL2, later SFML/GLFW) plug in through IPlatformRuntime; this
             header contains zero SDK includes and dispatches at runtime via
             compiled anchor factories (src/platform_sdl3_anchor.cpp,
             src/platform_sdl2_anchor.cpp). SDL2 and SDL3 headers must never
             share a translation unit — each backend owns its own anchor.
*/

#include <memory>
#include <string>

#include "shs/platform/platform_runtime.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace platform
    {
    enum class WindowBackend
    {
        Auto, // prefer SDL3, fall back to SDL2
        Sdl3,
        Sdl2,
    };

    inline const char* window_backend_name(WindowBackend backend)
    {
        switch (backend)
        {
            case WindowBackend::Sdl3: return "SDL3";
            case WindowBackend::Sdl2: return "SDL2";
            case WindowBackend::Auto: return "auto";
        }
        return "unknown";
    }

    struct WindowBackendCreateResult
    {
        std::unique_ptr<IPlatformRuntime> runtime;
        WindowBackend backend = WindowBackend::Auto; // backend actually created
        std::string note;                            // honest fallback/failure note

        bool ok() const { return runtime != nullptr && runtime->valid(); }
    };

    // Compiled backend factories. The symbols always exist (the anchor TUs
    // compile unconditionally); they return nullptr when the backend was not
    // compiled in (SDK not discovered) or runtime creation failed.
    IPlatformRuntime* create_sdl3_runtime(const WindowDesc& win, const SurfaceDesc& surface);
    IPlatformRuntime* create_sdl2_runtime(const WindowDesc& win, const SurfaceDesc& surface);

    inline WindowBackendCreateResult create_platform_runtime(
        const WindowDesc& win,
        const SurfaceDesc& surface,
        WindowBackend requested = WindowBackend::Auto)
    {
        WindowBackendCreateResult res;
        bool sdl3_attempted = false;
        bool sdl3_failed = false;

        if (requested == WindowBackend::Sdl3 || requested == WindowBackend::Auto)
        {
            sdl3_attempted = true;
            res.runtime.reset(create_sdl3_runtime(win, surface));
            if (res.runtime)
            {
                res.backend = WindowBackend::Sdl3;
                res.note = requested == WindowBackend::Auto
                    ? "auto: SDL3 runtime created"
                    : "requested SDL3 runtime created";
                return res;
            }
            sdl3_failed = true;
        }

        if (requested == WindowBackend::Sdl2 || requested == WindowBackend::Auto)
        {
            res.runtime.reset(create_sdl2_runtime(win, surface));
            if (res.runtime)
            {
                res.backend = WindowBackend::Sdl2;
                if (requested == WindowBackend::Auto)
                {
                    res.note = sdl3_failed
                        ? "auto: SDL3 runtime creation failed; SDL2 fallback created"
                        : "auto: SDL3 backend not compiled in; SDL2 runtime created";
                }
                else
                {
                    res.note = "requested SDL2 runtime created";
                }
                return res;
            }
        }

        // Honest failure — no silent swap, no fake success.
        if (requested == WindowBackend::Auto)
        {
            res.note = sdl3_attempted
                ? "auto: no windowing backend available (SDL3 and SDL2 both unavailable or creation failed)"
                : "auto: no windowing backend available (SDL2 unavailable or creation failed)";
        }
        else
        {
            res.note = std::string("requested ") + window_backend_name(requested) +
                       " runtime but creation failed (backend not compiled in, or window/renderer init failed)";
        }
        return res;
    }

    } // inline namespace platform
}