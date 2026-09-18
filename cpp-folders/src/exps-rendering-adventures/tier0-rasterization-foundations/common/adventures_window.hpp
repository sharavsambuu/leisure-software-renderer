#pragma once

/*
    exps-rendering-adventures — tier0 common (windowed presentation).

    Optional interactive front-end for the demos, built entirely on the
    platform seam (shs/platform/window_backend.hpp — zero windowing-SDK
    includes). Dispatch is SDL3-preferred auto: Auto tries SDL3 first and
    falls back to SDL2 with an honest note; --backend= pins one.

    Default demo behaviour is unchanged (headless render + PNG out, CI-safe).
    With --window the demo additionally opens a window showing the rendered
    frame: F12 exports PNGs (numbered <stem>_export_N.png next to the demo's
    own output), Esc or the close button quits.
*/

#include <cstdio>
#include <cstring>
#include <string>

#include "adventures_frame.hpp"
#include "shs/platform/window_backend.hpp"

namespace adventures
{
    struct DemoArgs
    {
        std::string out_path;  // PNG output path (positional argument)
        bool windowed = false; // --window
        shs::platform::WindowBackend backend =
            shs::platform::WindowBackend::Auto; // --backend=auto|sdl3|sdl2
    };

    // Shared CLI: [out.png] [--window] [--backend=auto|sdl3|sdl2]
    inline DemoArgs parse_demo_args(int argc, char** argv, const char* default_png)
    {
        DemoArgs args;
        args.out_path = default_png;
        for (int i = 1; i < argc; ++i)
        {
            const std::string a = argv[i];
            if (a == "--window")
            {
                args.windowed = true;
            }
            else if (a.rfind("--backend=", 0) == 0)
            {
                const std::string v = a.substr(10);
                if (v == "sdl3") args.backend = shs::platform::WindowBackend::Sdl3;
                else if (v == "sdl2") args.backend = shs::platform::WindowBackend::Sdl2;
                else if (v != "auto")
                {
                    std::fprintf(stderr,
                                 "unknown --backend '%s' (auto|sdl3|sdl2); using auto\n",
                                 v.c_str());
                }
            }
            else
            {
                args.out_path = a; // positional PNG path (last one wins)
            }
        }
        return args;
    }

    // <stem>_export_N.png next to the demo's own output path.
    inline std::string export_png_path(const std::string& out_path, int n)
    {
        const size_t dot = out_path.rfind('.');
        const std::string stem = (dot == std::string::npos) ? out_path : out_path.substr(0, dot);
        const std::string ext = (dot == std::string::npos) ? ".png" : out_path.substr(dot);
        return stem + "_export_" + std::to_string(n) + ext;
    }

    // Blocks presenting rgba (RGBA8, top-left origin) until the window is
    // closed. F12 exports the current image as a PNG. Returns the number of
    // exported images; -1 when no windowing backend could be created (honest
    // failure — the demo's own PNG has already been written by then).
    inline int present_rgba_windowed(const uint8_t* rgba, int width, int height,
                                     const std::string& title,
                                     const std::string& export_hint_path,
                                     shs::platform::WindowBackend backend)
    {
        if (!rgba || width <= 0 || height <= 0) return -1;

        const shs::platform::WindowBackendCreateResult result =
            shs::platform::create_platform_runtime(
                shs::platform::WindowDesc{ title, width, height },
                shs::platform::SurfaceDesc{ width, height },
                backend);
        if (!result.ok())
        {
            std::fprintf(stderr, "windowed mode unavailable: %s\n", result.note.c_str());
            return -1;
        }
        std::printf("window: %s — %s\n", title.c_str(), result.note.c_str());
        std::printf("  F12 = export PNG, Esc / close = quit\n");

        int exports = 0;
        for (;;)
        {
            shs::platform::PlatformInputState input;
            result.runtime->pump_input(input);
            if (input.quit) break;

            if (input.save_screenshot)
            {
                // Route the export through the same stb writer the demos use.
                Frame shot(width, height);
                std::memcpy(shot.rgba.data(), rgba, shot.rgba.size());
                const std::string path = export_png_path(export_hint_path, exports + 1);
                if (shot.save_png(path))
                {
                    ++exports;
                    std::printf("exported %s\n", path.c_str());
                }
                else
                {
                    std::fprintf(stderr, "export failed: %s\n", path.c_str());
                }
            }

            result.runtime->upload_rgba8(rgba, width, height, width * 4);
            result.runtime->present();
        }
        return exports;
    }

    // Frame convenience overload for the _sw demos.
    inline int present_frame_windowed(const Frame& frame, const std::string& title,
                                      const std::string& export_hint_path,
                                      shs::platform::WindowBackend backend =
                                          shs::platform::WindowBackend::Auto)
    {
        return present_rgba_windowed(frame.rgba.data(), frame.width, frame.height,
                                     title, export_hint_path, backend);
    }
}
