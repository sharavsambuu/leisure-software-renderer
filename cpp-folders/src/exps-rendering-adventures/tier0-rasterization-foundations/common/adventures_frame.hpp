#pragma once

/*
    exps-rendering-adventures — tier0 common (software side).

    Minimal RGBA8 framebuffer + PNG output shared by every *_sw demo, so the
    demos stay headless (no SDL window): they render to memory, write a PNG,
    and exit. This keeps them runnable in CI and directly comparable against
    the *_vk twins, which render offscreen and write the same format.

    Origin is top-left (PNG order). Straight-alpha blending with the same
    SRC_ALPHA / ONE_MINUS_SRC_ALPHA factors the _vk pipelines pin.
*/

#include <cstdint>
#include <string>
#include <vector>

namespace adventures
{
    struct Frame
    {
        int width  = 0;
        int height = 0;
        std::vector<uint8_t> rgba{};

        Frame() = default;
        Frame(int w, int h) { resize(w, h); }

        void resize(int w, int h)
        {
            width  = w;
            height = h;
            rgba.assign(size_t(w) * size_t(h) * 4u, 0u);
        }

        void clear(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255u)
        {
            for (size_t i = 0; i < rgba.size(); i += 4u)
            {
                rgba[i + 0] = r;
                rgba[i + 1] = g;
                rgba[i + 2] = b;
                rgba[i + 3] = a;
            }
        }

        void put(int x, int y, uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255u)
        {
            if (x < 0 || y < 0 || x >= width || y >= height) return;
            const size_t idx = (size_t(y) * size_t(width) + size_t(x)) * 4u;
            rgba[idx + 0] = r;
            rgba[idx + 1] = g;
            rgba[idx + 2] = b;
            rgba[idx + 3] = a;
        }

        void get(int x, int y, float& r, float& g, float& b, float& a) const
        {
            const size_t idx = (size_t(y) * size_t(width) + size_t(x)) * 4u;
            r = float(rgba[idx + 0]) / 255.0f;
            g = float(rgba[idx + 1]) / 255.0f;
            b = float(rgba[idx + 2]) / 255.0f;
            a = float(rgba[idx + 3]) / 255.0f;
        }

        // stb_image_write-backed; implemented in adventures_stb.cpp (single TU).
        bool save_png(const std::string& path) const;
    };

    // Straight-alpha "source over": result = src * a + dst * (1 - a), 0..1 floats.
    inline void blend_pixel(Frame& frame, int x, int y, float r, float g, float b, float a)
    {
        if (x < 0 || y < 0 || x >= frame.width || y >= frame.height) return;
        float dr, dg, db, da;
        frame.get(x, y, dr, dg, db, da);
        const float out_a = a + da * (1.0f - a);
        if (out_a <= 0.0f) { frame.put(x, y, 0, 0, 0, 0); return; }
        const float out_r = (r * a + dr * da * (1.0f - a)) / out_a;
        const float out_g = (g * a + dg * da * (1.0f - a)) / out_a;
        const float out_b = (b * a + db * da * (1.0f - a)) / out_a;
        frame.put(x, y,
                  uint8_t(out_r * 255.0f + 0.5f),
                  uint8_t(out_g * 255.0f + 0.5f),
                  uint8_t(out_b * 255.0f + 0.5f),
                  uint8_t(out_a * 255.0f + 0.5f));
    }

    inline void put_pixel(Frame& frame, int x, int y, float r, float g, float b)
    {
        frame.put(x, y,
                  uint8_t(r * 255.0f + 0.5f),
                  uint8_t(g * 255.0f + 0.5f),
                  uint8_t(b * 255.0f + 0.5f));
    }
}
