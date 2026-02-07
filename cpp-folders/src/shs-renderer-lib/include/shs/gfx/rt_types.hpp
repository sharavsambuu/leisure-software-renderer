#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: rt_types.hpp
    МОДУЛЬ: gfx
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн gfx модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <cstdint>
#include <vector>
#include <algorithm>

namespace shs
{
    struct Motion2f
    {
        float x = 0.0f;
        float y = 0.0f;
    };

    struct Color
    {
        uint8_t r, g, b, a;
    };

    struct ColorF
    {
        float r, g, b, a;
    };

    template<typename TPixel>
    struct PixelBuffer2D
    {
        int w = 0;
        int h = 0;
        std::vector<TPixel> data;

        PixelBuffer2D() = default;
        PixelBuffer2D(int W, int H, const TPixel& clear) { resize(W, H, clear); }

        void resize(int W, int H, const TPixel& clear)
        {
            w = W;
            h = H;
            data.assign((size_t)w * (size_t)h, clear);
        }

        void clear(const TPixel& clear_value)
        {
            std::fill(data.begin(), data.end(), clear_value);
        }

        TPixel& at(int x, int y) { return data[(size_t)y * (size_t)w + (size_t)x]; }
        const TPixel& at(int x, int y) const { return data[(size_t)y * (size_t)w + (size_t)x]; }
    };

    struct RT_ColorLDR
    {
        int w = 0;
        int h = 0;
        PixelBuffer2D<Color> color;

        RT_ColorLDR() = default;
        RT_ColorLDR(int W, int H, Color clear = {0, 0, 0, 255}) : w(W), h(H), color(W, H, clear) {}

        void clear(Color c = {0, 0, 0, 255}) { color.clear(c); }

        void set_rgba(int x, int y, uint8_t r, uint8_t g, uint8_t b, uint8_t a)
        {
            if (x < 0 || x >= w || y < 0 || y >= h) return;
            color.at(x, y) = Color{r, g, b, a};
        }
    };

    struct RT_ColorHDR
    {
        int w = 0;
        int h = 0;
        PixelBuffer2D<ColorF> color;

        RT_ColorHDR() = default;
        RT_ColorHDR(int W, int H, ColorF clear = {0.0f, 0.0f, 0.0f, 1.0f}) : w(W), h(H), color(W, H, clear) {}

        void clear(ColorF c = {0.0f, 0.0f, 0.0f, 1.0f}) { color.clear(c); }
    };

    struct RT_DepthBuffer
    {
        int w = 0;
        int h = 0;
        float zn = 0.1f;
        float zf = 1000.0f;
        PixelBuffer2D<float> depth;

        RT_DepthBuffer() = default;
        RT_DepthBuffer(int W, int H, float ZN = 0.1f, float ZF = 1000.0f)
            : w(W), h(H), zn(ZN), zf(ZF), depth(W, H, 1.0f)
        {}

        void clear(float d = 1.0f) { depth.clear(d); }
    };

    struct RT_ColorDepthVelocity
    {
        int   w     = 0;
        int   h     = 0;
        float zn    = 0.1f;
        float zf    = 1000.0f;
        Color clear = {0,0,0,255};
        PixelBuffer2D<Color> color;
        PixelBuffer2D<float> depth;
        PixelBuffer2D<Motion2f> motion;

        RT_ColorDepthVelocity() = default;

        RT_ColorDepthVelocity(int W, int H, float ZN, float ZF, Color Clear = {0,0,0,255})
            : w(W), h(H), zn(ZN), zf(ZF), clear(Clear), color(W, H, Clear), depth(W, H, 1.0f), motion(W, H, Motion2f{})
        {}

        void clear_all()
        {
            color.clear(clear);
            depth.clear(1.0f);
            motion.clear(Motion2f{});
        }
    };

    
    using RT_ColorDepthMotion = RT_ColorDepthVelocity;
    using DefaultRT           = RT_ColorDepthVelocity;
}
