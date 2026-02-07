#pragma once

#include <cstdint>

namespace shs
{
    struct Color
    {
        uint8_t r, g, b, a;
    };

    struct RT_ColorDepthVelocity
    {
        int   w     = 0;
        int   h     = 0;
        float zn    = 0.1f;
        float zf    = 1000.0f;
        Color clear = {0,0,0,255};

        RT_ColorDepthVelocity() = default;

        RT_ColorDepthVelocity(int W, int H, float ZN, float ZF, Color Clear = {0,0,0,255})
            : w(W), h(H), zn(ZN), zf(ZF), clear(Clear)
        {}
    };

    
    using RT_ColorDepthMotion = RT_ColorDepthVelocity;
    using DefaultRT           = RT_ColorDepthVelocity;
}
