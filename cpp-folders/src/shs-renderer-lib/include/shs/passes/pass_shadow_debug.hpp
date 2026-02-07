#pragma once
/*
    shs-renderer-lib
    Shadow debug blit (depth -> grayscale)

    ТАЙЛБАР:
    - Shadow map render болсон эсэхийг шалгах
*/

#include <algorithm>
#include <cstdint>

#include <shs/passes/rt_shadow.hpp>

namespace shs {

// must adapt this signature to color RT type.
// Provide a tiny adapter function in demo if needed.
template<class RT_Color>
inline void blit_shadow_depth_to_color(
    const RT_ShadowDepth& sm,
    RT_Color& out_color,
    float depth_min = 0.0f,
    float depth_max = 1.0f
){
    // out_color must provide: w,h and set_pixel(x,y, r,g,b,a) OR direct buffer.
    // Minimal contract: out_color.w/out_color.h and out_color.set_rgba(x,y, uint8,uint8,uint8,uint8)
    const float inv = 1.0f / std::max(1e-6f, (depth_max - depth_min));

    const int W = std::min(sm.w, out_color.w);
    const int H = std::min(sm.h, out_color.h);

    for(int y=0;y<H;y++){
        for(int x=0;x<W;x++){
            float d = sm.at(x,y);
            float g = (d - depth_min) * inv;
            g = std::clamp(g, 0.0f, 1.0f);
            uint8_t c = (uint8_t)(g * 255.0f);

            out_color.set_rgba(x,y, c,c,c, 255);
        }
    }
}

} // namespace shs

