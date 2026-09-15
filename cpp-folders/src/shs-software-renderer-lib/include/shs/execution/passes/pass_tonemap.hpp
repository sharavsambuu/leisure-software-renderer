#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: pass_tonemap.hpp
    МОДУЛЬ: passes
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн passes модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include "shs/frame/frame_params.hpp"
#include "shs/gfx/rt_handle.hpp"
#include "shs/gfx/rt_registry.hpp"
#include "shs/job/parallel_for.hpp"

#include <algorithm>
#include <cmath>

namespace shs
{
    struct Context;

    class PassTonemap
    {
    public:
        struct Inputs
        {
            const FrameParams* fp = nullptr;
            RTRegistry* rtr = nullptr;

            RTHandle rt_hdr{}; // input
            RTHandle rt_ldr{}; // output
        };

        void execute(Context& ctx, const Inputs& in)
        {
            (void)ctx;
            if (!in.fp || !in.rtr) return;
            if (!in.rt_hdr.valid() || !in.rt_ldr.valid()) return;

            auto* hdr = static_cast<RT_ColorHDR*>(in.rtr->get(in.rt_hdr));
            auto* ldr = static_cast<RT_ColorLDR*>(in.rtr->get(in.rt_ldr));
            if (!hdr || !ldr || hdr->w <= 0 || hdr->h <= 0 || ldr->w <= 0 || ldr->h <= 0) return;

            const int w = std::min(hdr->w, ldr->w);
            const int h = std::min(hdr->h, ldr->h);
            const float exposure = std::max(0.0001f, in.fp->pass.tonemap.exposure);
            const float inv_gamma = 1.0f / std::max(0.001f, in.fp->pass.tonemap.gamma);

            parallel_for_1d(ctx.job_system, 0, h, 8, [&](int yb, int ye)
            {
                for (int y = yb; y < ye; ++y)
                {
                    for (int x = 0; x < w; ++x)
                    {
                        const ColorF s = hdr->color.at(x, y);

                        // Exposure
                        float r = std::max(0.0f, s.r * exposure);
                        float g = std::max(0.0f, s.g * exposure);
                        float b = std::max(0.0f, s.b * exposure);

                        // Reinhard tone map
                        r = r / (1.0f + r);
                        g = g / (1.0f + g);
                        b = b / (1.0f + b);

                        // Gamma
                        r = std::pow(r, inv_gamma);
                        g = std::pow(g, inv_gamma);
                        b = std::pow(b, inv_gamma);

                        ldr->color.at(x, y) = Color{
                            (uint8_t)std::clamp((int)std::lround(r * 255.0f), 0, 255),
                            (uint8_t)std::clamp((int)std::lround(g * 255.0f), 0, 255),
                            (uint8_t)std::clamp((int)std::lround(b * 255.0f), 0, 255),
                            255
                        };
                    }
                }
            });
        }
    };
}
