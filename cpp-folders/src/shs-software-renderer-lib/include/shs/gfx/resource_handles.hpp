/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: resource_handles.hpp
    МОДУЛЬ: gfx
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн gfx модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#pragma once

#include <cstdint>

#include "shs/gfx/rt_types.hpp"

namespace shs
{
    // ---------------------------------------------
    // RendererResources: бүх pass-уудын хамтын RT-ууд
    // ---------------------------------------------
    struct RendererResources
    {
        // --- Main frame targets ---
        // rt_types.hpp дээрх RT төрлүүдийг ашиглана.

        // GBuffer / DefaultRT (өнгө + гүн + motion/velocity)
        RT_ColorDepthMotion gbuf;

        // HDR өнгө (тонемап-аас өмнө)
        RT_ColorHDR hdr;

        // LDR өнгө (тонемап + gamma дараа)
        RT_ColorLDR ldr;

        // --- Shadow ---
        RT_DepthBuffer shadow_depth;

        // --- Post buffers ---
        RT_ColorLDR tmp_a;
        RT_ColorLDR tmp_b;

        // --- Sizing / init helpers ---
        int   w  = 0;
        int   h  = 0;
        float zn = 0.1f;
        float zf = 1000.0f;

        // Нэг удаа дууддаг init (header-only байлгах зорилготой)
        inline void init(int width, int height, float znear, float zfar)
        {
            w  = width;
            h  = height;
            zn = znear;
            zf = zfar;

            gbuf         = RT_ColorDepthMotion(w, h, zn, zf);
            hdr          = RT_ColorHDR(w, h);
            ldr          = RT_ColorLDR(w, h);

            // Shadow map хэмжээ тусдаа байж болно (жишээ нь 2048)
            shadow_depth = RT_DepthBuffer(2048, 2048, zn, zf);

            tmp_a        = RT_ColorLDR(w, h);
            tmp_b        = RT_ColorLDR(w, h);
        }
    };
} // namespace shs
