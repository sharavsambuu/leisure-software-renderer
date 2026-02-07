/*

    shs/passes/resource_handles.hpp

    SHARED RESOURCE HANDLES (RT-ууд, env, shadow, temp)

    ЗОРИЛГО:
    - Pass бүр өөрийн RT төрлүүдийг parameter hell болгож дамжуулахгүй.
    - Ерөнхий нэршлийн тохиролцоо: frame.hdr, frame.ldr, shadow.depth гэх мэт.
    - Demo бүр нэг л газар resource-оо үүсгээд, PassContext.resources-аар дамжуулна.

*/

#pragma once

#include <cstdint>

#include "shs/passes/rt_types.hpp"

namespace shs
{
    // ---------------------------------------------
    // RendererResources: бүх pass-уудын хамтын RT-ууд
    // ---------------------------------------------
    struct RendererResources
    {
        // --- Main frame targets ---
        // Одоо хэрэглэдэг бодит төрлүүдээрээ тааруулж солино.
        // rt_types.hpp дотор юу байна, түүнтэй таарах ёстой.

        // GBuffer / DefaultRT (өнгө + гүн + motion/velocity)
        RT_ColorDepthMotion gbuf;

        // HDR өнгө (тонемап-аас өмнө)
        RT_ColorHDR hdr;

        // LDR өнгө (тонемап + gamma дараа)
        RT_ColorLDR ldr;

        // --- Shadow ---
        RT_Depth shadow_depth;

        // --- Post temp buffers ---
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

            // Доорх ctor signature-ууд rt_types.hpp-тэй таарах ёстой.
            // Хэрэв өөр байвал энд л нэг дор засна.

            gbuf         = RT_ColorDepthMotion(w, h, zn, zf);
            hdr          = RT_ColorHDR(w, h);
            ldr          = RT_ColorLDR(w, h);

            // Shadow map хэмжээ тусдаа байж болно (жишээ нь 2048)
            shadow_depth = RT_Depth(2048, 2048, zn, zf);

            tmp_a        = RT_ColorLDR(w, h);
            tmp_b        = RT_ColorLDR(w, h);
        }
    };
} // namespace shs
