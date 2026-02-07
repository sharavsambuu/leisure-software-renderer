// File: src/shs-renderer-lib/include/shs/frame/frame_params.hpp
#pragma once
/*
    SHS RENDERER LIB - FRAME PARAMS

    ЗОРИЛГО:
    - Нэг фрэймийн (render tick) тохиргоо/параметрүүдийг нэг дор төвлөрүүлэх
    - Pass бүр өөрт хэрэгтэй хэсгийг нь ашиглана
*/

#include <cstdint>

namespace shs
{
    struct FrameParams
    {
        int w = 0;
        int h = 0;

        float dt   = 0.0f;
        float time = 0.0f;

        // HDR -> LDR
        float exposure = 1.0f;
        float gamma    = 2.2f;

        // Debug/feature toggles
        bool enable_shadows        = true;
        bool enable_skybox         = true;
        bool enable_light_shafts   = true;
        bool enable_motion_vectors = true;

        // Light shafts
        int   shafts_steps   = 48;
        float shafts_density = 0.8f;
        float shafts_weight  = 0.9f;
        float shafts_decay   = 0.95f;

        // Placeholder: DOF, bloom, etc нэмэхэд энд төвлөрүүлнэ
        bool enable_dof   = false;
        bool enable_bloom = false;
    };
}

