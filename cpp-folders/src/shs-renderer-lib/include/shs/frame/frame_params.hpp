#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: frame_params.hpp
    МОДУЛЬ: frame
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн frame модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <cstdint>

namespace shs
{
    enum class DebugViewMode : uint8_t
    {
        Final = 0,
        Albedo = 1,
        Normal = 2,
        Depth = 3
    };

    enum class CullMode : uint8_t
    {
        None = 0,
        Back = 1,
        Front = 2
    };

    enum class ShadingModel : uint8_t
    {
        PBRMetalRough = 0,
        BlinnPhong = 1
    };

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

        // Software rasterizer debugging/correctness controls.
        DebugViewMode debug_view = DebugViewMode::Final;
        CullMode cull_mode = CullMode::Back;
        bool front_face_ccw = true;
        ShadingModel shading_model = ShadingModel::PBRMetalRough;

        // Shadow softness controls.
        float shadow_bias_const = 0.0008f;
        float shadow_bias_slope = 0.0015f;
        int shadow_pcf_radius = 2;
        float shadow_pcf_step = 1.0f;
        float shadow_strength = 1.0f;
    };
}
