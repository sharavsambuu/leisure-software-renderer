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
    struct TonemapParams
    {
        float exposure = 1.0f;
        float gamma = 2.2f;
    };

    struct ShadowPassParams
    {
        bool enable = true;
        float bias_const = 0.0008f;
        float bias_slope = 0.0015f;
        int pcf_radius = 2;
        float pcf_step = 1.0f;
        float strength = 1.0f;
    };

    struct LightShaftsPassParams
    {
        bool enable = true;
        int steps = 48;
        float density = 0.8f;
        float weight = 0.9f;
        float decay = 0.95f;
    };

    struct MotionVectorParams
    {
        bool enable = true;
    };

    struct MotionBlurPassParams
    {
        bool enable = false;
        int samples = 10;
        float strength = 1.0f;
        float max_velocity_px = 20.0f;
        float min_velocity_px = 0.25f;
        float depth_reject = 0.08f;
    };

    struct HybridPipelineParams
    {
        // true үед pass бүр өөр backend дээр ажиллахыг зөвшөөрнө.
        bool allow_cross_backend_passes = true;
        // true үед backend олдохгүй pass таарвал кадрын алдаа болгоно.
        bool strict_backend_availability = false;
        // true үед Vulkan-like queue submission/runtime-ээр pass execution-ийг дамжуулна.
        bool emulate_vulkan_runtime = true;
        // Vulkan-like submission дотор task-уудыг job system-ээр зэрэг ажиллуулах эсэх.
        bool emulate_parallel_recording = true;
        // Vulkan frame-in-flight дуурайлтын slot тоо.
        uint32_t emulated_frames_in_flight = 2;
    };

    struct PassParamBlocks
    {
        TonemapParams tonemap{};
        ShadowPassParams shadow{};
        LightShaftsPassParams light_shafts{};
        MotionVectorParams motion_vectors{};
        MotionBlurPassParams motion_blur{};
    };

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
        bool enable_motion_blur    = false;

        // Light shafts
        int   shafts_steps   = 48;
        float shafts_density = 0.8f;
        float shafts_weight  = 0.9f;
        float shafts_decay   = 0.95f;

        // Placeholder: DOF, bloom, etc нэмэхэд энд төвлөрүүлнэ
        bool enable_dof   = false;
        bool enable_bloom = false;

        // Camera + per-object motion blur
        int   motion_blur_samples = 10;
        float motion_blur_strength = 1.0f;
        float motion_blur_max_velocity_px = 20.0f;
        float motion_blur_min_velocity_px = 0.25f;
        float motion_blur_depth_reject = 0.08f;

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

        // Шинэ pass-param API: pass бүр өөрийн block-оор тохиргоо авна.
        PassParamBlocks pass{};
        HybridPipelineParams hybrid{};

        void sync_legacy_to_blocks()
        {
            pass.tonemap.exposure = exposure;
            pass.tonemap.gamma = gamma;

            pass.shadow.enable = enable_shadows;
            pass.shadow.bias_const = shadow_bias_const;
            pass.shadow.bias_slope = shadow_bias_slope;
            pass.shadow.pcf_radius = shadow_pcf_radius;
            pass.shadow.pcf_step = shadow_pcf_step;
            pass.shadow.strength = shadow_strength;

            pass.light_shafts.enable = enable_light_shafts;
            pass.light_shafts.steps = shafts_steps;
            pass.light_shafts.density = shafts_density;
            pass.light_shafts.weight = shafts_weight;
            pass.light_shafts.decay = shafts_decay;

            pass.motion_vectors.enable = enable_motion_vectors;

            pass.motion_blur.enable = enable_motion_blur;
            pass.motion_blur.samples = motion_blur_samples;
            pass.motion_blur.strength = motion_blur_strength;
            pass.motion_blur.max_velocity_px = motion_blur_max_velocity_px;
            pass.motion_blur.min_velocity_px = motion_blur_min_velocity_px;
            pass.motion_blur.depth_reject = motion_blur_depth_reject;
        }
    };
}
