#pragma once

/*
    SHS RENDERER SAN

    FILE: demo_input_actions.hpp
    SCOPE: exp-rendering-techniques (Run 1 / P3 task: input edge -> tokens)
    PURPOSE: Closed action-token vocabulary for the forward-classic demo's
             keydown input edge, plus the pure tuning reducer.

    Design (Domain Pod / Constitution I S7):
      - The SDL edge (`handle_event`) is the ONLY place that knows SDL_Keycode;
        it translates keysyms into `DemoInputAction` tokens.
      - Everything downstream consumes tokens. The light/shadow tuning cluster
        is a pure value reducer (`apply_demo_light_tuning`) with no SDL, no
        Vulkan, no GLM dependencies — GPU-free testable.
      - Debug/F-key tokens stay demo-side (they mutate demo-internal debug
        state); the vocabulary is closed but owned by the demo pod, so adding
        keys never touches the renderer core.
*/

#include <algorithm>
#include <cstdint>

namespace shs::demo
{
    // Closed keydown action vocabulary for the demo (edge translates SDL -> these).
    enum class DemoInputAction : uint8_t
    {
        Quit,
        ToggleMultithreadRecording,
        CycleFramebufferDebugTarget,
        CycleForwardFramebufferDebugTarget,
        ToggleGpuCuller,
        ToggleLightVolumeDebug,
        CycleSemanticDebugTarget,
        ToggleTemporalAccumulation,
        PrintHelp,
        ToggleAutoCycleTechnique,
        // Renderpath pod intents (Run 1 / P3 task 2): path/technique hot-swap
        // requests are translated to renderpath::RenderPathCommand intents.
        CycleRenderPathRecipe,
        CycleRenderingTechnique,
        // Light/shadow tuning cluster — handled by the pure reducer below.
        ToggleSunShadow,
        LightOrbitScaleDec,
        LightOrbitScaleInc,
        LightHeightBiasDec,
        LightHeightBiasInc,
        LightRangeScaleDec,
        LightRangeScaleInc,
        LightIntensityScaleDec,
        LightIntensityScaleInc,
        SunShadowStrengthDec,
        SunShadowStrengthInc,
        ResetLightControls,
        ActiveLightCountDec,
        ActiveLightCountInc,
    };

    // Pure tuning state: everything the 1..0 / R / -/+ keys touch.
    struct DemoLightTuningState
    {
        float orbit_scale = 1.0f;
        float height_bias = 0.0f;
        float range_scale = 0.72f;
        float intensity_scale = 1.0f;
        bool sun_shadow_enabled = true;
        float sun_shadow_strength = 0.42f;
        uint32_t active_light_count = 384;
    };

    // Tuning bounds/steps (previously inline in the demo's keydown switch).
    inline constexpr float kDemoOrbitScaleMin = 0.35f;
    inline constexpr float kDemoOrbitScaleMax = 2.50f;
    inline constexpr float kDemoHeightBiasMin = -3.0f;
    inline constexpr float kDemoHeightBiasMax = 6.0f;
    inline constexpr float kDemoRangeScaleMin = 0.50f;
    inline constexpr float kDemoRangeScaleMax = 2.00f;
    inline constexpr float kDemoIntensityScaleMin = 0.30f;
    inline constexpr float kDemoIntensityScaleMax = 2.50f;
    inline constexpr float kDemoSunShadowStrengthMin = 0.0f;
    inline constexpr float kDemoSunShadowStrengthMax = 1.0f;
    inline constexpr uint32_t kDemoActiveLightMin = 64u;
    inline constexpr uint32_t kDemoActiveLightMax = 768u;
    inline constexpr uint32_t kDemoActiveLightStep = 64u;

    // Pure reducer: apply one tuning token. No side effects, no I/O.
    inline void apply_demo_light_tuning(DemoLightTuningState& s, DemoInputAction a)
    {
        switch (a)
        {
            case DemoInputAction::ToggleSunShadow:
                s.sun_shadow_enabled = !s.sun_shadow_enabled;
                break;
            case DemoInputAction::LightOrbitScaleDec:
                s.orbit_scale = std::clamp(s.orbit_scale - 0.10f, kDemoOrbitScaleMin, kDemoOrbitScaleMax);
                break;
            case DemoInputAction::LightOrbitScaleInc:
                s.orbit_scale = std::clamp(s.orbit_scale + 0.10f, kDemoOrbitScaleMin, kDemoOrbitScaleMax);
                break;
            case DemoInputAction::LightHeightBiasDec:
                s.height_bias = std::clamp(s.height_bias - 0.25f, kDemoHeightBiasMin, kDemoHeightBiasMax);
                break;
            case DemoInputAction::LightHeightBiasInc:
                s.height_bias = std::clamp(s.height_bias + 0.25f, kDemoHeightBiasMin, kDemoHeightBiasMax);
                break;
            case DemoInputAction::LightRangeScaleDec:
                s.range_scale = std::clamp(s.range_scale - 0.10f, kDemoRangeScaleMin, kDemoRangeScaleMax);
                break;
            case DemoInputAction::LightRangeScaleInc:
                s.range_scale = std::clamp(s.range_scale + 0.10f, kDemoRangeScaleMin, kDemoRangeScaleMax);
                break;
            case DemoInputAction::LightIntensityScaleDec:
                s.intensity_scale = std::clamp(s.intensity_scale - 0.10f, kDemoIntensityScaleMin, kDemoIntensityScaleMax);
                break;
            case DemoInputAction::LightIntensityScaleInc:
                s.intensity_scale = std::clamp(s.intensity_scale + 0.10f, kDemoIntensityScaleMin, kDemoIntensityScaleMax);
                break;
            case DemoInputAction::SunShadowStrengthDec:
                s.sun_shadow_strength = std::clamp(
                    s.sun_shadow_strength - 0.05f, kDemoSunShadowStrengthMin, kDemoSunShadowStrengthMax);
                break;
            case DemoInputAction::SunShadowStrengthInc:
                s.sun_shadow_strength = std::clamp(
                    s.sun_shadow_strength + 0.05f, kDemoSunShadowStrengthMin, kDemoSunShadowStrengthMax);
                break;
            case DemoInputAction::ResetLightControls:
                s.orbit_scale = 1.0f;
                s.height_bias = 0.0f;
                s.range_scale = 0.72f;
                s.intensity_scale = 1.0f;
                s.sun_shadow_enabled = true;
                s.sun_shadow_strength = 0.42f;
                break;
            case DemoInputAction::ActiveLightCountDec:
                s.active_light_count =
                    (s.active_light_count > kDemoActiveLightStep)
                        ? (s.active_light_count - kDemoActiveLightStep)
                        : kDemoActiveLightMin;
                break;
            case DemoInputAction::ActiveLightCountInc:
                s.active_light_count = std::min<uint32_t>(
                    kDemoActiveLightMax, s.active_light_count + kDemoActiveLightStep);
                break;
            default:
                break; // debug/F-key tokens are not tuning actions
        }
    }
}
