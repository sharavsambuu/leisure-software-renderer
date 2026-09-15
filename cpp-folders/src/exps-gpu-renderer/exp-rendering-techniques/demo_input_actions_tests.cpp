#include <cstdint>
#include <cstdio>

#include "demo_input_actions.hpp"

// Run 1 (P3) GPU-free tests: the demo input token vocabulary and the pure
// light/shadow tuning reducer (shs_demo_input_actions_tests). No SDL, no
// Vulkan, no GLM — the token layer stays testable on machines with no GPU
// (Run 2 DoD).
namespace
{
    using shs::demo::DemoInputAction;
    using shs::demo::DemoLightTuningState;

    bool test_defaults_match_demo()
    {
        const DemoLightTuningState d{};
        if (d.orbit_scale != 1.0f) return false;
        if (d.height_bias != 0.0f) return false;
        if (d.range_scale != 0.72f) return false;
        if (d.intensity_scale != 1.0f) return false;
        if (!d.sun_shadow_enabled) return false;
        if (d.sun_shadow_strength != 0.42f) return false;
        return d.active_light_count == 384u;
    }

    bool test_orbit_scale_clamped()
    {
        DemoLightTuningState s{};
        for (int i = 0; i < 100; ++i)
            shs::demo::apply_demo_light_tuning(s, DemoInputAction::LightOrbitScaleInc);
        if (s.orbit_scale != shs::demo::kDemoOrbitScaleMax) return false;
        for (int i = 0; i < 200; ++i)
            shs::demo::apply_demo_light_tuning(s, DemoInputAction::LightOrbitScaleDec);
        return s.orbit_scale == shs::demo::kDemoOrbitScaleMin;
    }

    bool test_height_and_range_clamped()
    {
        DemoLightTuningState s{};
        for (int i = 0; i < 100; ++i)
            shs::demo::apply_demo_light_tuning(s, DemoInputAction::LightHeightBiasInc);
        if (s.height_bias != shs::demo::kDemoHeightBiasMax) return false;
        for (int i = 0; i < 100; ++i)
            shs::demo::apply_demo_light_tuning(s, DemoInputAction::LightRangeScaleDec);
        if (s.range_scale != shs::demo::kDemoRangeScaleMin) return false;
        for (int i = 0; i < 100; ++i)
            shs::demo::apply_demo_light_tuning(s, DemoInputAction::LightRangeScaleInc);
        return s.range_scale == shs::demo::kDemoRangeScaleMax;
    }

    bool test_strength_and_intensity_clamped()
    {
        DemoLightTuningState s{};
        for (int i = 0; i < 100; ++i)
            shs::demo::apply_demo_light_tuning(s, DemoInputAction::SunShadowStrengthInc);
        if (s.sun_shadow_strength != shs::demo::kDemoSunShadowStrengthMax) return false;
        for (int i = 0; i < 100; ++i)
            shs::demo::apply_demo_light_tuning(s, DemoInputAction::SunShadowStrengthDec);
        if (s.sun_shadow_strength != shs::demo::kDemoSunShadowStrengthMin) return false;
        for (int i = 0; i < 100; ++i)
            shs::demo::apply_demo_light_tuning(s, DemoInputAction::LightIntensityScaleInc);
        return s.intensity_scale == shs::demo::kDemoIntensityScaleMax;
    }

    bool test_sun_shadow_toggle()
    {
        DemoLightTuningState s{};
        if (!s.sun_shadow_enabled) return false;
        shs::demo::apply_demo_light_tuning(s, DemoInputAction::ToggleSunShadow);
        if (s.sun_shadow_enabled) return false;
        shs::demo::apply_demo_light_tuning(s, DemoInputAction::ToggleSunShadow);
        return s.sun_shadow_enabled;
    }

    bool test_light_count_floor_and_ceiling()
    {
        DemoLightTuningState s{};
        s.active_light_count = 128u;
        shs::demo::apply_demo_light_tuning(s, DemoInputAction::ActiveLightCountDec);
        if (s.active_light_count != 64u) return false;
        shs::demo::apply_demo_light_tuning(s, DemoInputAction::ActiveLightCountDec);
        if (s.active_light_count != 64u) return false; // floor
        for (int i = 0; i < 100; ++i)
            shs::demo::apply_demo_light_tuning(s, DemoInputAction::ActiveLightCountInc);
        return s.active_light_count == shs::demo::kDemoActiveLightMax; // 768 ceiling
    }

    bool test_reset_restores_defaults()
    {
        DemoLightTuningState s{};
        shs::demo::apply_demo_light_tuning(s, DemoInputAction::LightOrbitScaleInc);
        shs::demo::apply_demo_light_tuning(s, DemoInputAction::SunShadowStrengthInc);
        shs::demo::apply_demo_light_tuning(s, DemoInputAction::ToggleSunShadow);
        shs::demo::apply_demo_light_tuning(s, DemoInputAction::ResetLightControls);
        if (s.orbit_scale != 1.0f) return false;
        if (s.height_bias != 0.0f) return false;
        if (s.range_scale != 0.72f) return false;
        if (s.intensity_scale != 1.0f) return false;
        if (!s.sun_shadow_enabled) return false;
        return s.sun_shadow_strength == 0.42f;
    }

    bool test_debug_tokens_are_not_tuning()
    {
        // F-key/debug tokens must pass through the reducer untouched.
        DemoLightTuningState s{};
        const DemoLightTuningState before = s;
        shs::demo::apply_demo_light_tuning(s, DemoInputAction::Quit);
        shs::demo::apply_demo_light_tuning(s, DemoInputAction::ToggleMultithreadRecording);
        shs::demo::apply_demo_light_tuning(s, DemoInputAction::CycleFramebufferDebugTarget);
        shs::demo::apply_demo_light_tuning(s, DemoInputAction::ToggleGpuCuller);
        shs::demo::apply_demo_light_tuning(s, DemoInputAction::PrintHelp);
        shs::demo::apply_demo_light_tuning(s, DemoInputAction::ToggleAutoCycleTechnique);
        return s.orbit_scale == before.orbit_scale &&
               s.height_bias == before.height_bias &&
               s.range_scale == before.range_scale &&
               s.intensity_scale == before.intensity_scale &&
               s.sun_shadow_enabled == before.sun_shadow_enabled &&
               s.sun_shadow_strength == before.sun_shadow_strength &&
               s.active_light_count == before.active_light_count;
    }
}

int main()
{
    struct Named { const char* name; bool (*fn)(); };
    const Named tests[] = {
        {"defaults_match_demo", test_defaults_match_demo},
        {"orbit_scale_clamped", test_orbit_scale_clamped},
        {"height_and_range_clamped", test_height_and_range_clamped},
        {"strength_and_intensity_clamped", test_strength_and_intensity_clamped},
        {"sun_shadow_toggle", test_sun_shadow_toggle},
        {"light_count_floor_and_ceiling", test_light_count_floor_and_ceiling},
        {"reset_restores_defaults", test_reset_restores_defaults},
        {"debug_tokens_are_not_tuning", test_debug_tokens_are_not_tuning},
    };

    bool ok = true;
    for (const Named& t : tests)
    {
        const bool passed = t.fn();
        std::printf("[%s] %s\n", passed ? "PASS" : "FAIL", t.name);
        ok = ok && passed;
    }
    if (!ok)
    {
        std::printf("shs_demo_input_actions_tests: FAILED\n");
        return 1;
    }
    std::printf("shs_demo_input_actions_tests: all %zu checks passed\n", sizeof(tests) / sizeof(tests[0]));
    return 0;
}
