#pragma once
// tetris/domains/environment/environment.plan.hpp — DIORAMA BATCH PLANNER
// (tetris::environment) Reads the environment snapshot READ-ONLY and appends
// the Encore Finale set-dressing triangles to the shared scene batch: crowd
// silhouettes with light-wave pulses, animated neon pedestal strips, and the
// blackout spotlight cone. Gated by the caller (main wires env_finale) so
// every other stage renders pixel-identical. Pure value math — no platform I/O.
#include <cmath>
#include <glm/glm.hpp>

#include <domains/spatial_fx/spatial_fx.contract.hpp>

namespace tetris::environment {

    // Phase ids mirrored as plain ints: this planner consumes ONLY plain
    // values, so spatial_fx's planner never depends on the environment pod's
    // contract header (Constitution II §6.3 / Rule 8.1).
    static constexpr int FINALE_PHASE_CALM      = 1;
    static constexpr int FINALE_PHASE_RAIN      = 2;
    static constexpr int FINALE_PHASE_BLACKOUT  = 3;
    static constexpr int FINALE_PHASE_CRESCENDO = 4;

    // Plain-value inputs (Rule 1): mood/dim/crowd/phase arrive as the same
    // wires main already puts on FxState — one source of truth, no snapshot.
    struct FinaleInputs {
        int   phase       = FINALE_PHASE_CALM;
        float mood        = 0.0f;   // 0 cyan .. 1 gold
        float dim         = 0.0f;   // blackout dimming 0..1
        float crowd_pulse = 0.0f;   // crowd energy 0..1
    };

    // Append the finale diorama to `tris`. fx.time drives the light wave;
    // env.crowd_pulse scales its amplitude; env.mood tints emissive strips
    // cyan → crimson → gold; env.dim gates the blackout light shaft.
    static inline void append_finale_diorama(
        std::vector<spatial_fx::LowPolyTriangle>& tris,
        const FinaleInputs&                       env,
        float                                     time
    ) {

        // Mood palette: cyan (0) → crimson (0.5) → gold (1).
        const float mood = glm::clamp(env.mood, 0.0f, 1.0f);
        const glm::vec3 mood_c = (mood < 0.5f)
            ? glm::mix(glm::vec3(0.16f, 0.86f, 0.94f),
                       glm::vec3(0.96f, 0.22f, 0.22f), mood * 2.0f)
            : glm::mix(glm::vec3(0.96f, 0.22f, 0.22f),
                       glm::vec3(1.00f, 0.84f, 0.18f), (mood - 0.5f) * 2.0f);

        // Crowd-silhouette diorama: two staggered rows of dark head-and-
        // shoulder boxes behind the well, bobbing on a sine wave whose
        // amplitude rides crowd_pulse.
        const shs::render::Color crowd_dark{ 12, 12, 20, 255 };
        for (int i = 0; i < 14; ++i) {
            const float x   = -19.5f + (float)i * 3.0f;
            const float z   = ((i & 1) == 0) ? -11.0f : -13.5f;
            const float ph  = time * (1.6f + 0.35f * (float)(i % 4))
                            + (float)(i % 5) * 1.31f;
            const float amp = 0.15f + 0.85f * glm::clamp(env.crowd_pulse, 0.0f, 1.0f);
            const float y   = -0.35f + std::sin(ph) * 0.55f * amp;
            spatial_fx::MeshGen::add_box(tris, glm::vec3(x, y, z),
                             glm::vec3(1.7f, 2.4f, 1.4f),
                             crowd_dark, crowd_dark, crowd_dark);
            spatial_fx::MeshGen::add_box(tris, glm::vec3(x, y + 1.75f, z),
                             glm::vec3(1.05f, 1.05f, 1.05f),
                             crowd_dark, crowd_dark, crowd_dark);
            // Light-wave strip: each fan holds an emissive bar that brightens
            // as a traveling wave passes; brightness scales with crowd energy.
            const float wave = (0.5f + 0.5f * std::sin(time * 3.1f - x * 0.45f))
                             * glm::clamp(env.crowd_pulse + 0.15f, 0.0f, 1.0f);
            if (wave > 0.08f) {
                const uint8_t lv = static_cast<uint8_t>(40.0f + 200.0f * wave);
                const float ratio_g = glm::clamp(mood_c.g / glm::max(mood_c.r, 0.05f), 0.0f, 2.0f);
                const float ratio_b = glm::clamp(mood_c.b / glm::max(mood_c.r, 0.05f), 0.0f, 2.0f);
                const shs::render::Color wc{ lv,
                    static_cast<uint8_t>(glm::clamp((float)lv * ratio_g, 0.0f, 255.0f)),
                    static_cast<uint8_t>(glm::clamp((float)lv * ratio_b, 0.0f, 255.0f)),
                    255 };
                spatial_fx::MeshGen::add_box(tris, glm::vec3(x, y + 3.1f, z),
                                 glm::vec3(1.2f, 0.18f, 0.18f), wc, wc, wc);
            }
        }

        // Animated neon pedestal: emissive strips ringing the floor in the
        // mood color, pulsing; the crescendo adds a fast strobe layer.
        const float pulse = 0.65f + 0.35f * std::sin(time * 2.4f)
                          + ((env.phase >= FINALE_PHASE_CRESCENDO)
                             ? 0.25f * std::sin(time * 9.0f) : 0.0f);
        for (int i = 0; i < 10; ++i) {
            const float ang = (float)i / 10.0f * glm::two_pi<float>() + time * 0.15f;
            const float rx  = std::cos(ang) * 17.0f;
            const float rz  = 3.0f + std::sin(ang) * 8.0f;
            const uint8_t pr = static_cast<uint8_t>(glm::clamp(
                mood_c.r * (90.0f + 160.0f * pulse), 30.0f, 255.0f));
            const uint8_t pg = static_cast<uint8_t>(glm::clamp(
                mood_c.g * (90.0f + 160.0f * pulse), 30.0f, 255.0f));
            const uint8_t pb = static_cast<uint8_t>(glm::clamp(
                mood_c.b * (90.0f + 160.0f * pulse), 30.0f, 255.0f));
            const shs::render::Color pc{ pr, pg, pb, 255 };
            spatial_fx::MeshGen::add_box(tris, glm::vec3(rx, -0.82f, rz),
                             glm::vec3(2.6f, 0.16f, 0.5f), pc, pc, pc);
        }

        // Blackout spotlight: during PHASE_BLACKOUT a centered vertical light
        // shaft isolates the well while the rest of the world dims.
        if (env.dim > 0.05f && env.phase == FINALE_PHASE_BLACKOUT) {
            const float shaft = env.dim;
            for (int k = 0; k < 6; ++k) {
                const float t  = (float)k / 6.0f;
                const float w  = 7.5f - 4.5f * t;           // narrows downward
                const float y  = 20.5f - t * 21.0f;          // top → board
                const uint8_t lv = static_cast<uint8_t>(
                    28.0f + 70.0f * shaft * (1.0f - t));
                const shs::render::Color sc{ lv, lv, (uint8_t)(lv + 12), 255 };
                spatial_fx::MeshGen::add_box(tris, glm::vec3(0.0f, y, -0.6f),
                                 glm::vec3(w, 3.6f, 0.12f), sc, sc, sc);
            }
        }
    }

} // namespace tetris::environment
