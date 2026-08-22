#pragma once
// tetris/domains/spatial_fx/spatial_fx.reducer.hpp — FX STEP (tetris::spatial_fx)
// Event-fed particle physics + camera spring/pulse + shockwave rings.
// Deterministic xorshift replaces rand() (headless reproducibility).
// Consumes BOTH matrix raw facts and progression derived events — listeners
// only; never touches the grid or score state (Rule 8.1).
#include <algorithm>
#include <cstdint>
#include <span>

#include <domains/matrix/matrix.contract.hpp>
#include <domains/matrix/matrix.event.hpp>
#include <domains/progression/progression.event.hpp>
#include <domains/spatial_fx/spatial_fx.contract.hpp>

namespace tetris::spatial_fx {
using tetris::matrix::MatrixEvent;
using tetris::matrix::MatrixEventType;
using tetris::matrix::GRID_W;
using tetris::progression::ProgressionEvent;
using tetris::progression::ProgressionEventType;

    static inline uint32_t fx_rand(uint32_t& s) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        return s;
    }

    // Amber spark trail along the hard-drop column (speed feel).
    static inline void spark_trail(FxState& fx, const MatrixEvent& ev) {
        const float x = ev.world_position.x;
        const int cells = std::min(static_cast<int>(ev.cells), 18);
        for (int i = 0; i < cells; ++i) {
            for (int k = 0; k < 2; ++k) {
                glm::vec3 p(x + ((float)(fx_rand(fx.rng_state) % 100) / 100.0f - 0.5f),
                            ev.world_position.y + (float)i + 0.5f, 0.0f);
                glm::vec3 vel(((float)(fx_rand(fx.rng_state) % 100) / 50.0f - 1.0f) * 2.0f,
                              1.0f + (float)(fx_rand(fx.rng_state) % 100) / 60.0f,
                              -0.5f);
                fx.particles.add(p, vel, shs::Color{ 255, 190, 80, 255 }, 0.35f);
            }
        }
    }

    // Golden confetti/firework burst on the victory crescendo ("photo finish").
    static inline void victory_fireworks(FxState& fx) {
        static const shs::Color GOLD[] = {
            shs::Color{ 255, 215,  80, 255 },
            shs::Color{ 255, 170,  40, 255 },
            shs::Color{ 255, 240, 180, 255 },
            shs::Color{ 255, 225,  45, 255 }
        };
        for (int i = 0; i < 140; ++i) {
            glm::vec3 p((float)(fx_rand(fx.rng_state) % 200) / 10.0f - 10.0f,
                        (float)(fx_rand(fx.rng_state) % 160) / 10.0f + 2.0f,
                        (float)(fx_rand(fx.rng_state) % 100) / 40.0f - 1.0f);
            glm::vec3 vel((float)(fx_rand(fx.rng_state) % 100) / 20.0f - 2.5f,
                          4.0f + (float)(fx_rand(fx.rng_state) % 100) / 25.0f,
                          (float)(fx_rand(fx.rng_state) % 100) / 50.0f - 1.0f);
            fx.particles.add(p, vel, GOLD[fx_rand(fx.rng_state) % 4], 1.6f);
        }
    }

    // Steps FxState in place (long-lived PMR-backed particles/rings).
    static inline void step_fx(FxState& fx,
                               std::span<const MatrixEvent> matrix_events,
                               std::span<const ProgressionEvent> progression_events,
                               float dt) {
        fx.time += dt;
        if (fx.camera_shake > 0.0f) {
            fx.camera_shake = std::max(0.0f, fx.camera_shake - dt * 4.0f);
        }
        if (fx.camera_pulse > 0.0f) {
            fx.camera_pulse = std::max(0.0f, fx.camera_pulse - dt * 2.2f);
        }

        // --- Matrix raw facts --------------------------------------------------
        for (const auto& ev : matrix_events) {
            switch (ev.type) {
            case MatrixEventType::HARD_DROP_SLAM:
                fx.camera_shake = 0.35f;
                spark_trail(fx, ev);
                break;
            case MatrixEventType::LINES_CLEARED: {
                const bool tetris = (ev.lines_cleared_count >= 4);
                fx.camera_shake = tetris ? 0.65f : 0.25f;
                if (tetris) fx.camera_pulse = 1.0f;   // zoom punch on 4-line clears
                for (int i = 0; i < ev.lines_cleared_count; ++i) {
                    float row_y = (float)ev.cleared_rows[i];
                    for (int col = 0; col < GRID_W; ++col) {
                        glm::vec3 p((float)col - 4.5f, row_y, 0.0f);
                        const float energy = tetris ? 1.6f : 1.0f;   // oversized burst
                        glm::vec3 vel(
                            ((col - 4.5f) * 1.2f) + ((float)(fx_rand(fx.rng_state) % 100) / 50.0f - 1.0f),
                            (3.0f + ((float)(fx_rand(fx.rng_state) % 100) / 30.0f)) * energy,
                            -2.5f - ((float)(fx_rand(fx.rng_state) % 100) / 40.0f)
                        );
                        // Gold flecks mixed into tetris bursts
                        shs::Color pc = (tetris && (col & 1) == 0)
                            ? shs::Color{ 255, 210, 70, 255 }
                            : shs::Color{ 40, 220, 240, 255 };
                        fx.particles.add(p, vel, pc, 1.2f);
                    }
                }
                break;
            }
            default:
                break;
            }
        }

        // --- Progression derived events -----------------------------------------
        for (const auto& pev : progression_events) {
            switch (pev.type) {
            case ProgressionEventType::CLOCK_TICK:
                // Threshold shockwave ring from the board every 30-second tick.
                fx.rings.add(glm::vec3(0.0f, 9.5f, 0.4f), 1.0f, 16.0f,
                             shs::Color{ 120, 220, 255, 255 }, 0.8f);
                break;
            case ProgressionEventType::OBJECTIVE_COMPLETED:
                victory_fireworks(fx);
                fx.camera_pulse = 1.0f;
                break;
            default:
                break;
            }
        }

        // Integrate + compact particles (erase pattern preserved from original demo)
        auto& P = fx.particles;
        for (size_t i = 0; i < P.position.size();) {
            P.position[i] += P.velocity[i] * dt;
            P.velocity[i].y -= 18.0f * dt; // Gravity
            P.life[i] -= dt;
            if (P.life[i] <= 0.0f) {
                P.position.erase(P.position.begin() + i);
                P.velocity.erase(P.velocity.begin() + i);
                P.color.erase(P.color.begin() + i);
                P.life.erase(P.life.begin() + i);
            } else {
                ++i;
            }
        }

        // Integrate + compact rings
        auto& R = fx.rings;
        for (size_t i = 0; i < R.center.size();) {
            R.radius[i] += R.speed[i] * dt;
            R.life[i] -= dt;
            if (R.life[i] <= 0.0f) {
                R.center.erase(R.center.begin() + i);
                R.radius.erase(R.radius.begin() + i);
                R.speed.erase(R.speed.begin() + i);
                R.life.erase(R.life.begin() + i);
                R.max_life.erase(R.max_life.begin() + i);
                R.color.erase(R.color.begin() + i);
            } else {
                ++i;
            }
        }
    }

} // namespace tetris::spatial_fx