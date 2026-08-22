#pragma once
// tetris/domains/spatial_fx/spatial_fx.reducer.hpp — FX STEP (tetris::spatial_fx)
// Event-fed particle physics + camera-shake spring. Deterministic xorshift
// replaces the old rand() debris velocities (headless reproducibility).
#include <algorithm>
#include <cstdint>
#include <span>

#include <domains/matrix/matrix.contract.hpp>
#include <domains/matrix/matrix.event.hpp>
#include <domains/spatial_fx/spatial_fx.contract.hpp>

namespace tetris::spatial_fx {
using tetris::matrix::MatrixEvent;
using tetris::matrix::MatrixEventType;

    static inline uint32_t fx_rand(uint32_t& s) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        return s;
    }

    // Steps FxState in place (long-lived PMR-backed particles).
    static inline void step_fx(FxState& fx, std::span<const MatrixEvent> events, float dt) {
        fx.time += dt;
        if (fx.camera_shake > 0.0f) {
            fx.camera_shake = std::max(0.0f, fx.camera_shake - dt * 4.0f);
        }

        for (const auto& ev : events) {
            switch (ev.type) {
            case MatrixEventType::HARD_DROP_SLAM:
                fx.camera_shake = 0.35f;
                break;
            case MatrixEventType::LINES_CLEARED: {
                fx.camera_shake = (ev.lines_cleared_count >= 4) ? 0.65f : 0.25f;
                for (int i = 0; i < ev.lines_cleared_count; ++i) {
                    float row_y = (float)ev.cleared_rows[i];
                    for (int col = 0; col < GRID_W; ++col) {
                        glm::vec3 p((float)col - 4.5f, row_y, 0.0f);
                        glm::vec3 vel(
                            ((col - 4.5f) * 1.2f) + ((float)(fx_rand(fx.rng_state) % 100) / 50.0f - 1.0f),
                            3.0f + ((float)(fx_rand(fx.rng_state) % 100) / 30.0f),
                            -2.5f - ((float)(fx_rand(fx.rng_state) % 100) / 40.0f)
                        );
                        fx.particles.add(p, vel, shs::Color{ 40, 220, 240, 255 }, 1.2f);
                    }
                }
                break;
            }
            default:
                break;
            }
        }

        // Integrate + compact (erase pattern preserved from original demo)
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
    }

} // namespace tetris::spatial_fx
