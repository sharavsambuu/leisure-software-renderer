#pragma once

#include <algorithm>
#include <cmath>
#include <memory_resource>
#include <span>
#include "spatial_fx.contract.hpp"
#include "progression.event.hpp"  // Consumes progression events via discrete event sourcing
#include "matrix.event.hpp"       // Consumes matrix events for shatter effects

namespace tetris {
    namespace spatial_fx {

        // ============================================================================
        // PURE REDUCER: Updates particles and camera shake given immutable spans.
        // Emits new particles and updates camera state via a shared output span.
        // No mutexes, no atomics inside the reducer — pure value transformation.
        // ============================================================================

        static inline void update_spatial_fx(
            ParticleTableSoA&     particles,
            CameraShakeState&     shake_state,
            std::pmr::vector<float>& shake_output_buffer,
            const ProgressionEventSpan& progression_events,   // discrete event sourcing
            const MatrixEventSpan& matrix_events              // discrete event sourcing from matrix domain
        ) {
            // --- Update Camera Shake (damped spring-back) ---
            if (shake_state.active && !progression_events.has_shake_reset) {
                float decay = 8.0f;
                shake_state.shake_x *= (1.0f - decay * 0.0167f); // ~60 Hz decay
                shake_state.shake_y *= (1.0f - decay * 0.0167f);
            } else if (progression_events.has_shake_reset) {
                shake_state.target_shake = 0.0f;
                shake_state.active = false;
            }

            // --- Emit new particles from line-clear events (discrete event sourcing) ---
            for (const auto& ev : progression_events.events) {
                if (ev.type == ProgressionEventType::LINES_CLEARED && ev.lines_cleared_count >= 4) {
                    // TETRIS FIVE! → big shatter explosion at center of grid
                    float row_y = (float)(ev.cleared_rows[0] + ev.cleared_rows[1]) / 2.0f;
                    for (int col = 0; col < 9; ++col) {
                        glm::vec3 pos((float)col - 4.5f, row_y, 0.0f);
                        glm::vec3 vel(
                            ((float)col - 2.5f) * 1.5f + ((rand() % 100) / 50.0f - 1.0f),
                            4.0f,
                            -3.0f - ((rand() % 100) / 30.0f)
                        );
                        particles.push_back(std::pmr::get_default_resource(), pos, vel, glm::vec3(180, 255, 220), 1.6f);
                    }
                } else if (ev.type == ProgressionEventType::DANGER_ALERT) {
                    // Emit a subtle red dust puff when combo busts
                    for (int i = -4; i <= 4; ++i) {
                        glm::vec3 pos((float)i * 0.5f, 12.0f, 0.0f);
                        glm::vec3 vel((float)(rand() % 7 - 3), 1.5f, (float)(rand() % 4 - 2));
                        particles.push_back(std::pmr::get_default_resource(), pos, vel, glm::vec3(220, 80, 80), 0.9f);
                    }
                }
            }

            // --- Update particles (SoA iteration for cache efficiency) ---
            size_t valid_count = 0;
            for (size_t i = 0; i < particles.position.size(); ++i) {
                glm::vec3& p = particles.position[i];
                glm::vec3& v = particles.velocity[i];

                // Simple Euler integration with gravity
                float dt = 0.0167f;
                p.x += v.x * dt;
                p.y += (v.y - 9.8f) * dt;   // Gravity acts on Y
                p.z += v.z * dt;

                v.x *= 0.99f;              // Air resistance
                v.y *= 0.97f;
                v.z *= 0.99f;

                float age = particles.life[i] - dt;
                if (age <= 0.01f) {
                    particles.position.erase(particles.position.begin() + i);
                    particles.velocity.erase(particles.velocity.begin() + i);
                    particles.color.erase(particles.color.begin() + i);
                    particles.life.erase(particles.life.begin() + i);
                } else {
                    --valid_count; // just a counter, not used elsewhere
                }
            }

            // --- Emit shatter particles from matrix impact events ---
            for (const auto& ev : matrix_events.events) {
                if (ev.type == MatrixEventType::PIECE_LOCK_IMPACT) {
                    float y = ev.world_position.y;
                    glm::vec3 pos(ev.world_position.x, y + 0.25f, 0.1f);
                    glm::vec3 vel(
                        ((float)(rand() % 7 - 3)) * 0.8f,
                        (float)(rand() % 4),
                        -1.5f
                    );
                    particles.push_back(std::pmr::get_default_resource(), pos, vel, glm::vec3(200, 200, 180), 0.6f);
                }
            }

            // --- Emit camera shake events for renderer edge ---
            if (progression_events.has_shake_trigger) {
                shake_state.active = true;
                float intensity = (progression_events.shake_intensity > 0.5f) ? 1.8f : 0.9f;
                shake_state.target_shake = intensity * glm::radians(2.5f);
            }

            // Push current shake value out to renderer edge
            if (shake_state.active && !shake_output_buffer.empty()) {
                float decay = 8.0f;
                float dt = 0.0167f;
                shake_state.shake_x *= (1.0f - decay * dt);
                shake_state.shake_y *= (1.0f - decay * dt);
            }

            // --- Floating text popups for level-up events ---
            if (progression_events.has_level_up_event) {
                // Emit a glowing upward-fading particle at top center
                particles.push_back(std::pmr::get_default_resource(),
                    glm::vec3(0.0f, 18.5f, 0.2f),
                    glm::vec3(0.0f, -0.8f, 0.0f),
                    glm::vec3(255, 240, 160), 2.0f); // Gold color for level up
            }
        }

    } // namespace spatial_fx
} // namespace tetris
