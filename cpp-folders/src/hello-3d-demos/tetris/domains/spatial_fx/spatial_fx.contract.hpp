#pragma once

#include <cstdint>
#include <array>
#include <span>
#include "shs_renderer.hpp"

namespace tetris {
    namespace spatial_fx {

        // ============================================================================
        // POD 3: SPATIAL FX — 3D Voxel Shatter, Camera Spring, Floating Text Popups
        // Uses Structure-of-Arrays (SoA) for cache-friendly particle updates.
        // ============================================================================

        struct Particle {
            glm::vec3 position;
            glm::vec3 velocity;
            glm::vec3 color;
            float life = 1.2f;
            uint8_t   age    = 0u;     // Frame counter for deterministic emission
        };

        /// SoA-aligned particle table (dense, contiguous).
        struct ParticleTableSoA {
            std::pmr::vector<glm::vec3> position;
            std::pmr::vector<glm::vec3> velocity;
            std::pmr::vector<glm::vec3> color;
            std::pmr::vector<float>     life;

            explicit ParticleTableSoA(std::pmr::memory_resource* mr)
                : position(mr), velocity(mr), color(mr), life(mr) {}

            void add(glm::vec3 pos, glm::vec3 vel, glm::vec3 col, float duration = 1.2f);

            // Bump-alloc append for zero-allocation during particle lifetime
            inline void push_back(std::pmr::memory_resource* mr,
                                  const glm::vec3& p, const glm::vec3& v,
                                  const glm::vec3& c, float dur) {
                auto ptr = mr->allocate(Particle{}, sizeof(Particle));
                Particle* part = new (ptr) Particle();
                part->position  = p;
                part->velocity  = v;
                part->color     = c;
                part->life      = dur;
                part->age       = 0u;
                position.push_back(p);
                velocity.push_back(v);
                color.push_back(c);
                life.push_back(dur);
            }
        };

        /// Camera spring state: smoothly interpolates shake back to zero.
        struct CameraShakeState {
            float shake_x = 0.0f;
            float shake_y = 0.0f;
            float target_shake = 0.0f; // Set by line-clear or hard drop events
            bool active = false;

            void apply(float dt, std::pmr::vector<float>& out_buffer) {
                if (!active) return;
                float decay = 8.0f; // Damping factor per second (~16 Hz)
                shake_x *= (1.0f - decay * dt);
                shake_y *= (1.0f - decay * dt);
                out_buffer.push_back(shake_x); // For renderer edge consumption
            }
        };

    } // namespace spatial_fx
} // namespace tetris
