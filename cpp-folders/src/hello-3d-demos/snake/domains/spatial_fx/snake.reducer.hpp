#pragma once

#include "snake.contract.hpp"

// spatial_fx pod — pure particle reducer. Emits bursts, advances motion, culls dead particles.
// Allocation-free (input arena injected for any growth).

namespace snake::spatial_fx {

    inline void emit_burst(ShatterParticleSoA& out, const SpawnBurst& burst, uint32_t& rng) {
        std::pmr::vector<glm::vec2> deltas;
        deltas.reserve(burst.count);
        for (int i = 0; i < burst.count; ++i) {
            // Deterministic spread: radial directions from a seeded LCG.
            uint32_t r = rng * 1664525u + 1013904223u;
            float ang = (float)(r % 720u) / 180.0f * 3.14159265f;   // 0..2pi
            deltas.push_back({ std::cos(ang), std::sin(ang) });
        }
        for (const auto& d : deltas) {
            out.position.push_back(burst.origin);
            out.velocity.push_back(d * burst.speed);
            out.life.push_back(1.0f);
        }
    }

    inline ShatterParticleSoA reduce_spatial_fx(
        const ShatterParticleSoA& prev,
        float dt,
        std::pmr::memory_resource* arena)
    {
        // Advance + cull in one pass (copy-then-trim).
        std::pmr::vector<glm::vec3> pos;   pos.reserve(prev.position.size());
        std::pmr::vector<glm::vec2> vel;   vel.reserve(prev.velocity.size());
        std::pmr::vector<float>     life;  life.reserve(prev.life.size());

        for (size_t i = 0; i < prev.position.size(); ++i) {
            float decay = dt * 1.4f;   // per-frame lifetime drain
            if (prev.life[i] <= decay) continue;   // dead -> cull
            pos.push_back(prev.position[i] + prev.velocity[i] * dt);
            vel.push_back(prev.velocity[i]);
            life.push_back(std::max(0.0f, prev.life[i] - decay));
        }

        ShatterParticleSoA out;
        out.position.swap(pos);
        out.velocity.swap(vel);
        out.life.swap(life);
        return out;
    }

} // namespace snake::spatial_fx
