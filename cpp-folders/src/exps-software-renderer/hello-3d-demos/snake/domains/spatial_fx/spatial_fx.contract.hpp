#pragma once

// spatial_fx pod — rendering geometry + particle FX vocabulary. Pure data; consumed by the plan at runtime.
#include <cstdint>
#include <cstddef>
#include <array>
#include <memory_resource>
#include <glm/glm.hpp>
#include "shs_renderer.hpp"   // shs::render::Color (shared renderer from hello-shs-renderer; dir is on the global include path via parent aggregator)
#include "shs/containers/soa_table.hpp"   // P1.5: shared SoaTable backing store (§7.2)

namespace snake::spatial_fx {

    // ProcessedTriangle: canonical renderer format (see docs/spec/conventions.md) — clip-space corners +
    // pre-shaded color + depth bias. Declared before PipelineExecutionPlan, which stores them by value.
    struct ProcessedTriangle {
        glm::vec4  c0, c1, c2;
        shs::render::Color lit_color;
        float      depth_bias;
    };

    // PipelineExecutionPlan: render-ready geometry for one frame. Zero game logic — consumed by the rasterizer.
    struct PipelineExecutionPlan {
        std::pmr::vector<ProcessedTriangle> triangles;   // all visible faces across board, walls, food, snake + FX
        glm::mat4 view_matrix = glm::mat4(1.0f);         // world→view (orbiting top-down camera)
        glm::mat4 proj_matrix = glm::mat4(1.0f);         // view→clip (perspective)
        glm::mat4 vp_matrix   = glm::mat4(1.0f);         // shared per-frame camera matrix (world→clip)
    };

    // Camera params for the orbiting top-down view of the semi-3D board.
    struct SnakeCameraParams {
        float orbit_angle = 0.0f;       // radians — slow yaw around the arena center (orbiting top-down)
        glm::vec3 position{ 0.0f, 5.0f, 12.0f };   // elevated side view offset from arena center
        float focalLength = 45.0f;
        float fieldOfView = 60.0f * (glm::pi<float>() / 180.0f);
    };

    // Light params — world-space ambient + direction for flat-shaded faces.
    struct SnakeLightParams {
        glm::vec3 ambient{ 0.25f, 0.25f, 0.30f };
        glm::vec3 direction{ -1.0f, -1.0f, -1.0f };   // top-left lighting (world space)
    };

    // ShatterParticleSoA: particle system for FX bursts (food-eat sparkle + game-over shatter).
    // P1.5: backed by the shared lib SoaTable (shs/containers/soa_table.hpp) — one
    // contiguous 64-byte-aligned pmr allocation per column, generational handles,
    // swap-and-pop removal (§7.2). Column order: position, velocity, color, life.
    // Mirrors tetris's 4-vector SoA (kept there until its own P1.5 migration).
    using ShatterParticleSoA = shs::containers::SoaTable<glm::vec3, glm::vec3, shs::render::Color, float>;

    // Named column indices for the particle table (readability at call sites).
    inline constexpr std::size_t kParticlePosition = 0;
    inline constexpr std::size_t kParticleVelocity = 1;
    inline constexpr std::size_t kParticleColor    = 2;
    inline constexpr std::size_t kParticleLife     = 3;

    // Burst helper: emits one particle row (was SoaTable-free struct's add()).
    inline shs::containers::SoaHandle add_particle(ShatterParticleSoA& particles,
                                                   glm::vec3 pos, glm::vec3 vel,
                                                   shs::render::Color col, float duration = 0.8f)
    {
        return particles.insert(pos, vel, col, duration);
    }

} // namespace snake::spatial_fx
