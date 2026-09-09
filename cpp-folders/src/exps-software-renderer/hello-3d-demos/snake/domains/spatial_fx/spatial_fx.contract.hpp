#pragma once

// spatial_fx pod — rendering geometry + particle FX vocabulary. Pure data; consumed by the plan at runtime.
#include <cstdint>
#include <array>
#include <memory_resource>
#include <glm/glm.hpp>
#include "shs_renderer.hpp"   // shs::Color (shared renderer from hello-shs-renderer; dir is on the global include path via parent aggregator)

namespace snake::spatial_fx {

    // ProcessedTriangle: canonical renderer format (see docs/spec/conventions.md) — clip-space corners +
    // pre-shaded color + depth bias. Declared before PipelineExecutionPlan, which stores them by value.
    struct ProcessedTriangle {
        glm::vec4  c0, c1, c2;
        shs::Color lit_color;
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

    // ShatterParticleSoA: particle system for FX bursts (food-eat sparkle + game-over shatter). Mirrors tetris's 4-vector SoA.
    struct ShatterParticleSoA {
        std::pmr::vector<glm::vec3> position;   // world-space position
        std::pmr::vector<glm::vec3> velocity;   // spread in the board plane (xy, z == 0)
        std::pmr::vector<shs::Color> color;     // rgb01 base color
        std::pmr::vector<float> life;           // remaining lifetime (seconds)

        explicit ShatterParticleSoA(std::pmr::memory_resource* mr)
            : position(mr), velocity(mr), color(mr), life(mr) {}

        void add(glm::vec3 pos, glm::vec3 vel, shs::Color col, float duration = 0.8f);
    };

    inline void ShatterParticleSoA::add(glm::vec3 pos, glm::vec3 vel, shs::Color col, float duration) {
        position.push_back(pos);
        velocity.push_back(vel);
        color.push_back(col);
        life.push_back(duration);
    }

} // namespace snake::spatial_fx
