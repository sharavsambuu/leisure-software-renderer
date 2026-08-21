#pragma once

// spatial_fx pod — rendering geometry + particle FX vocabulary. Pure data; consumed by the plan at runtime.
#include <cstdint>
#include <array>
#include <memory_resource>
#include <glm/glm.hpp>
#include "../../../hello-shs-renderer/shs_renderer.hpp"   // shs::Color, shs::Camera3D

namespace snake::spatial_fx {

    // Face: one render-ready triangle (world-space) with a base color that the rasterizer lights.
    struct Face {
        glm::vec4 v0{ 0.0f, 0.0f, 0.0f, 1.0f };   // world-space corner (w = 1)
        glm::vec4 v1{ 0.0f, 0.0f, 0.0f, 1.0f };
        glm::vec4 v2{ 0.0f, 0.0f, 0.0f, 1.0f };
        glm::vec3 normal{ 0.0f, 1.0f, 0.0f };      // unit outward normal (world space) — used for flat shading
        shs::Color color;                           // base/face color (Lambert-lit by the renderer)
        float depth_bias = 0.0f;                    // per-face z-buffer bias (food/snake pop above tiles)
    };

    // PipelineExecutionPlan: render-ready geometry for one frame. Zero game logic — consumed by the rasterizer.
    struct PipelineExecutionPlan {
        std::pmr::vector<Face> faces;   // all visible faces across board, walls, food, snake + FX
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
