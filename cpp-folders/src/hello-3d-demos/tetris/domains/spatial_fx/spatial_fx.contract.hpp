#pragma once
// tetris/domains/spatial_fx/spatial_fx.contract.hpp — RENDER VOCABULARY + FX STATE
// Low-poly authoring primitives, batch planner types, SoA shatter particles,
// camera-shake spring state, and the piece palette (tetris::spatial_fx).
#include <memory_resource>
#include <vector>
#include <glm/glm.hpp>
#include "shs_renderer.hpp"

#include <domains/matrix/matrix.contract.hpp>

namespace tetris::spatial_fx {
using tetris::matrix::PieceType;
using tetris::matrix::GRID_W;
using tetris::matrix::GRID_H;
using tetris::matrix::VISIBLE_H;
using tetris::matrix::CELL_SIZE;
using tetris::matrix::BLOCK_GAP;

    struct LowPolyTriangle {
        glm::vec3  p0, p1, p2;
        shs::Color color;
        float      depth_bias = 0.0f;

        LowPolyTriangle(glm::vec3 a, glm::vec3 b, glm::vec3 c, shs::Color col, float bias = 0.0f)
            : p0(a), p1(b), p2(c), color(col), depth_bias(bias) {}
    };

    struct ShatterParticleSoA {
        std::pmr::vector<glm::vec3> position;
        std::pmr::vector<glm::vec3> velocity;
        std::pmr::vector<shs::Color> color;
        std::pmr::vector<float>     life;

        explicit ShatterParticleSoA(std::pmr::memory_resource* mr)
            : position(mr), velocity(mr), color(mr), life(mr) {}

        void add(glm::vec3 pos, glm::vec3 vel, shs::Color col, float duration = 1.2f) {
            position.push_back(pos);
            velocity.push_back(vel);
            color.push_back(col);
            life.push_back(duration);
        }
    };

    // Shockwave ring primitive (event-fed: blitz clock ticks). Expanding circle
    // of voxel segments in the board plane; planner renders it as a batch.
    struct RingFxSoA {
        std::pmr::vector<glm::vec3>  center;
        std::pmr::vector<float>      radius;
        std::pmr::vector<float>      speed;
        std::pmr::vector<float>      life;
        std::pmr::vector<float>      max_life;
        std::pmr::vector<shs::Color> color;

        explicit RingFxSoA(std::pmr::memory_resource* mr)
            : center(mr), radius(mr), speed(mr), life(mr), max_life(mr), color(mr) {}

        void add(glm::vec3 c, float r0, float spd, shs::Color col, float duration = 0.8f) {
            center.push_back(c);
            radius.push_back(r0);
            speed.push_back(spd);
            life.push_back(duration);
            max_life.push_back(duration);
            color.push_back(col);
        }
    };

    // Render-vocabulary color helpers (channel math in float, rounded back).
    static inline shs::Color lerp_color(shs::Color a, shs::Color b, float t) {
        t = glm::clamp(t, 0.0f, 1.0f);
        return shs::Color{
            static_cast<uint8_t>(a.r + (b.r - a.r) * t + 0.5f),
            static_cast<uint8_t>(a.g + (b.g - a.g) * t + 0.5f),
            static_cast<uint8_t>(a.b + (b.b - a.b) * t + 0.5f),
            a.a
        };
    }

    static inline shs::Color fade_color(shs::Color c, float fade) {
        fade = glm::clamp(fade, 0.0f, 1.0f);
        return shs::Color{
            static_cast<uint8_t>(c.r * fade + 0.5f),
            static_cast<uint8_t>(c.g * fade + 0.5f),
            static_cast<uint8_t>(c.b * fade + 0.5f),
            c.a
        };
    }

    namespace MeshGen {
        static inline void add_quad(
            std::vector<LowPolyTriangle>& tris,
            glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, glm::vec3 v3,
            shs::Color col, float bias = 0.0f
        ) {
            tris.emplace_back(v0, v1, v2, col, bias);
            tris.emplace_back(v0, v2, v3, col, bias);
        }

        static inline void add_box(
            std::vector<LowPolyTriangle>& tris,
            glm::vec3 center, glm::vec3 size,
            shs::Color c_top, shs::Color c_side, shs::Color c_bot,
            float bias = 0.0f
        ) {
            glm::vec3 h = size * 0.5f;
            glm::vec3 p000 = center + glm::vec3(-h.x, -h.y, -h.z);
            glm::vec3 p100 = center + glm::vec3( h.x, -h.y, -h.z);
            glm::vec3 p110 = center + glm::vec3( h.x,  h.y, -h.z);
            glm::vec3 p010 = center + glm::vec3(-h.x,  h.y, -h.z);
            glm::vec3 p001 = center + glm::vec3(-h.x, -h.y,  h.z);
            glm::vec3 p101 = center + glm::vec3( h.x, -h.y,  h.z);
            glm::vec3 p111 = center + glm::vec3( h.x,  h.y,  h.z);
            glm::vec3 p011 = center + glm::vec3(-h.x,  h.y,  h.z);

            add_quad(tris, p001, p101, p111, p011, c_side, bias); // Front (+Z)
            add_quad(tris, p100, p000, p010, p110, c_side, bias); // Back (-Z)
            add_quad(tris, p010, p011, p111, p110, c_top , bias); // Top (+Y)
            add_quad(tris, p000, p100, p101, p001, c_bot , bias); // Bottom (-Y)
            add_quad(tris, p100, p110, p111, p101, c_side, bias); // Right (+X)
            add_quad(tris, p000, p001, p011, p010, c_side, bias); // Left (-X)
        }
    }

    struct ProcessedTriangle {
        glm::vec4  c0, c1, c2;
        shs::Color lit_color;
        float      depth_bias;
    };

    struct PipelineExecutionPlan {
        std::pmr::vector<ProcessedTriangle> triangles;
        glm::mat4                           view_matrix;
        glm::mat4                           proj_matrix;
        glm::mat4                           vp_matrix;

        explicit PipelineExecutionPlan(std::pmr::memory_resource* mr)
            : triangles(mr) {}
    };

    // FX lifecycle state (particles + rings + camera spring/pulse + mood).
    // Long-lived; stepped in-place each frame by spatial_fx::step_fx.
    struct FxState {
        ShatterParticleSoA particles;
        RingFxSoA          rings;
        float              camera_shake   = 0.0f;
        float              camera_pulse   = 0.0f;   // zoom punch (tetris / victory)
        float              mood_intensity = 0.0f;   // 0..1 environment mood (main wires
                                               // from blitz clock drain; pod-5 embryo)
        float              time           = 0.0f;
        uint32_t           rng_state      = 0x9e3779b9u;   // deterministic debris velocities

        explicit FxState(std::pmr::memory_resource* mr) : particles(mr), rings(mr) {}
    };

    // Piece palette (render vocabulary — moved out of the grid contract).
    static inline shs::Color get_piece_color(PieceType type) {
        switch (type) {
            case PieceType::I: return shs::Color{  40, 220, 240, 255 }; // Cyan
            case PieceType::O: return shs::Color{ 255, 225,  45, 255 }; // Yellow
            case PieceType::T: return shs::Color{ 185,  70, 240, 255 }; // Purple
            case PieceType::S: return shs::Color{  60, 230,  95, 255 }; // Green
            case PieceType::Z: return shs::Color{ 245,  55,  55, 255 }; // Red
            case PieceType::J: return shs::Color{  45, 110, 245, 255 }; // Blue
            case PieceType::L: return shs::Color{ 255, 140,  35, 255 }; // Orange
            default:           return shs::Color{  80,  90, 105, 255 };
        }
    }

} // namespace tetris::spatial_fx
