#pragma once

// snake plan — builds a render-ready triangle list (clip-space corners + pre-shaded colors) for one frame.
// Mirrors tetris::plan_tetris_scene: world-space boxes → vp transform → Lambert diffuse shading. Pure function;
// never touches SDL or game state. Output is the canonical PipelineExecutionPlan{triangles} consumed by main's
// tiled rasterizer (see docs/spec/conventions.md).
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "shs_renderer.hpp"   // shs::Color, shs::Math (shared renderer from hello-3d-primitives)
#include "snake.contract.hpp"                              // PipelineExecutionPlan{triangles}, ShatterParticleSoA
#include "snake_level_01.hpp"       // arena bounds (GRID_W/H) + food table

namespace snake::spatial_fx {

    // Arena center in world space — derived from the level data so no free-function helper is needed.
    inline glm::vec3 level_arena_center(const SnakeLevel01& level) {
        return level.arena_center;
    }

    struct LowPolyTriangle {
        glm::vec3  p0, p1, p2;
        shs::Color color;
        float      depth_bias = 0.0f;
        explicit LowPolyTriangle(glm::vec3 a, glm::vec3 b, glm::vec3 c, shs::Color col, float bias = 0.0f)
            : p0(a), p1(b), p2(c), color(col), depth_bias(bias) {}
    };

    // Extrude a square in the xy-plane into a box prism (6 faces → 12 triangles).
    static inline void add_box(
        std::vector<LowPolyTriangle>& tris,
        glm::vec3 center, glm::vec3 size,
        shs::Color c_top, shs::Color c_side, shs::Color c_bot,
        float bias = 0.0f)
    {
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
        add_quad(tris, p100, p000, p010, p110, c_side, bias); // Back  (-Z)
        add_quad(tris, p010, p011, p111, p110, c_top , bias); // Top   (+Y)
        add_quad(tris, p000, p100, p101, p001, c_bot , bias); // Bottom(-Y)
        add_quad(tris, p100, p110, p111, p101, c_side, bias); // Right (+X)
        add_quad(tris, p000, p001, p011, p010, c_side, bias); // Left  (-X)
    }

    static inline void add_quad(
        std::vector<LowPolyTriangle>& tris,
        glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, glm::vec3 v3,
        shs::Color col, float bias)
    {
        tris.emplace_back(v0, v1, v2, col, bias);
        tris.emplace_back(v0, v2, v3, col, bias);
    }

    // Build the full render-ready scene: board tiles + snake segments + food orb.
    static inline PipelineExecutionPlan plan_snake_scene(
        const SnakeSnapshot& snap,
        std::span<const SnakeCommand> commands,   // unused — geometry is static per frame; kept for API symmetry
        const config::Difficulty& difficulty,
        const SnakeLevel01& level,
        ShatterParticleSoA& particles)
    {
        PipelineExecutionPlan plan;

        float cell = 1.0f;   // world-space size of one grid cell (matches level arena)
        glm::vec3 arena_center = level_arena_center(level);
        const int GW = level.GRID_W, GH = level.GRID_H;

        std::vector<LowPolyTriangle> tris;
        tris.reserve(4000);

        // Orbiting top-down camera: elevated eye that slowly yaws around the arena center.
        float yaw = 0.6f * snap.body.position.size();   // slow rotation proportional to body length (visual)
        glm::vec3 eye = arena_center + glm::vec3(std::sin(yaw) * 14.0f, 9.0f, std::cos(yaw) * 14.0f);
        plan.view_matrix = glm::lookAtLH(eye, arena_center, glm::vec3(0, 1, 0));
        plan.proj_matrix = glm::perspectiveLH_NO(glm::radians(60.0f), 1.0f, 0.15f, 120.0f);
        plan.vp_matrix   = plan.proj_matrix * plan.view_matrix;

        // 1. Board tiles — every empty cell is a semi-3D extruded box (checkerboard top color).
        for (int y = 0; y < GH; ++y) {
            for (int x = 0; x < GW; ++x) {
                glm::ivec2 p(x, y);
                bool occupied = false;
                for (const auto& seg : snap.body.position) { if (seg.x == p.x && seg.y == p.y) { occupied = true; break; } }
                if (occupied) continue;

                glm::vec3 c(arena_center.x + cell * float(x), arena_center.y + cell * float(y), 0.0f);
                bool dark = ((x + y) & 1) != 0;
                shs::Color top_color{ dark ? 48 : 62, dark ? 58 : 72, dark ? 92 : 106, 255 };   // subtle checkerboard
                float tile_height = (difficulty.solid_walls) ? 0.9f : 0.5f;

                add_box(tris, c, glm::vec3(cell * 0.96f, tile_height, cell * 0.96f), top_color, top_color, top_color);
            }
        }

        // 2. Snake body — gradient extruded prisms head→tail; head is a slightly taller block.
        const int n = static_cast<int>(snap.body.position.size());
        for (int i = 0; i < n; ++i) {
            glm::vec3 seg(snap.body.position[i].x, snap.body.position[i].y, 0.0f);
            float t = static_cast<float>(i) / std::max(1, n - 1);   // 0 at head, 1 at tail
            shs::Color color{ (20 + 45 * t), (180 - 90 * t), (170 - 130 * t), 255 };

            float height = (i == 0) ? 0.6f : 0.45f;
            add_box(tris, seg, glm::vec3(cell * 0.84f, height, cell * 0.84f), color, color, color);
        }

        // 3. Food — glowing berry approximated as a small box with depth bias (pops above tiles).
        if (snap.food.pos.x >= 0) {
            glm::vec3 c(arena_center.x + cell * float(snap.food.pos.x), arena_center.y + cell * float(snap.food.pos.y), 0.0f);
            add_box(tris, c, glm::vec3(cell * 0.8f, 0.9f, cell * 0.8f), shs::Color{ 255, 180, 140, 255 }, shs::Color{ 150, 40, 30, 255 }, shs::Color{ 150, 40, 30, 255 }, 0.06f);
        }

        // 4. Particle FX — emit a shatter burst at the head on game-over.
        if (!snap.alive) {
            glm::vec3 head(snap.head_pos.x, snap.head_pos.y, 0.0f);
            for (int i = 0; i < 40; ++i) {
                uint32_t rng = level.rng_state;
                uint32_t r = rng * 1664525u + 1013904223u;
                float ang = (float)(r % 720u) / 180.0f * glm::pi<float>();   // 0..2pi
                glm::vec2 dir(std::cos(ang), std::sin(ang));
                particles.add(head, dir * ((float)(r % 120u) / 30.0f + 1.0f), shs::Color{ 255, 90, 60, 255 }, 0.8f);
            }
        }

        // Transform & shade triangles (Lambert diffuse). World-space normal → dot with light dir.
        glm::vec3 SUN_DIR = glm::normalize(glm::vec3(0.45f, -0.85f, 0.35f));   // world-space sun
        glm::vec3 L       = -SUN_DIR;

        for (const auto& tri : tris) {
            glm::vec4 c0 = plan.vp_matrix * glm::vec4(tri.p0, 1.0f);
            glm::vec4 c1 = plan.vp_matrix * glm::vec4(tri.p1, 1.0f);
            glm::vec4 c2 = plan.vp_matrix * glm::vec4(tri.p2, 1.0f);

            glm::vec3 N = glm::cross(tri.p1 - tri.p0, tri.p2 - tri.p0);
            float len = glm::length(N);
            if (len < 1e-6f) continue;
            N /= len;

            float diffuse = std::max(0.0f, glm::dot(N, L));
            float ambient = std::max(0.0f, N.y) * 0.20f + 0.15f;   // top faces get ambient fill

            glm::vec3 base_col = glm::vec3(tri.color.r, tri.color.g, tri.color.b) / 255.0f;
            glm::vec3 lit_rgb = base_col * (diffuse * glm::vec3(1.0f, 0.98f, 0.92f) + ambient * glm::vec3(0.50f, 0.70f, 1.0f));

            plan.triangles.push_back({ c0, c1, c2, shs::rgb01_to_color(lit_rgb), tri.depth_bias });
        }

        return plan;
    }

} // namespace snake::spatial_fx
