#pragma once

// snake plan — builds a render-ready triangle list (clip-space corners + pre-shaded colors) for one frame.
// Mirrors tetris::plan_tetris_scene: world-space boxes → vp transform → Lambert diffuse shading. Pure function;
// never touches SDL or game state. Output is the canonical PipelineExecutionPlan{triangles} consumed by main's
// tiled rasterizer (see docs/spec/conventions.md).
#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "shs_renderer.hpp"   // shs::Color, shs::Math (shared renderer from hello-shs-renderer)
#include "spatial_fx.contract.hpp"  // spatial_fx pod contract: PipelineExecutionPlan{triangles}, ShatterParticleSoA
#include "snake.contract.hpp"  // matrix pod vocabulary: SnakeSnapshot, SnakeCommand (resolved via -I <snake>/domains/matrix)
#include "difficulty.hpp"      // snake::config::Difficulty (resolved via global include dir: <snake>/config)
#include "snake_level_01.hpp"  // arena bounds (GRID_W/H) + food table (resolved via global include dir: <snake>/config/levels)

namespace snake::spatial_fx {

    // Matrix pod vocabulary used in plan signatures. Sibling namespaces are NOT searched by
    // unqualified lookup, so pull the names in explicitly (mirrors tetris's flat root contract).
    using snake::matrix::SnakeSnapshot;
    using snake::matrix::SnakeCommand;

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

    static inline void add_quad(
        std::vector<LowPolyTriangle>& tris,
        glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, glm::vec3 v3,
        shs::Color col, float bias)
    {
        tris.emplace_back(v0, v1, v2, col, bias);
        tris.emplace_back(v0, v2, v3, col, bias);
    }

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

    // Build the full render-ready scene: board tiles + snake segments + food orb.
    // Coordinate convention (Constitution I, docs/spec/conventions.md): LH space, +Y up, +Z forward.
    // The board is a FLOOR in the XZ plane: grid cell (x,y) maps to world (x, 0, -y) — a proper
    // rotation about +X (det=+1), NOT a reflection (x,0,y), which would flip triangle winding and
    // mirror the scene (backfaces culled / inside-out lighting). Grid +y runs away from the default
    // camera (yaw=0 eye sits at +Z), so maps read naturally: x → right, y → up-screen.
    static inline PipelineExecutionPlan plan_snake_scene(
        const SnakeSnapshot& snap,
        std::span<const SnakeCommand> commands,   // unused — geometry is static per frame; kept for API symmetry
        const config::Difficulty& difficulty,
        const SnakeLevel01& level,
        ShatterParticleSoA& particles,
        int canvas_w, int canvas_h)               // for correct projection aspect (mirrors tetris)
    {
        (void)commands;   // reserved for future motion-trail planning
        (void)particles;  // FX emission lives in the main entry edge; kept for API symmetry
        PipelineExecutionPlan plan;

        float cell = 1.0f;   // world-space size of one grid cell (matches level arena)
        glm::vec3 arena_center = level_arena_center(level);
        const int GW = level.GRID_W, GH = level.GRID_H;

        std::vector<LowPolyTriangle> tris;
        tris.reserve(4000);

        // Fixed front-facing camera: parked on the +Z side of the floor (the grid's y=0 row),
        // elevated and tilted DOWN toward the arena center. Deterministic — no orbit, no drift.
        // (Eye is above the XZ plane looking down at it per Constitution I. Aspect = canvas dims;
        // hardcoded 1.0 squeezed the image horizontally — see tetris.plan.hpp for the convention.)
        //
        // VIEW BASIS IS HAND-ROLLED — do NOT switch back to glm::lookAtLH! This GLM's lookAtLH
        // builds the side vector as s = cross(up, f), which for this camera yields s = (-1,0,0):
        // the whole view renders HORIZONTALLY MIRRORED (world +X on screen-left). Verified
        // empirically via the snake gradient (cyan head rendered left of the olive tail) and an
        // autodrive ArrowRight test (bar moved screen-LEFT). Hand-rolled basis below uses the
        // standard right = cross(forward, up) = (+1,0,0) here, so grid +x = screen-RIGHT and
        // grid +y (away from camera) = screen-UP — matching matrix/snake.action.hpp's contract.
        const glm::vec3 eye    = arena_center + glm::vec3(0.0f, 13.0f, 17.0f);
        const glm::vec3 target = arena_center + glm::vec3(0.0f, 0.0f, -1.0f);   // aim a touch past center
        const glm::vec3 fwd = glm::normalize(target - eye);
        const glm::vec3 rgt = glm::normalize(glm::cross(fwd, glm::vec3(0, 1, 0)));   // screen-right
        const glm::vec3 upv = glm::cross(rgt, fwd);                                  // screen-up
        plan.view_matrix = glm::mat4(
            rgt.x, upv.x, fwd.x, 0.0f,
            rgt.y, upv.y, fwd.y, 0.0f,
            rgt.z, upv.z, fwd.z, 0.0f,
            -glm::dot(rgt, eye), -glm::dot(upv, eye), -glm::dot(fwd, eye), 1.0f);
        const float aspect = (canvas_h > 0) ? static_cast<float>(canvas_w) / static_cast<float>(canvas_h) : 1.0f;
        plan.proj_matrix = glm::perspectiveLH_NO(glm::radians(60.0f), aspect, 0.15f, 120.0f);
        plan.vp_matrix   = plan.proj_matrix * plan.view_matrix;

        // 1. Board tiles — every empty cell is a semi-3D extruded box (checkerboard top color).
        for (int y = 0; y < GH; ++y) {
            for (int x = 0; x < GW; ++x) {
                const glm::ivec2 p(x, y);
                bool occupied = false;
                for (const auto& seg : snap.body.position) {
                    // compare in grid space (float -> int cells are exact by construction)
                    if (glm::ivec2(static_cast<int>(seg.x), static_cast<int>(seg.y)) == p) { occupied = true; break; }
                }
                if (occupied) continue;

                glm::vec3 c(float(x), 0.0f, -float(y));   // floor mapping (see convention comment above)
                const bool dark = ((x + y) & 1) != 0;
                // static_cast<uint8_t> silences -Wnarrowing (int expression -> Color channel)
                const shs::Color top_color{ static_cast<uint8_t>(dark ? 48 : 62), static_cast<uint8_t>(dark ? 58 : 72), static_cast<uint8_t>(dark ? 92 : 106), 255 };   // subtle checkerboard
                float tile_height = (difficulty.solid_walls) ? 0.9f : 0.5f;

                add_box(tris, c, glm::vec3(cell * 0.96f, tile_height, cell * 0.96f), top_color, top_color, top_color);
            }
        }

        // 2. Snake body — gradient extruded prisms head→tail; head is a slightly taller block.
        const size_t n = snap.body.position.size();
        for (size_t i = 0; i < n; ++i) {
            glm::vec3 seg(snap.body.position[i].x, 0.0f, -snap.body.position[i].y);
            float t = static_cast<float>(i) / static_cast<float>(std::max<size_t>(1, n - 1));   // 0 at head, 1 at tail
            shs::Color color{ static_cast<uint8_t>(20 + 45 * t), static_cast<uint8_t>(180 - 90 * t), static_cast<uint8_t>(170 - 130 * t), 255 };

            float height = (i == 0) ? 0.6f : 0.45f;
            add_box(tris, seg, glm::vec3(cell * 0.84f, height, cell * 0.84f), color, color, color);
        }

        // 3. Food — glowing berry approximated as a small box sitting ON TOP of the tiles.
        // Lifted so its top (y=0.9) clearly clears the tile tops (y=0.25), and depth_bias is 0:
        // the old +0.06 bias EXCEEDED the tiny NDC depth gap between the food's top face and the
        // tile beneath it, so the food lost every depth test against its own tile and rendered
        // INVISIBLE (verified: zero orange pixels in the frame dump). No bias needed once lifted.
        if (snap.food.pos.x >= 0) {
            glm::vec3 c(float(snap.food.pos.x), 0.45f, -float(snap.food.pos.y));
            add_box(tris, c, glm::vec3(cell * 0.8f, 0.9f, cell * 0.8f), shs::Color{ 255, 180, 140, 255 }, shs::Color{ 150, 40, 30, 255 }, shs::Color{ 150, 40, 30, 255 }, 0.0f);
        }

        // NOTE: death shatter FX is emitted by the main entry edge on the alive→dead transition
        // (SnakeSnapshot carries no alive flag by design; the reducer reports it via SnakeStepResult).

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