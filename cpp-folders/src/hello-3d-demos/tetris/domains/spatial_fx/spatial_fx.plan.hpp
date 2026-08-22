#pragma once
// tetris/domains/spatial_fx/spatial_fx.plan.hpp — PURE SCENE PLANNER (tetris::spatial_fx)
// Reads the matrix contract READ-ONLY (same privilege model as the fps demo)
// and batches lit triangles into the frame's PipelineExecutionPlan.
#include <memory_resource>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "shs_renderer.hpp"

#include <domains/matrix/matrix.contract.hpp>
#include <domains/matrix/matrix.reducer.hpp>
#include <domains/spatial_fx/spatial_fx.contract.hpp>

namespace tetris::spatial_fx {
using tetris::matrix::MatrixSnapshot;
using tetris::matrix::get_ghost_y;
using tetris::matrix::get_piece_blocks;
    // PURE 3D BATCH & SCENE PLANNER
    static inline PipelineExecutionPlan plan_tetris_scene(
        const MatrixSnapshot&       m,
        const FxState&              fx,
        int                         canvas_w,
        int                         canvas_h,
        std::pmr::memory_resource*  arena
    ) {
        PipelineExecutionPlan plan(arena);
        plan.triangles.reserve(4000);

        // Camera stationed with proper vertical clearance for the bottom row.
        // camera_pulse = dolly-in zoom punch (tetris clears / victory crescendo).
        glm::vec3 eye = glm::vec3(0.0f, 10.6f, -18.4f + fx.camera_pulse * 1.4f);
        if (fx.camera_shake > 0.0f) {
            eye.y -= fx.camera_shake * 0.35f;
            eye.x += (std::sin(fx.time * 60.0f) * fx.camera_shake * 0.15f);
        }

        glm::vec3 target = glm::vec3(0.0f, 9.6f, 0.0f);
        plan.view_matrix = glm::lookAtLH(eye, target, glm::vec3(0, 1, 0));
        plan.proj_matrix = glm::perspectiveLH_NO(glm::radians(60.0f), (float)canvas_w / (float)canvas_h, 0.15f, 150.0f);
        plan.vp_matrix   = plan.proj_matrix * plan.view_matrix;

        std::vector<LowPolyTriangle> tris;
        tris.reserve(3000);

        glm::vec3 SUN_DIR = glm::normalize(glm::vec3(0.45f, -0.85f, 0.35f));
        glm::vec3 L       = -SUN_DIR;

        // 1. PLAYFIELD MATRIX WELL CONTAINER
        // Environment mood (pod-5 embryo): trim lerps cyan → amber as the blitz
        // clock drains (fx.mood_intensity is a plain value wired by main).
        // L3 canyon (fx.env_dusk): surfaces shift to dusk/desert sandstone.
        const float dusk = glm::clamp(fx.env_dusk, 0.0f, 1.0f);
        shs::Color rail_col = lerp_color(shs::Color{ 60,  70,  90, 255 },
                                         shs::Color{ 96,  66,  44, 255 }, dusk);
        shs::Color trim_col = lerp_color(
            lerp_color(shs::Color{ 40, 180, 240, 255 },
                       shs::Color{ 255, 160, 40, 255 },
                       fx.mood_intensity),
            shs::Color{ 255, 150, 60, 255 }, dusk * 0.8f);
        shs::Color bg_grid  = lerp_color(shs::Color{ 18,  22,  30, 255 },
                                         shs::Color{ 36,  26,  26, 255 }, dusk);

        // Backplane
        MeshGen::add_box(tris, glm::vec3(0.0f, 9.5f, 0.60f), glm::vec3(10.2f, 20.2f, 0.1f), bg_grid, bg_grid, bg_grid);

        // Left, Right, and Bottom Rails
        MeshGen::add_box(tris, glm::vec3(-5.35f,  9.5f, 0.0f), glm::vec3(0.5f, 20.4f, 1.1f), trim_col, rail_col, rail_col);
        MeshGen::add_box(tris, glm::vec3( 5.35f,  9.5f, 0.0f), glm::vec3(0.5f, 20.4f, 1.1f), trim_col, rail_col, rail_col);
        MeshGen::add_box(tris, glm::vec3(  0.0f, -0.7f, 0.0f), glm::vec3(11.2f, 0.5f, 1.1f), trim_col, rail_col, rail_col);

        // Pedestal Floor (sandstone in the canyon)
        shs::Color floor_top  = lerp_color(shs::Color{ 25, 30, 42, 255 },
                                           shs::Color{ 104, 78, 50, 255 }, dusk);
        shs::Color floor_side = lerp_color(shs::Color{ 14, 16, 22, 255 },
                                           shs::Color{ 58, 42, 30, 255 }, dusk);
        MeshGen::add_box(tris, glm::vec3(0.0f, -1.2f, 1.0f), glm::vec3(26.0f, 0.6f, 14.0f), floor_top, floor_side, floor_side);

        // L3 canyon diorama embryo: mesa silhouettes + flickering torches.
        // Gated on env_dusk so every other stage renders pixel-identical.
        if (dusk > 0.5f) {
            const shs::Color mesa_far  { 46, 32, 36, 255 };
            const shs::Color mesa_near { 58, 38, 38, 255 };
            MeshGen::add_box(tris, glm::vec3(-17.0f,  5.0f, -10.0f), glm::vec3(13.0f, 16.0f, 2.5f), mesa_far,  mesa_far,  mesa_far);
            MeshGen::add_box(tris, glm::vec3( 17.0f,  7.0f, -10.0f), glm::vec3(11.0f, 20.0f, 2.5f), mesa_far,  mesa_far,  mesa_far);
            MeshGen::add_box(tris, glm::vec3( -6.0f,  2.0f, -13.0f), glm::vec3(9.0f,  9.0f, 2.0f),  mesa_near, mesa_near, mesa_near);
            MeshGen::add_box(tris, glm::vec3(  7.0f,  3.0f, -13.0f), glm::vec3(7.0f, 11.0f, 2.0f),  mesa_near, mesa_near, mesa_near);

            // Torch flames on the well rails — deterministic flicker from fx.time.
            const float flick = 0.75f + 0.18f * std::sin(fx.time * 11.3f)
                                      + 0.07f * std::sin(fx.time * 23.7f);
            const shs::Color flame{ (uint8_t)(255), (uint8_t)(150 * flick + 40), 40, 255 };
            for (const float tx : { -5.35f, 5.35f }) {
                MeshGen::add_box(tris, glm::vec3(tx, 19.9f, 0.55f),
                                 glm::vec3(0.34f, 0.62f * flick, 0.34f), flame, flame, flame, -0.004f);
                MeshGen::add_box(tris, glm::vec3(tx, 19.45f, 0.55f),
                                 glm::vec3(0.22f, 0.28f, 0.22f), shs::Color{ 70, 52, 40, 255 },
                                 shs::Color{ 70, 52, 40, 255 }, shs::Color{ 70, 52, 40, 255 });
            }
        }

        // 2. RESTING MATRIX VOXEL BLOCKS
        float block_size = CELL_SIZE - BLOCK_GAP;
        for (int y = 0; y < VISIBLE_H; ++y) {
            for (int x = 0; x < GRID_W; ++x) {
                uint8_t cell = m.grid[y][x];
                if (cell != 0) {
                    shs::Color col = get_piece_color(static_cast<PieceType>(cell));
                    glm::vec3 center((float)x - 4.5f, (float)y, 0.0f);
                    MeshGen::add_box(tris, center, glm::vec3(block_size, block_size, 0.85f), col, col, col);
                }
            }
        }

        // 3. REAL-TIME GHOST PIECE PROJECTION
        if (m.active.type != PieceType::None && !m.game_over) {
            int ghost_y = get_ghost_y(m.grid, m.active);
            auto blocks = get_piece_blocks(m.active.type, m.active.rotation);
            shs::Color ghost_col{ 50, 60, 80, 255 };

            for (const auto& b : blocks) {
                int gx = m.active.pos.x + b.x;
                int gy = ghost_y + b.y;
                if (gy < VISIBLE_H) {
                    glm::vec3 center((float)gx - 4.5f, (float)gy, 0.0f);
                    MeshGen::add_box(tris, center, glm::vec3(block_size * 0.96f, block_size * 0.96f, 0.40f), ghost_col, ghost_col, ghost_col, 0.002f);
                }
            }
        }

        // 4. ACTIVE FALLING TETROMINO
        if (m.active.type != PieceType::None && !m.game_over) {
            shs::Color active_col = get_piece_color(m.active.type);
            auto blocks = get_piece_blocks(m.active.type, m.active.rotation);

            for (const auto& b : blocks) {
                int gx = m.active.pos.x + b.x;
                int gy = m.active.pos.y + b.y;
                if (gy < VISIBLE_H) {
                    glm::vec3 center((float)gx - 4.5f, (float)gy, 0.0f);
                    MeshGen::add_box(tris, center, glm::vec3(block_size, block_size, 0.92f), active_col, active_col, active_col, -0.001f);
                }
            }
        }

        // 5. 3D FLOATING HOLD & NEXT QUEUE PODS (Always render platforms)
        auto add_preview_piece = [&](PieceType type, glm::vec3 pod_center) {
            // Hovering pedestal disc is ALWAYS drawn
            MeshGen::add_box(tris, pod_center - glm::vec3(0, 1.2f, 0), glm::vec3(3.8f, 0.25f, 3.8f), trim_col, rail_col, rail_col);

            if (type == PieceType::None) return; // Skip piece blocks if empty
            shs::Color col = get_piece_color(type);
            auto blocks = get_piece_blocks(type, 0);

            for (const auto& b : blocks) {
                glm::vec3 bp = pod_center + glm::vec3((float)b.x * 0.7f - 0.35f, (float)b.y * 0.7f, 0.0f);
                MeshGen::add_box(tris, bp, glm::vec3(0.62f, 0.62f, 0.62f), col, col, col);
            }
            };

        // Left Pod: HOLD (Always visible)
        add_preview_piece(m.hold_piece, glm::vec3(-8.6f, 15.5f, 0.5f));

        // Right Pods: NEXT QUEUE (Top 3)
        add_preview_piece(m.next_queue[0], glm::vec3(8.6f, 16.0f, 0.5f));
        add_preview_piece(m.next_queue[1], glm::vec3(8.6f, 11.8f, 0.5f));
        add_preview_piece(m.next_queue[2], glm::vec3(8.6f,  7.6f, 0.5f));

        // 6. 3D SHATTER VOXEL PARTICLES
        for (size_t i = 0; i < fx.particles.position.size(); ++i) {
            if (fx.particles.life[i] > 0.0f) {
                MeshGen::add_box(tris, fx.particles.position[i], glm::vec3(0.24f), fx.particles.color[i], fx.particles.color[i], fx.particles.color[i]);
            }
        }

        // 7. SHOCKWAVE RINGS (event-fed: blitz clock ticks) — expanding circle
        // of voxel segments in the board plane, fading as life drains.
        for (size_t i = 0; i < fx.rings.center.size(); ++i) {
            const float fade = fx.rings.life[i] / fx.rings.max_life[i];
            const shs::Color rc = fade_color(fx.rings.color[i], 0.25f + 0.75f * fade);
            const float seg_box = 0.14f + 0.12f * fade;
            constexpr int SEGS = 26;
            for (int sgi = 0; sgi < SEGS; ++sgi) {
                const float ang = (float)sgi / (float)SEGS * glm::two_pi<float>();
                const glm::vec3 p = fx.rings.center[i]
                    + glm::vec3(std::cos(ang) * fx.rings.radius[i],
                                std::sin(ang) * fx.rings.radius[i], 0.0f);
                MeshGen::add_box(tris, p, glm::vec3(seg_box), rc, rc, rc);
            }
        }

        // Transform and Shade Triangles
        for (const auto& tri : tris) {
            glm::vec4 c0 = plan.vp_matrix * glm::vec4(tri.p0, 1.0f);
            glm::vec4 c1 = plan.vp_matrix * glm::vec4(tri.p1, 1.0f);
            glm::vec4 c2 = plan.vp_matrix * glm::vec4(tri.p2, 1.0f);

            glm::vec3 N = glm::cross(tri.p1 - tri.p0, tri.p2 - tri.p0);
            float len = glm::length(N);
            if (len < 1e-6f) continue;
            N /= len;

            float NdotL = std::max(0.0f, glm::dot(N, L));
            float diffuse = NdotL * 0.70f + 0.30f;
            float ambient = std::max(0.0f, N.y) * 0.20f + 0.15f;

            glm::vec3 base_col = glm::vec3(tri.color.r, tri.color.g, tri.color.b) / 255.0f;
            // Dusk light tint: warm key light + ember ambient in the canyon.
            const glm::vec3 key_tint   = glm::mix(glm::vec3(1.00f, 0.98f, 0.92f), glm::vec3(1.06f, 0.88f, 0.68f), dusk);
            const glm::vec3 amb_tint   = glm::mix(glm::vec3(0.50f, 0.70f, 1.00f), glm::vec3(0.85f, 0.55f, 0.35f), dusk);
            glm::vec3 lit_rgb = base_col * (diffuse * key_tint + ambient * amb_tint);

            plan.triangles.push_back({
                c0, c1, c2,
                shs::rgb01_to_color(lit_rgb),
                tri.depth_bias
                });
        }

        return plan;
    }
} // namespace tetris::spatial_fx
