#pragma once

#include <vector>
#include <memory_resource>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "shs_renderer.hpp"
#include "tetris.contract.hpp"

namespace tetris {

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

    // PURE 3D BATCH & SCENE PLANNER
    static inline PipelineExecutionPlan plan_tetris_scene(
        const TetrisSnapshot&       world,
        const ShatterParticleSoA&   particles,
        int                         canvas_w,
        int                         canvas_h,
        float                       camera_shake,
        std::pmr::memory_resource*  arena
    ) {
        PipelineExecutionPlan plan(arena);
        plan.triangles.reserve(4000);

        // Camera stationed with proper vertical clearance for the bottom row
        glm::vec3 eye = glm::vec3(0.0f, 10.6f, -18.4f);
        if (camera_shake > 0.0f) {
            eye.y -= camera_shake * 0.35f;
            eye.x += (std::sin(world.game_time * 60.0f) * camera_shake * 0.15f);
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
        shs::Color rail_col = shs::Color{ 60,  70,  90, 255 };
        shs::Color trim_col = shs::Color{ 40, 180, 240, 255 };
        shs::Color bg_grid  = shs::Color{ 18,  22,  30, 255 };

        // Backplane
        MeshGen::add_box(tris, glm::vec3(0.0f, 9.5f, 0.60f), glm::vec3(10.2f, 20.2f, 0.1f), bg_grid, bg_grid, bg_grid);

        // Left, Right, and Bottom Rails
        MeshGen::add_box(tris, glm::vec3(-5.35f,  9.5f, 0.0f), glm::vec3(0.5f, 20.4f, 1.1f), trim_col, rail_col, rail_col);
        MeshGen::add_box(tris, glm::vec3( 5.35f,  9.5f, 0.0f), glm::vec3(0.5f, 20.4f, 1.1f), trim_col, rail_col, rail_col);
        MeshGen::add_box(tris, glm::vec3(  0.0f, -0.7f, 0.0f), glm::vec3(11.2f, 0.5f, 1.1f), trim_col, rail_col, rail_col);

        // Pedestal Floor
        shs::Color floor_top = shs::Color{ 25, 30, 42, 255 };
        shs::Color floor_side = shs::Color{ 14, 16, 22, 255 };
        MeshGen::add_box(tris, glm::vec3(0.0f, -1.2f, 1.0f), glm::vec3(26.0f, 0.6f, 14.0f), floor_top, floor_side, floor_side);

        // 2. RESTING MATRIX VOXEL BLOCKS
        float block_size = CELL_SIZE - BLOCK_GAP;
        for (int y = 0; y < VISIBLE_H; ++y) {
            for (int x = 0; x < GRID_W; ++x) {
                uint8_t cell = world.grid[y][x];
                if (cell != 0) {
                    shs::Color col = get_piece_color(static_cast<PieceType>(cell));
                    glm::vec3 center((float)x - 4.5f, (float)y, 0.0f);
                    MeshGen::add_box(tris, center, glm::vec3(block_size, block_size, 0.85f), col, col, col);
                }
            }
        }

        // 3. REAL-TIME GHOST PIECE PROJECTION
        if (world.active.type != PieceType::None && !world.game_over) {
            int ghost_y = get_ghost_y(world.grid, world.active);
            auto blocks = get_piece_blocks(world.active.type, world.active.rotation);
            shs::Color ghost_col{ 50, 60, 80, 255 };

            for (const auto& b : blocks) {
                int gx = world.active.pos.x + b.x;
                int gy = ghost_y + b.y;
                if (gy < VISIBLE_H) {
                    glm::vec3 center((float)gx - 4.5f, (float)gy, 0.0f);
                    MeshGen::add_box(tris, center, glm::vec3(block_size * 0.96f, block_size * 0.96f, 0.40f), ghost_col, ghost_col, ghost_col, 0.002f);
                }
            }
        }

        // 4. ACTIVE FALLING TETROMINO
        if (world.active.type != PieceType::None && !world.game_over) {
            shs::Color active_col = get_piece_color(world.active.type);
            auto blocks = get_piece_blocks(world.active.type, world.active.rotation);

            for (const auto& b : blocks) {
                int gx = world.active.pos.x + b.x;
                int gy = world.active.pos.y + b.y;
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
        add_preview_piece(world.hold_piece, glm::vec3(-8.6f, 15.5f, 0.5f));

        // Right Pods: NEXT QUEUE (Top 3)
        add_preview_piece(world.next_queue[0], glm::vec3(8.6f, 16.0f, 0.5f));
        add_preview_piece(world.next_queue[1], glm::vec3(8.6f, 11.8f, 0.5f));
        add_preview_piece(world.next_queue[2], glm::vec3(8.6f,  7.6f, 0.5f));

        // 6. 3D SHATTER VOXEL PARTICLES
        for (size_t i = 0; i < particles.position.size(); ++i) {
            if (particles.life[i] > 0.0f) {
                MeshGen::add_box(tris, particles.position[i], glm::vec3(0.24f), particles.color[i], particles.color[i], particles.color[i]);
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
            glm::vec3 lit_rgb = base_col * (diffuse * glm::vec3(1.0f, 0.98f, 0.92f) + ambient * glm::vec3(0.50f, 0.70f, 1.0f));

            plan.triangles.push_back({
                c0, c1, c2,
                shs::rgb01_to_color(lit_rgb),
                tri.depth_bias
                });
        }

        return plan;
    }

} // namespace tetris