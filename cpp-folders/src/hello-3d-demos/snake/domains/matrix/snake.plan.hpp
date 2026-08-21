#pragma once

#include <vector>
#include <memory_resource>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "shs_renderer.hpp"
#include "snake.contract.hpp"
#include "../config/levels/snake_level_01.hpp"

// matrix pod — pure 3D batch & scene planner. Mirrors tetris::plan_tetris_scene().
// Renders the snake as a low-poly tube ribbon (one box per segment, oriented along heading),
// plus food and motion-trail particles. Camera + Lambert shading reused verbatim from tetris.

namespace snake::matrix {

    struct LowPolyTriangle {
        glm::vec3  p0, p1, p2;
        shs::Color color;
        float      depth_bias = 0.0f;
        LowPolyTriangle(glm::vec3 a, glm::vec3 b, glm::vec3 c, shs::Color col, float bias = 0.0f)
            : p0(a), p1(b), p2(c), color(col), depth_bias(bias) {}
    };

    struct ShatterParticleSoA {
        std::pmr::vector<glm::vec3> position;   // world-space particle positions (consumed here)
        std::pmr::vector<float>     life;       // remaining lifetime 0..1
        explicit ShatterParticleSoA(std::pmr::memory_resource* mr) : position(mr), life(mr) {}
    };

    struct PipelineExecutionPlan {
        std::pmr::vector<ProcessedTriangle> triangles;
        glm::mat4                           view_matrix;
        glm::mat4                           proj_matrix;
        glm::mat4                           vp_matrix;
        explicit PipelineExecutionPlan(std::pmr::memory_resource* mr) : triangles(mr) {}
    };

    namespace MeshGen {
        static inline void add_quad(
            std::vector<LowPolyTriangle>& tris,
            glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, glm::vec3 v3,
            shs::Color col, float bias = 0.0f) {
            tris.emplace_back(v0, v1, v2, col, bias);
            tris.emplace_back(v0, v2, v3, col, bias);
        }

        static inline void add_box(
            std::vector<LowPolyTriangle>& tris,
            glm::vec3 center, glm::vec3 size, shs::Color c_top, shs::Color c_side, shs::Color c_bot, float bias = 0.0f) {
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

        static inline void add_sphere(
            std::vector<LowPolyTriangle>& tris,
            glm::vec3 center, float radius, shs::Color col, float bias = 0.0f) {
            // Low-poly sphere: hardcoded regular icosahedron (12 vertices + 20 faces).
            static const float R = 1.618033988749895f;   // golden ratio
            glm::vec3 V[12] = {
                {-R,  R,  0}, { R,  R,  0}, { R, -R,  0}, {-R, -R,  0},
                { 0, -R,  R}, { 0, -R, -R}, { 0,  R,  R}, { 0,  R, -R},
                { R,  0,  R}, { R,  0, -R}, {-R,  0,  R}, {-R,  0, -R}
            };
            static const int F[20][3] = {
                {0,8,3},{0,4,8},{0,2,4},{0,1,2},{1,9,2},
                {1,11,9},{2,10,11},{2,5,10},{2,3,5},{3,7,5},
                {3,8,7},{3,6,7},{4,6,5},{4,9,6},{8,9,4},
                {5,10,11},{9,11,10},{11,8,9},{10,7,11},{7,12,8},
                {3,12,13},{3,13,2},{6,14,5},{6,1,14},{1,14,9},
                {14,10,1},{14,11,10},{13,7,12},{13,12,3},{8,3,13}
            };
            for (int f = 0; f < 20; ++f) {
                glm::vec3 a = center + V[F[f][0]];
                glm::vec3 b = center + V[F[f][1]];
                glm::vec3 c = center + V[F[f][2]];
                tris.emplace_back(a, b, c, col, bias);
            }
        }
    }

    struct ProcessedTriangle {
        glm::vec4  c0, c1, c2;
        shs::Color lit_color;
        float      depth_bias;
    };

    // PURE batch renderer — builds world -> draw tokens.
    static inline PipelineExecutionPlan plan_snake_scene(
        const SnakeSnapshot&       world,
        const ShatterParticleSoA&   particles,
        int                         canvas_w,
        int                         canvas_h,
        float                       camera_shake,
        std::pmr::memory_resource*  arena)
    {
        PipelineExecutionPlan plan(arena);
        plan.triangles.reserve(4000);

        // Camera: elevated angle looking down the grid (reused from tetris).
        glm::vec3 eye = glm::vec3(0.0f, 12.0f, -20.0f);
        if (camera_shake > 0.0f) {
            eye.y -= camera_shake * 0.35f;
            eye.x += (std::sin(world.score.length * 60.0f) * camera_shake * 0.15f);
        }
        glm::vec3 target = world.head.pos;   // look at the head cell
        plan.view_matrix = glm::lookAtLH(eye, target, glm::vec3(0, 1, 0));
        plan.proj_matrix = glm::perspectiveLH_NO(glm::radians(60.0f), (float)canvas_w / (float)canvas_h, 0.15f, 200.0f);
        plan.vp_matrix   = plan.proj_matrix * plan.view_matrix;

        std::vector<LowPolyTriangle> tris;
        tris.reserve(3000);

        glm::vec3 SUN_DIR = glm::normalize(glm::vec3(0.45f, -0.85f, 0.35f));
        glm::vec3 L       = -SUN_DIR;

        // --- Arena shell (backplane + rails) ---
        shs::Color rail_col = shs::Color{ 60, 70, 90, 255 };
        shs::Color trim_col = shs::Color{ 40, 180, 240, 255 };
        shs::Color bg_grid  = shs::Color{ 18, 22, 30, 255 };
        MeshGen::add_box(tris, glm::vec3(0.0f, 9.5f, 0.60f), glm::vec3((float)world.score.length * 0.5f + 1.0f, (float)world.score.length * 0.5f + 2.0f, 0.1f), bg_grid, bg_grid, bg_grid);

        // --- Snake body: tube ribbon, one box per segment oriented along heading ---
        const float seg_size = 0.9f;   // cell footprint in world units
        glm::vec3 dir(0.0f, 1.0f, 0.0f);   // default facing (+Y)
        if (world.head.dir.x != 0) dir = glm::vec3(world.head.dir.x, 0.0f, 0.0f);
        else                      dir = glm::vec3(0.0f, world.head.dir.y, 0.0f);

        // Body color ramps from head (bright) to tail (dim).
        for (size_t i = 0; i < world.body.position.size(); ++i) {
            const auto& cell = world.body.position[i];
            float t = (float)i / std::max(1, (int)world.body.position.size());
            glm::vec3 base((1.0f - t) * 0.9f + 0.1f, ((1.0f - t) * 0.5f + 0.6f), ((1.0f - t) * 0.2f + 0.4f));
            shs::Color col = shs::rgb01_to_color(base);

            // Segment center at the midpoint between this cell and its predecessor (or head).
            glm::vec3 mid((float)cell.x, (float)cell.y, 0.0f);
            if (i == 0) {   // tail-most: anchor near head direction
                mid = world.head.pos;
            } else {
                const auto& prev = world.body.position[i - 1];
                mid = glm::vec3((float)(cell.x + prev.x) * 0.5f, (float)(cell.y + prev.y) * 0.5f, 0.0f);
            }

            // Offset the ribbon slightly toward the head so it reads as a continuous tube.
            glm::vec3 center = mid - dir * 0.15f;
            MeshGen::add_box(tris, center, glm::vec3(seg_size, seg_size, 0.85f), col, col, col);

            // Motion-trail particles along the body path (subtle).
            if ((float)i % 2 == 0) {
                tris.emplace_back(center + dir * 0.4f, center - dir * 0.4f, mid, shs::Color{ 60, 180, 240, 120 }, 0.003f);
            }
        }

        // --- Head: brighter box at head cell (front-biased so it overlaps body) ---
        {
            glm::vec3 center((float)world.head.pos.x, (float)world.head.pos.y, 0.0f);
            shs::Color col = shs::rgb01_to_color(glm::vec3(1.0f, 0.55f, 0.2f));   // amber head
            MeshGen::add_box(tris, center, glm::vec3(seg_size * 1.05f, seg_size * 1.05f, 0.95f), col, col, col, -0.004f);
        }

        // --- Food: glowing sphere at food cell ---
        if (world.food.active) {
            glm::vec3 center((float)world.food.pos.x, (float)world.food.pos.y, 0.0f);
            MeshGen::add_sphere(tris, center, 0.55f, shs::Color{ 240, 80, 60, 255 }, -0.001f);
        }

        // --- Shatter particles (eat/crash bursts) — position + life only; fixed color ---
        for (size_t i = 0; i < particles.position.size(); ++i) {
            if (particles.life[i] > 0.0f) {
                MeshGen::add_box(tris, particles.position[i], glm::vec3(0.24f), shs::Color{ 60, 180, 240, 255 }, shs::Color{ 40, 120, 180, 255 }, particles.life[i] * 0.6f);
            }
        }

        // --- Transform & shade (Lambert) ---
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

            plan.triangles.push_back({ c0, c1, c2, shs::rgb01_to_color(lit_rgb), tri.depth_bias });
        }

        return plan;
    }

} // namespace snake::matrix
