#pragma once

// ============================================================================
// fps/domains/spatial_fx/spatial_fx.contract.hpp — RENDER VOCABULARY
// LowPolyTriangle + MeshBuilder (world-space mesh authoring) and the
// render-ready PipelineExecutionPlan consumed by the rasterizer edge.
// Pure data; no SDL.
// ============================================================================

#include <memory_resource>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include "shs_renderer.hpp"

namespace fps::spatial_fx {

    // ------------------------------------------------------------------
    // World-space mesh vocabulary (authored at startup, transformed per frame)
    // ------------------------------------------------------------------
    struct LowPolyTriangle {
        glm::vec3  p0;
        glm::vec3  p1;
        glm::vec3  p2;
        shs::Color color;
        float      depth_bias;

        LowPolyTriangle(glm::vec3 a, glm::vec3 b, glm::vec3 c, shs::Color col, float bias = 0.0f)
            : p0(a), p1(b), p2(c), color(col), depth_bias(bias) {
        }
    };

    namespace MeshBuilder {
        inline void add_quad(std::vector<LowPolyTriangle>& tris,
                             glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, glm::vec3 v3,
                             shs::Color c, float bias = 0.0f) {
            tris.emplace_back(v0, v1, v2, c, bias);
            tris.emplace_back(v0, v2, v3, c, bias);
        }

        inline void add_box(std::vector<LowPolyTriangle>& tris,
                            glm::vec3 center, glm::vec3 size,
                            shs::Color c_top, shs::Color c_side, shs::Color c_bot,
                            float bias = 0.0f) {
            const glm::vec3 h = size * 0.5f;
            const glm::vec3 p000 = center + glm::vec3(-h.x, -h.y, -h.z);
            const glm::vec3 p100 = center + glm::vec3( h.x, -h.y, -h.z);
            const glm::vec3 p110 = center + glm::vec3( h.x,  h.y, -h.z);
            const glm::vec3 p010 = center + glm::vec3(-h.x,  h.y, -h.z);
            const glm::vec3 p001 = center + glm::vec3(-h.x, -h.y,  h.z);
            const glm::vec3 p101 = center + glm::vec3( h.x, -h.y,  h.z);
            const glm::vec3 p111 = center + glm::vec3( h.x,  h.y,  h.z);
            const glm::vec3 p011 = center + glm::vec3(-h.x,  h.y,  h.z);

            add_quad(tris, p001, p101, p111, p011, c_side, bias);
            add_quad(tris, p100, p000, p010, p110, c_side, bias);
            add_quad(tris, p010, p011, p111, p110, c_top , bias);
            add_quad(tris, p000, p100, p101, p001, c_bot , bias);
            add_quad(tris, p100, p110, p111, p101, c_side, bias);
            add_quad(tris, p000, p001, p011, p010, c_side, bias);
        }

        inline void add_cylinder(std::vector<LowPolyTriangle>& tris,
                                 glm::vec3 base_center, float radius, float height, int segments,
                                 shs::Color color) {
            const glm::vec3 top_center = base_center + glm::vec3(0, height, 0);
            const float     step       = glm::two_pi<float>() / static_cast<float>(segments);

            for (int i = 0; i < segments; ++i) {
                const float a0 = static_cast<float>(i) * step;
                const float a1 = static_cast<float>(i + 1) * step;

                const glm::vec3 b0 = base_center + glm::vec3(std::cos(a0) * radius, 0.0f, std::sin(a0) * radius);
                const glm::vec3 b1 = base_center + glm::vec3(std::cos(a1) * radius, 0.0f, std::sin(a1) * radius);
                const glm::vec3 t0 = b0 + glm::vec3(0, height, 0);
                const glm::vec3 t1 = b1 + glm::vec3(0, height, 0);

                tris.emplace_back(b0, t0, t1, color, 0.0f);
                tris.emplace_back(b0, t1, b1, color, 0.0f);

                tris.emplace_back(top_center, t1, t0, color, 0.0f);
                tris.emplace_back(base_center, b0, b1, color, 0.0f);
            }
        }
    } // namespace MeshBuilder

    // ------------------------------------------------------------------
    // Render-ready plan (consumed by edges/rasterizer)
    // ------------------------------------------------------------------
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
            : triangles(mr) {
        }

        PipelineExecutionPlan(PipelineExecutionPlan&&) noexcept            = default;
        PipelineExecutionPlan& operator=(PipelineExecutionPlan&&) noexcept = default;
    };

} // namespace fps::spatial_fx