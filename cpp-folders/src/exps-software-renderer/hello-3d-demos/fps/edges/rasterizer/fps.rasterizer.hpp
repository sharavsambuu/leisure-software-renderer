#pragma once

// ============================================================================
// fps/edges/rasterizer/fps.rasterizer.hpp — TILED MULTITHREADED RASTER EDGE
// (fps::raster) Frustum clip -> screen projection -> barycentric + Z-buffer
// per tile job. Wait-free per tile: each job owns a disjoint screen region.
// ============================================================================

#include <span>
#include <vector>

#include <glm/glm.hpp>

#include "shs_renderer.hpp"

#include <domains/spatial_fx/spatial_fx.contract.hpp>

namespace fps::raster {

    inline glm::vec4 clip_to_screen_vec4(const glm::vec4& clip, int W, int H) {
        const float inv_w = 1.0f / clip.w;
        const glm::vec3 ndc = glm::vec3(clip) * inv_w;

        glm::vec4 s;
        s.x = (ndc.x + 1.0f) * 0.5f * static_cast<float>(W - 1);
        s.y = (1.0f - ndc.y) * 0.5f * static_cast<float>(H - 1);
        s.z = ndc.z;
        s.w = inv_w;
        return s;
    }

    inline void rasterize_perspective_triangle_tile(
        shs::Canvas& canvas, shs::ZBuffer& z_buffer,
        const glm::vec4& sc0, const glm::vec4& sc1, const glm::vec4& sc2,
        shs::Color lit_color, float depth_bias,
        glm::ivec2 tile_min, glm::ivec2 tile_max)
    {
        const glm::vec2 v0(sc0.x, sc0.y);
        const glm::vec2 v1(sc1.x, sc1.y);
        const glm::vec2 v2(sc2.x, sc2.y);

        const float area = (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);
        if (!shs::Raster::is_front_facing_screen(area, shs::Raster::FrontFace::CounterClockwise)) return;

        const glm::vec2 bboxmin = glm::max(glm::vec2(tile_min), glm::min(v0, glm::min(v1, v2)));
        const glm::vec2 bboxmax = glm::min(glm::vec2(tile_max), glm::max(v0, glm::max(v1, v2)));
        if (bboxmin.x > bboxmax.x || bboxmin.y > bboxmax.y) return;

        const std::vector<glm::vec2> v2d = { v0, v1, v2 };
        const int min_x = static_cast<int>(bboxmin.x);
        const int max_x = static_cast<int>(bboxmax.x);
        const int min_y = static_cast<int>(bboxmin.y);
        const int max_y = static_cast<int>(bboxmax.y);

        for (int py = min_y; py <= max_y; ++py) {
            for (int px = min_x; px <= max_x; ++px) {
                const glm::vec3 bc = shs::Canvas::barycentric_coordinate(
                    glm::vec2(static_cast<float>(px) + 0.5f, static_cast<float>(py) + 0.5f), v2d);
                if (bc.x < 0.0f || bc.y < 0.0f || bc.z < 0.0f) continue;

                const float interp_z = shs::Raster::interpolate_ndc_depth(bc, sc0.z, sc1.z, sc2.z);
                const float final_z  = interp_z + depth_bias;
                if (final_z < -1.0f || final_z > 1.0f) continue;

                if (z_buffer.test_and_set_depth_screen_space(px, py, final_z)) {
                    canvas.draw_pixel_screen_space(px, py, lit_color);
                }
            }
        }
    }

    struct TileRasterContract {
        std::span<const spatial_fx::ProcessedTriangle> active_triangles;
        glm::ivec2                                     tile_min;
        glm::ivec2                                     tile_max;
        int                                            canvas_w;
        int                                            canvas_h;
    };

    inline void execute_tile_raster_job(
        shs::Canvas&              canvas,
        shs::ZBuffer&             z_buffer,
        const TileRasterContract& contract
    ) {
        for (const auto& tri : contract.active_triangles) {
            const shs::Raster::FrustumClipPolygon poly =
                shs::Raster::clip_triangle_to_frustum(tri.c0, tri.c1, tri.c2);
            if (poly.count < 3) continue;

            const glm::vec4 s0 = clip_to_screen_vec4(poly.vertices[0], contract.canvas_w, contract.canvas_h);
            for (int i = 1; i + 1 < poly.count; ++i) {
                const glm::vec4 s1 = clip_to_screen_vec4(poly.vertices[static_cast<size_t>(i)],     contract.canvas_w, contract.canvas_h);
                const glm::vec4 s2 = clip_to_screen_vec4(poly.vertices[static_cast<size_t>(i + 1)], contract.canvas_w, contract.canvas_h);
                rasterize_perspective_triangle_tile(canvas, z_buffer, s0, s1, s2,
                                                   tri.lit_color, tri.depth_bias,
                                                   contract.tile_min, contract.tile_max);
            }
        }
    }

} // namespace fps::raster