#pragma once
// tetris/edges/rasterizer/tetris.rasterizer.hpp — TILED RASTER EDGE
// Screen-space helpers (tetris::raster::vop); tile jobs live in main wiring.
#include <glm/glm.hpp>
#include "shs_renderer.hpp"

namespace tetris {
namespace vop {
    static inline glm::vec4 clip_to_screen_vec4(const glm::vec4& clip, int W, int H) {
        float inv_w = 1.0f / clip.w;
        glm::vec3 ndc = glm::vec3(clip) * inv_w;
        return glm::vec4(
            (ndc.x + 1.0f) * 0.5f * (float)(W - 1),
            (1.0f - ndc.y) * 0.5f * (float)(H - 1),
            ndc.z,
            inv_w
        );
    }

    static void rasterize_triangle_tile(
        shs::Canvas& canvas, shs::ZBuffer& z_buffer,
        const glm::vec4& sc0, const glm::vec4& sc1, const glm::vec4& sc2,
        shs::Color lit_color, float depth_bias,
        glm::ivec2 tile_min, glm::ivec2 tile_max
    ) {
        glm::vec2 v0(sc0.x, sc0.y), v1(sc1.x, sc1.y), v2(sc2.x, sc2.y);
        float area = (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);
        if (!shs::Raster::is_front_facing_screen(area, shs::Raster::FrontFace::CounterClockwise)) return;

        glm::vec2 bboxmin = glm::max(glm::vec2(tile_min), glm::min(v0, glm::min(v1, v2)));
        glm::vec2 bboxmax = glm::min(glm::vec2(tile_max), glm::max(v0, glm::max(v1, v2)));
        if (bboxmin.x > bboxmax.x || bboxmin.y > bboxmax.y) return;

        std::vector<glm::vec2> v2d = { v0, v1, v2 };
        int min_x = (int)bboxmin.x, max_x = (int)bboxmax.x;
        int min_y = (int)bboxmin.y, max_y = (int)bboxmax.y;

        for (int py = min_y; py <= max_y; ++py) {
            for (int px = min_x; px <= max_x; ++px) {
                glm::vec3 bc = shs::Canvas::barycentric_coordinate(glm::vec2((float)px + 0.5f, (float)py + 0.5f), v2d);
                if (bc.x < 0.0f || bc.y < 0.0f || bc.z < 0.0f) continue;

                float interp_z = shs::Raster::interpolate_ndc_depth(bc, sc0.z, sc1.z, sc2.z);
                float final_z  = interp_z + depth_bias;
                if (final_z < -1.0f || final_z > 1.0f) continue;

                if (z_buffer.test_and_set_depth_screen_space(px, py, final_z)) {
                    canvas.draw_pixel_screen_space(px, py, lit_color);
                }
            }
        }
    }
}
} // namespace tetris
