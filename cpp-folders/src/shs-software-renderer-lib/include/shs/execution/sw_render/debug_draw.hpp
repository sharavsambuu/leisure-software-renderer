#pragma once

#include <vector>
#include <cmath>
#include <algorithm>

#include <glm/glm.hpp>

#include "shs/gfx/rt_types.hpp"
#include "shs/geometry/jolt_debug_draw.hpp"

namespace shs
{
namespace debug_draw
{

inline void draw_line_rt(RT_ColorLDR& rt, int x0, int y0, int x1, int y1, Color c)
{
    int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;
    while (true)
    {
        if (x0 >= 0 && x0 < rt.w && y0 >= 0 && y0 < rt.h)
        {
            rt.set_rgba(x0, y0, c.r, c.g, c.b, c.a);
        }
        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

inline float edge_fn(const glm::vec2& a, const glm::vec2& b, const glm::vec2& p)
{
    return (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);
}

inline bool project_world_to_screen(
    const glm::vec3& world,
    const glm::mat4& vp,
    int canvas_w,
    int canvas_h,
    glm::vec2& out_xy,
    float& out_depth)
{
    const glm::vec4 clip = vp * glm::vec4(world, 1.0f);
    if (clip.w <= 0.001f) return false;
    const glm::vec3 ndc = glm::vec3(clip) / clip.w;
    if (ndc.z < -1.0f || ndc.z > 1.0f) return false;

    out_xy = glm::vec2(
        (ndc.x + 1.0f) * 0.5f * static_cast<float>(canvas_w),
        (ndc.y + 1.0f) * 0.5f * static_cast<float>(canvas_h));
    out_depth = ndc.z * 0.5f + 0.5f;
    return true;
}

inline void draw_filled_triangle(
    RT_ColorLDR& rt,
    std::span<float> depth_buffer,
    const glm::vec2& p0, float z0,
    const glm::vec2& p1, float z1,
    const glm::vec2& p2, float z2,
    Color c)
{
    const float area = edge_fn(p0, p1, p2);
    if (std::abs(area) <= 1e-6f) return;

    const float min_xf = std::min(p0.x, std::min(p1.x, p2.x));
    const float min_yf = std::min(p0.y, std::min(p1.y, p2.y));
    const float max_xf = std::max(p0.x, std::max(p1.x, p2.x));
    const float max_yf = std::max(p0.y, std::max(p1.y, p2.y));

    const int min_x = std::max(0, static_cast<int>(std::floor(min_xf)));
    const int min_y = std::max(0, static_cast<int>(std::floor(min_yf)));
    const int max_x = std::min(rt.w - 1, static_cast<int>(std::ceil(max_xf)));
    const int max_y = std::min(rt.h - 1, static_cast<int>(std::ceil(max_yf)));
    if (min_x > max_x || min_y > max_y) return;

    const bool ccw = area > 0.0f;
    for (int y = min_y; y <= max_y; ++y)
    {
        for (int x = min_x; x <= max_x; ++x)
        {
            const glm::vec2 p(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f);
            const float w0 = edge_fn(p1, p2, p);
            const float w1 = edge_fn(p2, p0, p);
            const float w2 = edge_fn(p0, p1, p);
            const bool inside = ccw ? (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f)
                                    : (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f);
            if (!inside) continue;

            const float iw0 = w0 / area;
            const float iw1 = w1 / area;
            const float iw2 = w2 / area;
            const float depth = iw0 * z0 + iw1 * z1 + iw2 * z2;
            if (depth < 0.0f || depth > 1.0f) continue;

            const size_t di = static_cast<size_t>(y) * static_cast<size_t>(rt.w) + static_cast<size_t>(x);
            if (depth < depth_buffer[di])
            {
                depth_buffer[di] = depth;
                rt.set_rgba(x, y, c.r, c.g, c.b, c.a);
            }
        }
    }
}

inline void draw_debug_mesh_wireframe_transformed(
    RT_ColorLDR& rt,
    const DebugMesh& mesh_local,
    const glm::mat4& model,
    const glm::mat4& vp,
    int canvas_w,
    int canvas_h,
    Color line_color)
{
    std::vector<glm::ivec2> projected(mesh_local.vertices.size(), glm::ivec2(-1, -1));
    for (size_t i = 0; i < mesh_local.vertices.size(); ++i)
    {
        const glm::vec3 world = glm::vec3(model * glm::vec4(mesh_local.vertices[i], 1.0f));
        glm::vec2 s{};
        float z = 1.0f;
        if (!project_world_to_screen(world, vp, canvas_w, canvas_h, s, z)) continue;
        projected[i] = glm::ivec2(static_cast<int>(s.x), static_cast<int>(s.y));
    }

    for (size_t i = 0; i + 2 < mesh_local.indices.size(); i += 3)
    {
        const uint32_t i0 = mesh_local.indices[i + 0];
        const uint32_t i1 = mesh_local.indices[i + 1];
        const uint32_t i2 = mesh_local.indices[i + 2];
        if (i0 >= projected.size() || i1 >= projected.size() || i2 >= projected.size()) continue;

        const glm::ivec2 v0 = projected[i0];
        const glm::ivec2 v1 = projected[i1];
        const glm::ivec2 v2 = projected[i2];

        if (v0.x >= 0 && v1.x >= 0) draw_line_rt(rt, v0.x, v0.y, v1.x, v1.y, line_color);
        if (v1.x >= 0 && v2.x >= 0) draw_line_rt(rt, v1.x, v1.y, v2.x, v2.y, line_color);
        if (v2.x >= 0 && v0.x >= 0) draw_line_rt(rt, v2.x, v2.y, v0.x, v0.y, line_color);
    }
}

inline void draw_mesh_blinn_phong_transformed(
    RT_ColorLDR& rt,
    std::span<float> depth_buffer,
    const DebugMesh& mesh_local,
    const glm::mat4& model,
    const glm::mat4& vp,
    int canvas_w,
    int canvas_h,
    const glm::vec3& camera_pos,
    const glm::vec3& light_dir_ws,
    const glm::vec3& base_color)
{
    const glm::vec3 L = glm::normalize(-light_dir_ws);
    for (size_t i = 0; i + 2 < mesh_local.indices.size(); i += 3)
    {
        const glm::vec3 lp0 = mesh_local.vertices[mesh_local.indices[i + 0]];
        const glm::vec3 lp1 = mesh_local.vertices[mesh_local.indices[i + 1]];
        const glm::vec3 lp2 = mesh_local.vertices[mesh_local.indices[i + 2]];

        const glm::vec3 p0 = glm::vec3(model * glm::vec4(lp0, 1.0f));
        const glm::vec3 p1 = glm::vec3(model * glm::vec4(lp1, 1.0f));
        const glm::vec3 p2 = glm::vec3(model * glm::vec4(lp2, 1.0f));

        glm::vec2 s0, s1, s2;
        float z0 = 1.0f, z1 = 1.0f, z2 = 1.0f;
        if (!project_world_to_screen(p0, vp, canvas_w, canvas_h, s0, z0)) continue;
        if (!project_world_to_screen(p1, vp, canvas_w, canvas_h, s1, z1)) continue;
        if (!project_world_to_screen(p2, vp, canvas_w, canvas_h, s2, z2)) continue;

        // Mesh winding follows LH + clockwise front faces, so flip RH cross order.
        glm::vec3 n = glm::cross(p2 - p0, p1 - p0);
        const float n2 = glm::dot(n, n);
        if (n2 <= 1e-10f) continue;
        n = glm::normalize(n);

        const glm::vec3 centroid = (p0 + p1 + p2) * (1.0f / 3.0f);
        const glm::vec3 V = glm::normalize(camera_pos - centroid);
        const glm::vec3 H = glm::normalize(L + V);

        const float ndotl = std::max(0.0f, glm::dot(n, L));
        const float ndoth = std::max(0.0f, glm::dot(n, H));
        const float ambient = 0.18f;
        const float diffuse = 0.72f * ndotl;
        const float specular = (ndotl > 0.0f) ? (0.35f * std::pow(ndoth, 32.0f)) : 0.0f;

        glm::vec3 lit = base_color * (ambient + diffuse) + glm::vec3(specular);
        lit = glm::clamp(lit, glm::vec3(0.0f), glm::vec3(1.0f));
        const Color c{
            static_cast<uint8_t>(std::clamp(lit.r * 255.0f, 0.0f, 255.0f)),
            static_cast<uint8_t>(std::clamp(lit.g * 255.0f, 0.0f, 255.0f)),
            static_cast<uint8_t>(std::clamp(lit.b * 255.0f, 0.0f, 255.0f)),
            255
        };

        draw_filled_triangle(rt, depth_buffer, s0, z0, s1, z1, s2, z2, c);
    }
}

} // namespace debug_draw
} // namespace shs
