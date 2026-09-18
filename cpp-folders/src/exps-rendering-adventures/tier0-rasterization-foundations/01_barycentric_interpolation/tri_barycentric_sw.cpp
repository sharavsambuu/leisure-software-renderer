// tier0 demo 01 — triangle rasterization with barycentric varying interpolation.
// This is the from-scratch lesson: coverage via edge functions, barycentric
// weights as normalized sub-areas, varying (color) interpolation. The *_vk
// twin lets the GPU rasterizer do the same math; compare the PNGs.
//
// Run: t0_tri_barycentric_sw [out.png]

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>

#include "../common/adventures_frame.hpp"
#include "../common/adventures_window.hpp"
#include "../common/t0_scenes.hpp"

using namespace adventures;

// Signed edge function AB x AP: >0 when p is left of AB (CCW convention).
static float edge_function(const glm::vec2& a, const glm::vec2& b, const glm::vec2& p)
{
    return (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);
}

static void rasterize_triangle_barycentric(Frame& frame, const T0Vertex& v0, const T0Vertex& v1, const T0Vertex& v2)
{
    // clip -> NDC -> pixels (y flip: framebuffer origin is top-left)
    auto to_px = [&](const T0Vertex& v)
    {
        return glm::vec2((v.pos[0] * 0.5f + 0.5f) * float(frame.width),
                         (1.0f - (v.pos[1] * 0.5f + 0.5f)) * float(frame.height));
    };
    const glm::vec2 a = to_px(v0);
    const glm::vec2 b = to_px(v1);
    const glm::vec2 c = to_px(v2);

    const float area = edge_function(a, b, c);
    if (area == 0.0f) return;

    const int min_x = std::max(0, int(std::min({ a.x, b.x, c.x })));
    const int max_x = std::min(frame.width - 1, int(std::max({ a.x, b.x, c.x })));
    const int min_y = std::max(0, int(std::min({ a.y, b.y, c.y })));
    const int max_y = std::min(frame.height - 1, int(std::max({ a.y, b.y, c.y })));

    long covered = 0;
    for (int y = min_y; y <= max_y; ++y)
    {
        for (int x = min_x; x <= max_x; ++x)
        {
            const glm::vec2 p(float(x) + 0.5f, float(y) + 0.5f);
            const float e0 = edge_function(a, b, p);
            const float e1 = edge_function(b, c, p);
            const float e2 = edge_function(c, a, p);
            if (e0 < 0.0f || e1 < 0.0f || e2 < 0.0f) continue; // inside = all edges >= 0

            // barycentric weights: each is the sub-triangle area over the total
            const float w0 = e1 / area; // weight of v0
            const float w1 = e2 / area; // weight of v1
            const float w2 = e0 / area; // weight of v2

            // varying interpolation — the whole point of this demo
            const float r = w0 * v0.col[0] + w1 * v1.col[0] + w2 * v2.col[0];
            const float g = w0 * v0.col[1] + w1 * v1.col[1] + w2 * v2.col[1];
            const float b = w0 * v0.col[2] + w1 * v1.col[2] + w2 * v2.col[2];

            frame.put(x, y,
                      uint8_t(r * 255.0f + 0.5f),
                      uint8_t(g * 255.0f + 0.5f),
                      uint8_t(b * 255.0f + 0.5f));
            ++covered;
        }
    }
    std::printf("covered pixels: %ld (triangle area %.1f px)\n", covered, area * 0.25f * 0.25f);
}

int main(int argc, char* argv[])
{
    const DemoArgs args = parse_demo_args(argc, argv, "t0_01_barycentric_sw.png");
    const std::string out_path = args.out_path;

    Frame frame(640, 480);
    frame.clear(12, 12, 16);  // must match the shared _vk harness clear (adventures_vk.cpp)

    const std::vector<T0Vertex> tri = scene_tri_barycentric();
    rasterize_triangle_barycentric(frame, tri[0], tri[1], tri[2]);

    // sanity probe: centroid must be the exact color average (0.5, 0.5, 0.5)-ish
    const glm::vec2 centroid_px = glm::vec2(
        (tri[0].pos[0] + tri[1].pos[0] + tri[2].pos[0]) / 3.0f * 0.5f + 0.5f,
        1.0f - ((tri[0].pos[1] + tri[1].pos[1] + tri[2].pos[1]) / 3.0f * 0.5f + 0.5f));
    const int cx = int(centroid_px.x * frame.width);
    const int cy = int(centroid_px.y * frame.height);
    float r, g, b, a;
    frame.get(cx, cy, r, g, b, a);
    std::printf("centroid sample @(%d,%d) = (%.3f, %.3f, %.3f) — expect ~(0.483, 0.517, 0.467)\n", cx, cy, r, g, b);

    if (!frame.save_png(out_path))
    {
        std::fprintf(stderr, "failed to write %s\n", out_path.c_str());
        return 1;
    }
    std::printf("wrote %s (%dx%d)\n", out_path.c_str(), frame.width, frame.height);
    if (args.windowed)
    {
        present_frame_windowed(frame, "t0 01 — barycentric interpolation (software)", out_path, args.backend);
    }
    return 0;
}
