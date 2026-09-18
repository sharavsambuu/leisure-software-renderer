// tier0 demo 04 — texture sampling: NEAREST vs BILINEAR + scissor test.
// A procedural checkerboard texture is sampled with uv tiled 8x (uv > 1
// exercises the REPEAT wrap mode). Left half samples NEAREST, right half
// BILINEAR — same texel data, same uv (the *_vk twin binds two sampler
// states on the same image). A horizontal scissor band restricts drawing
// to the bottom rows, mirroring vkCmdSetScissor behavior.
//
// Run: t0_texture_sampling_sw [out.png]

#include <cmath>
#include <cstdio>
#include <string>

#include <glm/glm.hpp>

#include "../common/adventures_frame.hpp"
#include "../common/adventures_window.hpp"
#include "../common/adventures_sw_raster.hpp"
#include "../common/t0_scenes.hpp"

using namespace adventures;

namespace
{
    struct Tex2D
    {
        int w = 0;
        int h = 0;
        std::vector<uint8_t> rgba{};
    };

    // 16x16 checkerboard with a colored cross marker (makes filtering visible)
    Tex2D make_checker(int size = 16)
    {
        Tex2D t{ size, size, std::vector<uint8_t>(size_t(size) * size * 4, 255) };
        for (int y = 0; y < size; ++y)
        {
            for (int x = 0; x < size; ++x)
            {
                const bool even = ((x / (size / 2)) + (y / (size / 2))) % 2 == 0;
                uint8_t    r    = even ? 230 : 40;
                uint8_t    g    = even ? 230 : 40;
                uint8_t    b    = even ? 230 : 220;
                if (x == size / 2 || y == size / 2) { r = 255; g = 60; b = 60; } // marker
                const size_t i = (size_t(y) * size + x) * 4;
                t.rgba[i]     = r;
                t.rgba[i + 1] = g;
                t.rgba[i + 2] = b;
            }
        }
        return t;
    }

    glm::vec4 texel_fetch(const Tex2D& t, int x, int y)
    {
        // REPEAT wrap (GL_REPEAT / VK_SAMPLER_ADDRESS_MODE_REPEAT)
        x = ((x % t.w) + t.w) % t.w;
        y = ((y % t.h) + t.h) % t.h;
        const size_t i = (size_t(y) * t.w + x) * 4;
        return { t.rgba[i] / 255.0f, t.rgba[i + 1] / 255.0f, t.rgba[i + 2] / 255.0f, 1.0f };
    }

    glm::vec4 sample_tex(const Tex2D& t, const glm::vec2& uv, bool bilinear)
    {
        const float fx = uv.x * t.w - 0.5f;
        const float fy = uv.y * t.h - 0.5f;
        if (!bilinear)
        {
            // NEAREST: round to closest texel center
            return texel_fetch(t, int(std::floor(fx + 0.5f)), int(std::floor(fy + 0.5f)));
        }
        // BILINEAR: weighted mix of the 4 surrounding texel centers
        const int       x0     = int(std::floor(fx));
        const int       y0     = int(std::floor(fy));
        const float     tx     = fx - float(x0);
        const float     ty     = fy - float(y0);
        const glm::vec4 c00    = texel_fetch(t, x0, y0);
        const glm::vec4 c10    = texel_fetch(t, x0 + 1, y0);
        const glm::vec4 c01    = texel_fetch(t, x0, y0 + 1);
        const glm::vec4 c11    = texel_fetch(t, x0 + 1, y0 + 1);
        const glm::vec4 top    = glm::mix(c00, c10, tx);
        const glm::vec4 bottom = glm::mix(c01, c11, tx);
        return glm::mix(top, bottom, ty);
    }

    // triangle with per-pixel uv interpolation + filtering + scissor
    void draw_textured_triangle(SwRaster& r, const PassPolicy& policy,
                                const T0Vertex& v0, const T0Vertex& v1, const T0Vertex& v2,
                                bool bilinear, const Tex2D& tex)
    {
        auto to_px = [&](const T0Vertex& v)
        {
            return glm::vec2((v.pos[0] * 0.5f + 0.5f) * float(r.frame.width),
                             (1.0f - (v.pos[1] * 0.5f + 0.5f)) * float(r.frame.height));
        };
        const glm::vec2 a    = to_px(v0);
        const glm::vec2 b    = to_px(v1);
        const glm::vec2 c    = to_px(v2);
        const float     area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
        if (area == 0.0f) return;

        const int min_x = std::max(0, int(std::min({ a.x, b.x, c.x })));
        const int max_x = std::min(r.frame.width - 1, int(std::max({ a.x, b.x, c.x })));
        const int min_y = std::max(0, int(std::min({ a.y, b.y, c.y })));
        const int max_y = std::min(r.frame.height - 1, int(std::max({ a.y, b.y, c.y })));

        for (int y = min_y; y <= max_y; ++y)
        {
            for (int x = min_x; x <= max_x; ++x)
            {
                const glm::vec2 p(float(x) + 0.5f, float(y) + 0.5f);
                const float e0 = (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);
                const float e1 = (p.x - b.x) * (c.y - b.y) - (p.y - b.y) * (c.x - b.x);
                const float e2 = (p.x - c.x) * (a.y - c.y) - (p.y - c.y) * (a.x - c.x);
                if (e0 < 0.0f || e1 < 0.0f || e2 < 0.0f || !scissor_allows(policy, x, y)) continue;

                // Barycentric weights. The e* edge functions above are defined as
                // cross(p−v_i, v_j−v_i) = −E_ij, so the true barycentrics are
                // w_a = E_bc/area = −e1/area, etc. (negation also makes this valid
                // for either winding, matching the _vk twin's cullMode=NONE).
                // Getting the sign wrong mirrors the interpolated uv through the
                // origin and, with REPEAT wrap, samples a phase-shifted texture.
                const float     w0 = -e1 / area;
                const float     w1 = -e2 / area;
                const float     w2 = -e0 / area;
                const glm::vec2 uv = w0 * glm::vec2(v0.uv[0], v0.uv[1]) + w1 * glm::vec2(v1.uv[0], v1.uv[1]) + w2 * glm::vec2(v2.uv[0], v2.uv[1]);
                const glm::vec4 s  = sample_tex(tex, uv, bilinear);
                r.frame.put(x, y, uint8_t(s.r * 255.0f), uint8_t(s.g * 255.0f), uint8_t(s.b * 255.0f));
            }
        }
    }
}

int main(int argc, char* argv[])
{
    const DemoArgs args = parse_demo_args(argc, argv, "t0_04_texture_sampling_sw.png");
    const std::string out_path = args.out_path;

    Frame frame(640, 480);
    frame.clear(12, 12, 16);
    SwRaster raster(frame);
    const Tex2D tex = make_checker();

    const std::vector<T0Vertex> quads = scene_texture_quads(); // 6 left + 6 right verts

    // AD2: the same policy object the *_vk twin hands to add_pipeline/render.
    // Scissor band rows 300..479, x1/y1 exclusive — the *_vk twin derives its
    // dynamic VkRect2D from this rectangle instead of spelling pixels twice.
    PassPolicy policy{};
    policy.scissor_enabled = true;
    policy.scissor         = ScissorRect{ 0, 300, frame.width, frame.height };

    for (int half = 0; half < 2; ++half)
    {
        const bool bilinear = half == 1;
        for (int t = 0; t < 2; ++t)
        {
            draw_textured_triangle(raster, policy, quads[half * 6 + t * 3 + 0], quads[half * 6 + t * 3 + 1],
                                   quads[half * 6 + t * 3 + 2], bilinear, tex);
        }
    }

    if (!frame.save_png(out_path))
    {
        std::fprintf(stderr, "failed to write %s\n", out_path.c_str());
        return 1;
    }
    std::printf("wrote %s (%dx%d) — left: NEAREST, right: BILINEAR; scissor band rows 300+\n",
                out_path.c_str(), frame.width, frame.height);
    if (args.windowed)
    {
        present_frame_windowed(frame, "t0 04 — texture sampling + scissor (software)", out_path, args.backend);
    }
    return 0;
}
