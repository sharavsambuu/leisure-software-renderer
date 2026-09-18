// tier1 demo 08 — normal mapping: flat vs perturbed (rung 8 pilot).
// Full-viewport quad, left half shaded with the geometric normal, right half
// with a tangent-space procedural bump map. Vertex normals ride COLOR0;
// the tangent frame is constructed per pixel (T = normalize(cross(up, N))),
// the same construction the *_vk Slang twin runs. Constants (light,
// albedo, ambient, bump strength) are literals here AND in the shader so
// both sides evaluate identical math; only sampler precision may drift.
//
// Run: t1_normal_mapping_sw [out.png]

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "../common/adventures_frame.hpp"
#include "../common/adventures_window.hpp"
#include "../common/adventures_sw_raster.hpp"
#include "../common/t1_scenes.hpp"

using namespace adventures;

namespace
{
    constexpr float k_bump_strength = 0.08f;
    const glm::vec3 k_light   = glm::normalize(glm::vec3(-0.35f, 0.5f, 0.8f));
    const glm::vec3 k_albedo  = glm::vec3(0.85f, 0.87f, 0.90f);
    constexpr float k_ambient = 0.08f;

    struct Tex2D
    {
        int w = 0;
        int h = 0;
        std::vector<uint8_t> rgba{};
    };

    // Procedural sine-bump normal map; analytic derivatives, encoded RGB.
    // The _vk twin generates bit-identical texels (same formula, same order).
    Tex2D make_bump_normals(int size = 16)
    {
        Tex2D t{ size, size, std::vector<uint8_t>(size_t(size) * size * 4, 255) };
        constexpr float k_two_pi = 6.283185307179586f;
        for (int y = 0; y < size; ++y)
        {
            for (int x = 0; x < size; ++x)
            {
                const float u = (float(x) + 0.5f) / float(size);
                const float v = (float(y) + 0.5f) / float(size);
                const float dhdu = 4.0f * k_two_pi * 0.5f * std::cos(2.0f * k_two_pi * u) * std::cos(2.0f * k_two_pi * v);
                const float dhdv = -4.0f * k_two_pi * 0.5f * std::sin(2.0f * k_two_pi * u) * std::sin(2.0f * k_two_pi * v);
                const glm::vec3 n = glm::normalize(glm::vec3(-k_bump_strength * dhdu, -k_bump_strength * dhdv, 1.0f));
                const size_t i = (size_t(y) * size + x) * 4;
                t.rgba[i]     = uint8_t(n.x * 127.5f + 127.5f);
                t.rgba[i + 1] = uint8_t(n.y * 127.5f + 127.5f);
                t.rgba[i + 2] = uint8_t(n.z * 127.5f + 127.5f);
            }
        }
        return t;
    }

    glm::vec4 texel_fetch(const Tex2D& t, int x, int y)
    {
        x = ((x % t.w) + t.w) % t.w; // REPEAT
        y = ((y % t.h) + t.h) % t.h;
        const size_t i = (size_t(y) * t.w + x) * 4;
        return { t.rgba[i] / 255.0f, t.rgba[i + 1] / 255.0f, t.rgba[i + 2] / 255.0f, 1.0f };
    }

    glm::vec4 sample_bilinear(const Tex2D& t, const glm::vec2& uv)
    {
        const float fx = uv.x * t.w - 0.5f;
        const float fy = uv.y * t.h - 0.5f;
        const int   x0 = int(std::floor(fx));
        const int   y0 = int(std::floor(fy));
        const float tx = fx - float(x0);
        const float ty = fy - float(y0);
        return glm::mix(glm::mix(texel_fetch(t, x0, y0), texel_fetch(t, x0 + 1, y0), tx),
                        glm::mix(texel_fetch(t, x0, y0 + 1), texel_fetch(t, x0 + 1, y0 + 1), tx), ty);
    }

    // Tangent frame from the (interpolated) geometric normal — the rung-8 operator.
    inline void tangent_frame(const glm::vec3& n, glm::vec3& t, glm::vec3& b)
    {
        t = glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), n));
        b = glm::cross(n, t);
    }

    void draw_normal_mapped(SwRaster& r, const T1Vertex& v0, const T1Vertex& v1, const T1Vertex& v2,
                            bool use_nmap, const Tex2D& bump)
    {
        auto to_px = [&](const T1Vertex& v)
        {
            return glm::vec2((v.pos[0] * 0.5f + 0.5f) * float(r.frame.width),
                             (1.0f - (v.pos[1] * 0.5f + 0.5f)) * float(r.frame.height));
        };
        const glm::vec2 a = to_px(v0);
        const glm::vec2 b = to_px(v1);
        const glm::vec2 c = to_px(v2);
        const float area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
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
                if (e0 < 0.0f || e1 < 0.0f || e2 < 0.0f) continue;

                const float w0 = -e1 / area;
                const float w1 = -e2 / area;
                const float w2 = -e0 / area;
                const glm::vec2 uv = w0 * glm::vec2(v0.uv[0], v0.uv[1]) + w1 * glm::vec2(v1.uv[0], v1.uv[1]) + w2 * glm::vec2(v2.uv[0], v2.uv[1]);
                glm::vec3 n = w0 * glm::vec3(v0.nrm[0], v0.nrm[1], v0.nrm[2])
                            + w1 * glm::vec3(v1.nrm[0], v1.nrm[1], v1.nrm[2])
                            + w2 * glm::vec3(v2.nrm[0], v2.nrm[1], v2.nrm[2]);
                n = glm::normalize(n);

                glm::vec3 shade_n = n;
                if (use_nmap)
                {
                    glm::vec3 t, bb;
                    tangent_frame(n, t, bb);
                    const glm::vec4 tex = sample_bilinear(bump, uv);
                    const glm::vec3 n_t = glm::vec3(tex.r, tex.g, tex.b) * 2.0f - glm::vec3(1.0f);
                    shade_n = glm::normalize(t * n_t.x + bb * n_t.y + n * n_t.z);
                }

                const float  diff = glm::max(glm::dot(shade_n, k_light), 0.0f);
                const glm::vec3 col = k_albedo * diff + glm::vec3(k_ambient);
                r.frame.put(x, y, uint8_t(col.r * 255.0f + 0.5f), uint8_t(col.g * 255.0f + 0.5f), uint8_t(col.b * 255.0f + 0.5f));
            }
        }
    }
} // namespace

int main(int argc, char* argv[])
{
    const DemoArgs args = parse_demo_args(argc, argv, "t1_08_normal_mapping_sw.png");
    const std::string out_path = args.out_path;

    Frame frame(640, 480);
    frame.clear(12, 12, 16); // shared clear convention (matches _vk harness)
    SwRaster raster(frame);

    const Tex2D bump = make_bump_normals();
    const std::vector<T1Vertex> quads = scene_nmap_quads(); // 6 left (flat) + 6 right (mapped)
    for (size_t i = 0; i + 2 < quads.size(); i += 3)
    {
        draw_normal_mapped(raster, quads[i], quads[i + 1], quads[i + 2], /*use_nmap=*/ i >= 6, bump);
    }

    if (!frame.save_png(out_path))
    {
        std::fprintf(stderr, "failed to write %s\n", out_path.c_str());
        return 1;
    }
    std::printf("wrote %s (%dx%d) — left: flat, right: normal-mapped\n", out_path.c_str(), frame.width, frame.height);
    if (args.windowed)
    {
        present_frame_windowed(frame, "t1 08 — normal mapping (software)", out_path, args.backend);
    }
    return 0;
}
