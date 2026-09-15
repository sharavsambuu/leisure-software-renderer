#pragma once

/*
    tier0 common (software side) — mini rasterizer state shared by the *_sw
    demos (demo 01 hand-rolls its own loop, since implementing barycentric
    coverage IS that demo's lesson).

    Mirrors the fixed-function state the *_vk twins configure through
    VkPipeline state: depth test/write, straight-alpha blending, stencil
    write/test, scissor.
*/

#include <algorithm>
#include <utility>
#include <vector>

#include "adventures_frame.hpp"
#include "t0_scenes.hpp"

namespace adventures
{
    struct SwState
    {
        bool    depth_test    = true;  // compare fragment z (0..1) against z-buffer
        bool    depth_write   = true;
        bool    blend         = false; // straight-alpha source-over
        bool    stencil_write = false; // set stencil = stencil_ref on covered pixels
        bool    stencil_test  = false; // keep only pixels passing the stencil test
        bool    stencil_invert = false;
        int     scissor[4]    = { -1, -1, -1, -1 }; // x0, y0, x1(exclusive), y1(exclusive)
        uint8_t stencil_ref   = 1;
    };

    class SwRaster
    {
    public:
        Frame& frame;
        std::vector<float> depth{};
        std::vector<uint8_t> stencil{};
        SwState state{};

        explicit SwRaster(Frame& f) : frame(f)
        {
            depth.assign(size_t(f.width) * size_t(f.height), 1.0f);
            stencil.assign(size_t(f.width) * size_t(f.height), 0u);
        }

        bool scissor_allows(int x, int y) const
        {
            if (state.scissor[0] < 0) return true;
            return x >= state.scissor[0] && x < state.scissor[2] &&
                   y >= state.scissor[1] && y < state.scissor[3];
        }
    };

    // Clip space -> pixel space. Y is flipped (framebuffer origin top-left,
    // mirroring the negative-viewport-height Vulkan pin); Z stays in [0, 1].
    inline glm::vec3 to_screen(const SwRaster& r, const glm::vec4& clip)
    {
        const float w     = (std::abs(clip.w) < 1e-8f) ? 1.0f : clip.w;
        const float ndc_x = clip.x / w;
        const float ndc_y = clip.y / w;
        return glm::vec3(
            (ndc_x * 0.5f + 0.5f) * float(r.frame.width),
            (1.0f - (ndc_y * 0.5f + 0.5f)) * float(r.frame.height),
            clip.z / w);
    }

    // Signed edge function: >0 when p is left of AB (CCW convention).
    inline float edge_function(const glm::vec2& a, const glm::vec2& b, const glm::vec2& p)
    {
        return (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);
    }

    // Top-left fill rule (mirrors Vulkan's CCW front-face bias rules):
    // a zero-area edge is "inside" only along top or left edges, so shared
    // triangle edges never double-draw.
    inline bool is_top_left(const glm::vec2& a, const glm::vec2& b)
    {
        const glm::vec2 e = b - a;
        if (e.y < 0.0f) return true;              // pointing down => left edge
        return e.y == 0.0f && e.x < 0.0f;         // horizontal pointing left => top edge
    }

    inline void draw_triangle(SwRaster& r, const T0Vertex& v0, const T0Vertex& v1, const T0Vertex& v2)
    {
        const int       w  = r.frame.width;
        const int       h  = r.frame.height;
        const glm::vec3 p0 = to_screen(r, glm::vec4(glm::vec3(v0.pos[0], v0.pos[1], v0.pos[2]), 1.0f));
        glm::vec3       p1 = to_screen(r, glm::vec4(glm::vec3(v1.pos[0], v1.pos[1], v1.pos[2]), 1.0f));
        glm::vec3       p2 = to_screen(r, glm::vec4(glm::vec3(v2.pos[0], v2.pos[1], v2.pos[2]), 1.0f));

        float area = edge_function(glm::vec2(p0), glm::vec2(p1), glm::vec2(p2));
        if (area == 0.0f) return; // degenerate

        // Winding-agnostic rasterization (mirrors the *_vk twin's cullMode=NONE):
        // a clockwise (negative signed area) triangle would never satisfy the
        // CCW inside test below and would silently vanish (back faces of the
        // projection cube rendered as background). Flip CCW instead of culling;
        // the top-left bias on the reversed edges selects the same geometric
        // top/left edges Vulkan biases for the original CW winding.
        const T0Vertex* a_v = &v0;
        const T0Vertex* b_v = &v1;
        const T0Vertex* c_v = &v2;
        if (area < 0.0f)
        {
            std::swap(p1, p2);
            std::swap(b_v, c_v);
            area = -area;
        }

        const glm::vec2 a(p0.x, p0.y), b(p1.x, p1.y), c(p2.x, p2.y);
        const int min_x = std::max(0, int(std::min({ a.x, b.x, c.x })));
        const int max_x = std::min(w - 1, int(std::max({ a.x, b.x, c.x })));
        const int min_y = std::max(0, int(std::min({ a.y, b.y, c.y })));
        const int max_y = std::min(h - 1, int(std::max({ a.y, b.y, c.y })));

        const SwState st  = r.state;
        const bool    tl0 = is_top_left(a, b);
        const bool    tl1 = is_top_left(b, c);
        const bool    tl2 = is_top_left(c, a);

        for (int y = min_y; y <= max_y; ++y)
        {
            for (int x = min_x; x <= max_x; ++x)
            {
                const glm::vec2 p(float(x) + 0.5f, float(y) + 0.5f); // pixel-center sampling
                const float e0 = edge_function(a, b, p);
                const float e1 = edge_function(b, c, p);
                const float e2 = edge_function(c, a, p);

                const bool inside = (e0 > 0.0f || (e0 == 0.0f && tl0)) &&
                                    (e1 > 0.0f || (e1 == 0.0f && tl1)) &&
                                    (e2 > 0.0f || (e2 == 0.0f && tl2));
                if (!inside || !r.scissor_allows(x, y)) continue;

                // barycentric weights = normalized sub-edge areas
                const float inv_area = 1.0f / area;
                const float w0       = e1 * inv_area;
                const float w1       = e2 * inv_area;
                const float w2       = e0 * inv_area;

                const float  depth = w0 * p0.z + w1 * p1.z + w2 * p2.z;
                const size_t idx   = size_t(y) * size_t(w) + size_t(x);

                if (st.depth_test && depth >= r.depth[idx]) continue;
                if (st.stencil_test)
                {
                    const bool match = st.stencil_invert
                                           ? (r.stencil[idx] != st.stencil_ref)
                                           : (r.stencil[idx] == st.stencil_ref);
                    if (!match) continue;
                }

                // varying interpolation (affine; constant w here so this equals
                // what the GPU's perspective-correct path produces for this scene)
                float cr = w0 * a_v->col[0] + w1 * b_v->col[0] + w2 * c_v->col[0];
                float cg = w0 * a_v->col[1] + w1 * b_v->col[1] + w2 * c_v->col[1];
                float cb = w0 * a_v->col[2] + w1 * b_v->col[2] + w2 * c_v->col[2];
                float ca = w0 * a_v->col[3] + w1 * b_v->col[3] + w2 * c_v->col[3];

                if (st.blend)
                {
                    float dr, dg, db, da;
                    r.frame.get(x, y, dr, dg, db, da);
                    const float out_a = ca + da * (1.0f - ca);
                    if (out_a > 0.0f)
                    {
                        cr = (cr * ca + dr * da * (1.0f - ca)) / out_a;
                        cg = (cg * ca + dg * da * (1.0f - ca)) / out_a;
                        cb = (cb * ca + db * da * (1.0f - ca)) / out_a;
                    }
                    ca = out_a;
                }

                if (st.depth_write) r.depth[idx] = depth;
                if (st.stencil_write) r.stencil[idx] = st.stencil_ref;

                r.frame.put(x, y,
                            uint8_t(cr * 255.0f + 0.5f),
                            uint8_t(cg * 255.0f + 0.5f),
                            uint8_t(cb * 255.0f + 0.5f),
                            uint8_t(ca * 255.0f + 0.5f));
            }
        }
    }

    inline void draw_triangles(SwRaster& r, const std::vector<T0Vertex>& verts)
    {
        for (size_t i = 0; i + 2 < verts.size(); i += 3)
        {
            draw_triangle(r, verts[i], verts[i + 1], verts[i + 2]);
        }
    }
}
