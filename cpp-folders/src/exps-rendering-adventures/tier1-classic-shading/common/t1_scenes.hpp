#pragma once

/*
    tier1 common — rung-08 scenes (normal mapping pilot).
    T1Vertex carries a real normal channel (pos + nrm + uv); the _vk twin
    reuses T0Vertex with the normal packed into COLOR0.xyz (see below), so
    both sides shade identical inputs through the shared OffscreenVulkan
    harness (vertex layout unchanged — no harness fork for the pilot).
*/

#include <vector>

#include <glm/glm.hpp>

#include "../../tier0-rasterization-foundations/common/t0_scenes.hpp"

namespace adventures
{
    struct T1Vertex
    {
        float pos[3];
        float nrm[3];
        float uv[2];
    };

    inline T1Vertex t1_vertex(const glm::vec3& p, const glm::vec3& n, const glm::vec2& uv)
    {
        return T1Vertex{ { p.x, p.y, p.z }, { n.x, n.y, n.z }, { uv.x, uv.y } };
    }

    // Full-viewport quad pair: left NDC x[-1,0] (flat), right x[0,1] (mapped).
    // uv tiles 0..4 across the width (REPEAT); normals face +Z everywhere.
    inline std::vector<T1Vertex> scene_nmap_quads()
    {
        const glm::vec3 n(0.0f, 0.0f, 1.0f);
        const glm::vec3 a(-1.0f, -1.0f, 0.5f), b(0.0f, -1.0f, 0.5f), c(0.0f, 1.0f, 0.5f), d(-1.0f, 1.0f, 0.5f);
        const glm::vec3 e(0.0f, -1.0f, 0.5f), f(1.0f, -1.0f, 0.5f), g(1.0f, 1.0f, 0.5f), h(0.0f, 1.0f, 0.5f);
        const glm::vec2 uv_a(0.0f, 0.0f), uv_b(2.0f, 0.0f), uv_c(2.0f, 1.0f), uv_d(0.0f, 1.0f);
        const glm::vec2 uv_e(2.0f, 0.0f), uv_f(4.0f, 0.0f), uv_g(4.0f, 1.0f), uv_h(2.0f, 1.0f);
        return {
            t1_vertex(a, n, uv_a), t1_vertex(b, n, uv_b), t1_vertex(c, n, uv_c),
            t1_vertex(a, n, uv_a), t1_vertex(c, n, uv_c), t1_vertex(d, n, uv_d),
            t1_vertex(e, n, uv_e), t1_vertex(f, n, uv_f), t1_vertex(g, n, uv_g),
            t1_vertex(e, n, uv_e), t1_vertex(g, n, uv_g), t1_vertex(h, n, uv_h),
        };
    }

    // Same geometry as T0Vertex (normal in COLOR0.xyz, alpha 1) for the _vk twin.
    inline std::vector<T0Vertex> scene_nmap_quads_t0()
    {
        const std::vector<T1Vertex> src = scene_nmap_quads();
        std::vector<T0Vertex> out;
        out.reserve(src.size());
        for (const auto& v : src)
        {
            out.push_back(t0_vertex(glm::vec3(v.pos[0], v.pos[1], v.pos[2]),
                                    glm::vec4(v.nrm[0], v.nrm[1], v.nrm[2], 1.0f),
                                    glm::vec2(v.uv[0], v.uv[1])));
        }
        return out;
    }
} // namespace adventures
