#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: primitives_builders.hpp
    МОДУЛЬ: geometry
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн geometry модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <algorithm>
#include <cmath>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include "shs/geometry/primitives.hpp"
#include "shs/resources/mesh.hpp"

namespace shs
{
    namespace detail
    {
        inline uint32_t add_vertex(MeshData& m, const glm::vec3& p, const glm::vec3& n, const glm::vec2& uv)
        {
            m.positions.push_back(p);
            m.normals.push_back(n);
            m.uvs.push_back(uv);
            return (uint32_t)m.positions.size() - 1;
        }

        inline void add_triangle(MeshData& m, uint32_t a, uint32_t b, uint32_t c)
        {
            m.indices.push_back(a);
            m.indices.push_back(b);
            m.indices.push_back(c);
        }

        inline void add_triangle_match_normals(MeshData& m, uint32_t a, uint32_t b, uint32_t c)
        {
            if (a < m.positions.size() && b < m.positions.size() && c < m.positions.size() &&
                a < m.normals.size() && b < m.normals.size() && c < m.normals.size())
            {
                const glm::vec3 pa = m.positions[a];
                const glm::vec3 pb = m.positions[b];
                const glm::vec3 pc = m.positions[c];
                const glm::vec3 face_n = glm::cross(pb - pa, pc - pa);
                const glm::vec3 avg_n = m.normals[a] + m.normals[b] + m.normals[c];
                if (glm::dot(face_n, avg_n) < 0.0f) std::swap(b, c);
            }
            add_triangle(m, a, b, c);
        }

        inline void add_grid_patch(
            MeshData& m,
            const glm::vec3& origin,
            const glm::vec3& axis_u,
            const glm::vec3& axis_v,
            const glm::vec3& normal,
            int seg_u,
            int seg_v
        )
        {
            seg_u = std::max(1, seg_u);
            seg_v = std::max(1, seg_v);
            const uint32_t base = (uint32_t)m.positions.size();

            for (int y = 0; y <= seg_v; ++y)
            {
                const float fv = (float)y / (float)seg_v;
                for (int x = 0; x <= seg_u; ++x)
                {
                    const float fu = (float)x / (float)seg_u;
                    const glm::vec3 p = origin + axis_u * fu + axis_v * fv;
                    add_vertex(m, p, normal, glm::vec2(fu, fv));
                }
            }

            const int stride = seg_u + 1;
            for (int y = 0; y < seg_v; ++y)
            {
                for (int x = 0; x < seg_u; ++x)
                {
                    const uint32_t i00 = base + (uint32_t)(y * stride + x);
                    const uint32_t i10 = i00 + 1;
                    const uint32_t i01 = i00 + (uint32_t)stride;
                    const uint32_t i11 = i01 + 1;
                    add_triangle_match_normals(m, i00, i01, i10);
                    add_triangle_match_normals(m, i10, i01, i11);
                }
            }
        }
    }

    inline MeshData make_plane(const PlaneDesc& d)
    {
        MeshData m{};
        const int sx = std::max(1, d.seg_x);
        const int sz = std::max(1, d.seg_z);
        const float hw = d.width * 0.5f;
        const float hz = d.depth * 0.5f;
        detail::add_grid_patch(
            m,
            glm::vec3(-hw, 0.0f, -hz),
            glm::vec3(d.width, 0.0f, 0.0f),
            glm::vec3(0.0f, 0.0f, d.depth),
            glm::vec3(0.0f, 1.0f, 0.0f),
            sx,
            sz
        );
        return m;
    }

    inline MeshData make_sphere(const SphereDesc& d)
    {
        MeshData m{};
        const int su = std::max(3, d.seg_u);
        const int sv = std::max(2, d.seg_v);

        for (int y = 0; y <= sv; ++y)
        {
            const float v = (float)y / (float)sv;
            const float phi = v * glm::pi<float>();
            const float sin_phi = std::sin(phi);
            const float cos_phi = std::cos(phi);

            for (int x = 0; x <= su; ++x)
            {
                const float u = (float)x / (float)su;
                const float theta = u * glm::two_pi<float>();
                const float sin_theta = std::sin(theta);
                const float cos_theta = std::cos(theta);

                glm::vec3 n{
                    sin_phi * cos_theta,
                    cos_phi,
                    sin_phi * sin_theta
                };
                n = glm::normalize(n);
                detail::add_vertex(m, n * d.radius, n, glm::vec2(u, v));
            }
        }

        const int stride = su + 1;
        for (int y = 0; y < sv; ++y)
        {
            for (int x = 0; x < su; ++x)
            {
                const uint32_t i00 = (uint32_t)(y * stride + x);
                const uint32_t i10 = i00 + 1;
                const uint32_t i01 = i00 + (uint32_t)stride;
                const uint32_t i11 = i01 + 1;
                detail::add_triangle_match_normals(m, i00, i01, i10);
                detail::add_triangle_match_normals(m, i10, i01, i11);
            }
        }
        return m;
    }

    inline MeshData make_box(const BoxDesc& d)
    {
        MeshData m{};
        const float hx = d.size.x * 0.5f;
        const float hy = d.size.y * 0.5f;
        const float hz = d.size.z * 0.5f;

        // +X / -X
        detail::add_grid_patch(m, { hx, -hy, -hz}, {0, 0, d.size.z}, {0, d.size.y, 0}, { 1, 0, 0}, d.seg_z, d.seg_y);
        detail::add_grid_patch(m, {-hx, -hy,  hz}, {0, 0,-d.size.z}, {0, d.size.y, 0}, {-1, 0, 0}, d.seg_z, d.seg_y);
        // +Y / -Y
        detail::add_grid_patch(m, {-hx,  hy, -hz}, {d.size.x, 0, 0}, {0, 0, d.size.z}, {0, 1, 0}, d.seg_x, d.seg_z);
        detail::add_grid_patch(m, {-hx, -hy,  hz}, {d.size.x, 0, 0}, {0, 0,-d.size.z}, {0,-1, 0}, d.seg_x, d.seg_z);
        // +Z / -Z
        detail::add_grid_patch(m, {-hx, -hy,  hz}, {d.size.x, 0, 0}, {0, d.size.y, 0}, {0, 0, 1}, d.seg_x, d.seg_y);
        detail::add_grid_patch(m, { hx, -hy, -hz}, {-d.size.x,0, 0}, {0, d.size.y, 0}, {0, 0,-1}, d.seg_x, d.seg_y);

        return m;
    }

    inline MeshData make_cone(const ConeDesc& d)
    {
        MeshData m{};
        const int sr = std::max(3, d.seg_radial);
        const int sh = std::max(1, d.seg_height);
        const float h = std::max(1e-6f, d.height);
        const float r = std::max(1e-6f, d.radius);

        // Side
        for (int y = 0; y <= sh; ++y)
        {
            const float v = (float)y / (float)sh;
            const float ry = h * (v - 0.5f);
            const float rr = r * (1.0f - v);
            for (int x = 0; x <= sr; ++x)
            {
                const float u = (float)x / (float)sr;
                const float t = u * glm::two_pi<float>();
                const float ct = std::cos(t);
                const float st = std::sin(t);
                const glm::vec3 p{rr * ct, ry, rr * st};
                glm::vec3 n{ct, r / h, st};
                n = glm::normalize(n);
                detail::add_vertex(m, p, n, glm::vec2(u, v));
            }
        }
        const int stride = sr + 1;
        for (int y = 0; y < sh; ++y)
        {
            for (int x = 0; x < sr; ++x)
            {
                const uint32_t i00 = (uint32_t)(y * stride + x);
                const uint32_t i10 = i00 + 1;
                const uint32_t i01 = i00 + (uint32_t)stride;
                const uint32_t i11 = i01 + 1;
                detail::add_triangle_match_normals(m, i00, i01, i10);
                detail::add_triangle_match_normals(m, i10, i01, i11);
            }
        }

        if (d.cap)
        {
            const uint32_t center = detail::add_vertex(m, {0.0f, -h * 0.5f, 0.0f}, {0, -1, 0}, {0.5f, 0.5f});
            for (int x = 0; x <= sr; ++x)
            {
                const float u = (float)x / (float)sr;
                const float t = u * glm::two_pi<float>();
                const float ct = std::cos(t);
                const float st = std::sin(t);
                const glm::vec3 p{r * ct, -h * 0.5f, r * st};
                detail::add_vertex(m, p, {0, -1, 0}, glm::vec2(0.5f + ct * 0.5f, 0.5f + st * 0.5f));
            }
            const uint32_t rim_base = center + 1;
            for (int x = 0; x < sr; ++x)
            {
                const uint32_t a = rim_base + (uint32_t)x;
                const uint32_t b = rim_base + (uint32_t)x + 1;
                detail::add_triangle_match_normals(m, center, b, a);
            }
        }

        return m;
    }
}
