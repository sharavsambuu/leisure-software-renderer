#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: frustum_culling.hpp
    МОДУЛЬ: geometry
    ЗОРИЛГО: Матрицаас харагдацын пирамидын 6 гадаргууг (Frustum extraction) гаргаж авах 
            болон бөмбөрцөг/хайрцгийг шалгаж ялгах функцүүд.
*/

#include <algorithm>
#include <array>
#include <cmath>

#include <glm/glm.hpp>

#include "shs/geometry/volumes.hpp"

namespace shs
{
    enum class FrustumPlane : uint32_t
    {
        Left = 0,
        Right = 1,
        Bottom = 2,
        Top = 3,
        Near = 4,
        Far = 5
    };

    inline Plane make_plane_from_vec4(const glm::vec4& eq)
    {
        Plane p{};
        const glm::vec3 n(eq.x, eq.y, eq.z);
        const float len = glm::length(n);
        if (len <= 1e-8f)
        {
            p.normal = glm::vec3(0.0f, 1.0f, 0.0f);
            p.d = eq.w;
            return p;
        }
        p.normal = n / len;
        p.d = eq.w / len;
        return p;
    }

    inline Frustum extract_frustum_planes(const glm::mat4& view_proj)
    {
        // GLM матрицын баганаар унших бүтэц (column-major):
        // Хэтийн төлөвийн орон зайн (clip-space) мөрүүдийн нийлбэрээр гадаргууг гаргаж авна.
        const glm::vec4 r0(view_proj[0][0], view_proj[1][0], view_proj[2][0], view_proj[3][0]);
        const glm::vec4 r1(view_proj[0][1], view_proj[1][1], view_proj[2][1], view_proj[3][1]);
        const glm::vec4 r2(view_proj[0][2], view_proj[1][2], view_proj[2][2], view_proj[3][2]);
        const glm::vec4 r3(view_proj[0][3], view_proj[1][3], view_proj[2][3], view_proj[3][3]);

        Frustum f{};
        f.planes[static_cast<size_t>(FrustumPlane::Left)] = make_plane_from_vec4(r3 + r0);
        f.planes[static_cast<size_t>(FrustumPlane::Right)] = make_plane_from_vec4(r3 - r0);
        f.planes[static_cast<size_t>(FrustumPlane::Bottom)] = make_plane_from_vec4(r3 + r1);
        f.planes[static_cast<size_t>(FrustumPlane::Top)] = make_plane_from_vec4(r3 - r1);
        f.planes[static_cast<size_t>(FrustumPlane::Near)] = make_plane_from_vec4(r3 + r2);
        f.planes[static_cast<size_t>(FrustumPlane::Far)] = make_plane_from_vec4(r3 - r2);
        return f;
    }

    inline bool intersects_frustum_sphere(const Frustum& f, const Sphere& s)
    {
        const float r = std::max(s.radius, 0.0f);
        for (const Plane& p : f.planes)
        {
            if (p.signed_distance(s.center) < -r) return false;
        }
        return true;
    }

    inline bool intersects_frustum_aabb(const Frustum& f, const AABB& box)
    {
        for (const Plane& p : f.planes)
        {
            glm::vec3 v{};
            v.x = (p.normal.x >= 0.0f) ? box.maxv.x : box.minv.x;
            v.y = (p.normal.y >= 0.0f) ? box.maxv.y : box.minv.y;
            v.z = (p.normal.z >= 0.0f) ? box.maxv.z : box.minv.z;
            if (p.signed_distance(v) < 0.0f) return false;
        }
        return true;
    }
}
