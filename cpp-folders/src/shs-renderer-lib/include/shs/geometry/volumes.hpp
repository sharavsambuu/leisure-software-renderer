#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: volumes.hpp
    МОДУЛЬ: geometry
    ЗОРИЛГО: Нийтлэг хэрэглэгдэх 3D/2D volume primitive-уудын нэгтгэсэн abstraction.
            (light culling, scene culling, debug proxy geometry, broad phase гэх мэт)
*/

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
#include <variant>

#include <glm/glm.hpp>

#include "shs/geometry/aabb.hpp"

namespace shs
{
    struct Point3
    {
        glm::vec3 p{0.0f};
    };

    struct LineSegment3
    {
        glm::vec3 a{0.0f};
        glm::vec3 b{0.0f};
    };

    struct Ray3
    {
        glm::vec3 origin{0.0f};
        glm::vec3 direction{0.0f, 0.0f, 1.0f};
    };

    struct Plane
    {
        glm::vec3 normal{0.0f, 1.0f, 0.0f};
        float d = 0.0f; // plane eq: dot(normal, x) + d = 0

        inline float signed_distance(const glm::vec3& p) const
        {
            return glm::dot(normal, p) + d;
        }
    };

    struct Sphere
    {
        glm::vec3 center{0.0f};
        float radius = 0.0f;
    };

    struct OBB
    {
        glm::vec3 center{0.0f};
        glm::vec3 axis_x{1.0f, 0.0f, 0.0f};
        glm::vec3 axis_y{0.0f, 1.0f, 0.0f};
        glm::vec3 axis_z{0.0f, 0.0f, 1.0f};
        glm::vec3 half_extents{0.5f, 0.5f, 0.5f};
    };

    struct Capsule
    {
        glm::vec3 a{0.0f};
        glm::vec3 b{0.0f, 1.0f, 0.0f};
        float radius = 0.25f;
    };

    struct Cone
    {
        glm::vec3 apex{0.0f};
        glm::vec3 axis{0.0f, -1.0f, 0.0f}; // normalized, apex -> base
        float height = 1.0f;
        float radius = 0.5f;
    };

    struct Cylinder
    {
        glm::vec3 center{0.0f};
        glm::vec3 axis{0.0f, 1.0f, 0.0f}; // normalized
        float half_height = 0.5f;
        float radius = 0.5f;
    };

    struct Frustum
    {
        std::array<Plane, 6> planes{};
    };

    struct OrientedRect
    {
        glm::vec3 center{0.0f};
        glm::vec3 right{1.0f, 0.0f, 0.0f};
        glm::vec3 up{0.0f, 1.0f, 0.0f};
        glm::vec2 half_extents{0.5f, 0.5f};
    };

    struct Disk
    {
        glm::vec3 center{0.0f};
        glm::vec3 normal{0.0f, 1.0f, 0.0f};
        float radius = 0.5f;
    };

    struct ConeFrustum
    {
        glm::vec3 apex{0.0f};
        glm::vec3 axis{0.0f, -1.0f, 0.0f}; // apex -> base
        float near_distance = 0.0f;
        float far_distance = 1.0f;
        float near_radius = 0.0f;
        float far_radius = 0.5f;
    };

    struct ConvexPolyhedron
    {
        std::vector<glm::vec3> vertices{};
        std::vector<Plane> planes{};
    };



    struct Rect2
    {
        glm::vec2 minv{0.0f};
        glm::vec2 maxv{0.0f};
    };

    inline glm::vec3 normalize_or(const glm::vec3& v, const glm::vec3& fallback)
    {
        const float len2 = glm::dot(v, v);
        if (len2 <= 1e-10f) return fallback;
        return v * (1.0f / std::sqrt(len2));
    }

    inline Plane make_plane_from_point_normal(const glm::vec3& point, const glm::vec3& normal)
    {
        Plane p{};
        p.normal = normalize_or(normal, glm::vec3(0.0f, 1.0f, 0.0f));
        p.d = -glm::dot(p.normal, point);
        return p;
    }

    inline Sphere sphere_from_aabb(const AABB& box)
    {
        Sphere s{};
        s.center = box.center();
        s.radius = glm::length(box.maxv - s.center);
        return s;
    }

    inline Sphere transform_sphere(const Sphere& local, const glm::mat4& model)
    {
        Sphere out{};
        out.center = glm::vec3(model * glm::vec4(local.center, 1.0f));
        const float sx = glm::length(glm::vec3(model[0]));
        const float sy = glm::length(glm::vec3(model[1]));
        const float sz = glm::length(glm::vec3(model[2]));
        const float s = std::max(sx, std::max(sy, sz));
        out.radius = local.radius * s;
        return out;
    }

    inline AABB transform_aabb(const AABB& local, const glm::mat4& model)
    {
        const glm::vec3 corners[8] = {
            {local.minv.x, local.minv.y, local.minv.z},
            {local.maxv.x, local.minv.y, local.minv.z},
            {local.minv.x, local.maxv.y, local.minv.z},
            {local.maxv.x, local.maxv.y, local.minv.z},
            {local.minv.x, local.minv.y, local.maxv.z},
            {local.maxv.x, local.minv.y, local.maxv.z},
            {local.minv.x, local.maxv.y, local.maxv.z},
            {local.maxv.x, local.maxv.y, local.maxv.z},
        };

        AABB out{};
        for (const glm::vec3& c : corners)
        {
            out.expand(glm::vec3(model * glm::vec4(c, 1.0f)));
        }
        return out;
    }
    inline bool intersect_three_planes(
        const Plane& p0,
        const Plane& p1,
        const Plane& p2,
        glm::vec3& out_point,
        float eps = 1e-8f)
    {
        const glm::vec3 c01 = glm::cross(p1.normal, p2.normal);
        const float det = glm::dot(p0.normal, c01);
        if (std::abs(det) <= eps) return false;

        out_point =
            (-p0.d * c01 - p1.d * glm::cross(p2.normal, p0.normal) - p2.d * glm::cross(p0.normal, p1.normal)) / det;
        return std::isfinite(out_point.x) && std::isfinite(out_point.y) && std::isfinite(out_point.z);
    }

    inline bool point_inside_planes(
        const glm::vec3& p,
        const std::vector<Plane>& planes,
        float eps = 1e-5f)
    {
        for (const Plane& plane : planes)
        {
            if (plane.signed_distance(p) < -eps) return false;
        }
        return true;
    }

    inline void append_unique_vertex(
        std::vector<glm::vec3>& out_vertices,
        const glm::vec3& v,
        float eps = 1e-5f)
    {
        const float eps2 = eps * eps;
        for (const glm::vec3& existing : out_vertices)
        {
            const glm::vec3 delta = existing - v;
            if (glm::dot(delta, delta) <= eps2) return;
        }
        out_vertices.push_back(v);
    }

    inline std::vector<glm::vec3> convex_vertices_from_planes(
        const std::vector<Plane>& planes,
        float eps = 1e-5f)
    {
        std::vector<glm::vec3> out{};
        if (planes.size() < 4) return out;

        for (size_t i = 0; i < planes.size(); ++i)
        {
            for (size_t j = i + 1; j < planes.size(); ++j)
            {
                for (size_t k = j + 1; k < planes.size(); ++k)
                {
                    glm::vec3 p{};
                    if (!intersect_three_planes(planes[i], planes[j], planes[k], p)) continue;
                    if (!point_inside_planes(p, planes, eps)) continue;
                    append_unique_vertex(out, p, eps * 2.0f);
                }
            }
        }
        return out;
    }
}
