#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: shape_volume.hpp
    МОДУЛЬ: geometry
    ЗОРИЛГО: AAA түвшний culling-д хэрэглэх нийтлэг ShapeVolume семантик.
*/

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <variant>
#include <vector>

#include <glm/glm.hpp>

#include "shs/geometry/aabb.hpp"
#include "shs/geometry/volumes.hpp"

namespace shs
{
    struct ConeFrustum
    {
        glm::vec3 apex{0.0f};
        glm::vec3 axis{0.0f, -1.0f, 0.0f}; // normalized, apex -> far cap center
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

    struct KDOP18
    {
        // 9 axis directions => 18 planes (min/max per axis).
        std::array<glm::vec3, 9> axes{};
        std::array<float, 9> min_proj{};
        std::array<float, 9> max_proj{};

        // Conservative bounds for fast broad-phase culling.
        AABB bounds_aabb{};
        Sphere bounds_sphere{};
    };

    struct KDOP26
    {
        // 13 axis directions => 26 planes (min/max per axis).
        std::array<glm::vec3, 13> axes{};
        std::array<float, 13> min_proj{};
        std::array<float, 13> max_proj{};

        // Conservative bounds for fast broad-phase culling.
        AABB bounds_aabb{};
        Sphere bounds_sphere{};
    };

    struct SweptCapsule
    {
        // Culling semantic: conservative convex hull of the endpoint capsules.
        Capsule at_t0{};
        Capsule at_t1{};
        float t0 = 0.0f;
        float t1 = 1.0f;
    };

    struct SweptOBB
    {
        // Culling semantic: conservative convex hull of the endpoint OBBs.
        OBB at_t0{};
        OBB at_t1{};
        float t0 = 0.0f;
        float t1 = 1.0f;
    };

    // Ergonomic aliases that make endpoint-hull semantics explicit.
    using EndpointHullCapsule = SweptCapsule;
    using EndpointHullOBB = SweptOBB;

    struct MeshletHull
    {
        ConvexPolyhedron hull{};
        uint32_t meshlet_index = 0;
    };

    struct ClusterHull
    {
        ConvexPolyhedron hull{};
        uint32_t cluster_index = 0;
    };

    enum class ShapeVolumeKind : uint8_t
    {
        Sphere = 0,
        AABB = 1,
        OBB = 2,
        Capsule = 3,
        Cone = 4,
        ConeFrustum = 5,
        Cylinder = 6,
        ConvexPolyhedron = 7,
        KDOP18 = 8,
        KDOP26 = 9,
        SweptCapsule = 10,
        SweptOBB = 11,
        MeshletHull = 12,
        ClusterHull = 13
    };

    using ShapeVolumeVariant = std::variant<
        Sphere,
        AABB,
        OBB,
        Capsule,
        Cone,
        ConeFrustum,
        Cylinder,
        ConvexPolyhedron,
        KDOP18,
        KDOP26,
        SweptCapsule,
        SweptOBB,
        MeshletHull,
        ClusterHull>;

    inline ShapeVolumeKind shape_volume_kind(const ShapeVolumeVariant& shape)
    {
        return std::visit([](const auto& v) -> ShapeVolumeKind {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, Sphere>) return ShapeVolumeKind::Sphere;
            if constexpr (std::is_same_v<T, AABB>) return ShapeVolumeKind::AABB;
            if constexpr (std::is_same_v<T, OBB>) return ShapeVolumeKind::OBB;
            if constexpr (std::is_same_v<T, Capsule>) return ShapeVolumeKind::Capsule;
            if constexpr (std::is_same_v<T, Cone>) return ShapeVolumeKind::Cone;
            if constexpr (std::is_same_v<T, ConeFrustum>) return ShapeVolumeKind::ConeFrustum;
            if constexpr (std::is_same_v<T, Cylinder>) return ShapeVolumeKind::Cylinder;
            if constexpr (std::is_same_v<T, ConvexPolyhedron>) return ShapeVolumeKind::ConvexPolyhedron;
            if constexpr (std::is_same_v<T, KDOP18>) return ShapeVolumeKind::KDOP18;
            if constexpr (std::is_same_v<T, KDOP26>) return ShapeVolumeKind::KDOP26;
            if constexpr (std::is_same_v<T, SweptCapsule>) return ShapeVolumeKind::SweptCapsule;
            if constexpr (std::is_same_v<T, SweptOBB>) return ShapeVolumeKind::SweptOBB;
            if constexpr (std::is_same_v<T, MeshletHull>) return ShapeVolumeKind::MeshletHull;
            if constexpr (std::is_same_v<T, ClusterHull>) return ShapeVolumeKind::ClusterHull;
            return ShapeVolumeKind::Sphere;
        }, shape);
    }

    struct ShapeVolume
    {
        ShapeVolumeVariant value{};
        uint32_t stable_id = 0;

        inline ShapeVolumeKind kind() const
        {
            return shape_volume_kind(value);
        }
    };

    inline bool aabb_has_valid_extents(const AABB& box)
    {
        return
            (box.minv.x <= box.maxv.x) &&
            (box.minv.y <= box.maxv.y) &&
            (box.minv.z <= box.maxv.z);
    }

    inline Sphere conservative_bounds_sphere_from_points(const std::vector<glm::vec3>& points)
    {
        if (points.empty()) return Sphere{};
        AABB bounds{};
        for (const glm::vec3& p : points) bounds.expand(p);
        return sphere_from_aabb(bounds);
    }

    inline std::vector<glm::vec3> convex_vertices_from_planes(const std::vector<Plane>& planes, float eps);
    inline std::vector<glm::vec3> convex_polyhedron_vertices(const ConvexPolyhedron& hull, float eps);
    inline std::vector<glm::vec3> kdop18_vertices(const KDOP18& kdop, float eps);
    inline std::vector<glm::vec3> kdop26_vertices(const KDOP26& kdop, float eps);

    inline Sphere conservative_bounds_sphere(const Sphere& sphere)
    {
        return Sphere{
            sphere.center,
            std::max(sphere.radius, 0.0f)
        };
    }

    inline Sphere conservative_bounds_sphere(const AABB& box)
    {
        return sphere_from_aabb(box);
    }

    inline Sphere conservative_bounds_sphere(const OBB& obb)
    {
        return Sphere{
            obb.center,
            glm::length(glm::max(obb.half_extents, glm::vec3(0.0f)))
        };
    }

    inline Sphere conservative_bounds_sphere(const Capsule& capsule)
    {
        const glm::vec3 center = 0.5f * (capsule.a + capsule.b);
        const float half_len = 0.5f * glm::length(capsule.b - capsule.a);
        return Sphere{
            center,
            std::max(0.0f, half_len + std::max(capsule.radius, 0.0f))
        };
    }

    inline Sphere conservative_bounds_sphere(const Cone& cone)
    {
        const glm::vec3 axis = normalize_or(cone.axis, glm::vec3(0.0f, -1.0f, 0.0f));
        const float h = std::max(cone.height, 0.0f);
        const float r = std::max(cone.radius, 0.0f);
        const glm::vec3 center = cone.apex + axis * (0.5f * h);
        return Sphere{
            center,
            std::sqrt((0.5f * h) * (0.5f * h) + r * r)
        };
    }

    inline Sphere conservative_bounds_sphere(const ConeFrustum& cone)
    {
        const glm::vec3 axis = normalize_or(cone.axis, glm::vec3(0.0f, -1.0f, 0.0f));
        const float near_d = std::max(cone.near_distance, 0.0f);
        const float far_d = std::max(cone.far_distance, near_d);
        const float near_r = std::max(cone.near_radius, 0.0f);
        const float far_r = std::max(cone.far_radius, 0.0f);
        const glm::vec3 near_c = cone.apex + axis * near_d;
        const glm::vec3 far_c = cone.apex + axis * far_d;
        const glm::vec3 center = 0.5f * (near_c + far_c);
        const float half_len = 0.5f * glm::length(far_c - near_c);
        return Sphere{
            center,
            half_len + std::max(near_r, far_r)
        };
    }

    inline Sphere conservative_bounds_sphere(const Cylinder& cylinder)
    {
        const float h = std::max(cylinder.half_height, 0.0f);
        const float r = std::max(cylinder.radius, 0.0f);
        return Sphere{
            cylinder.center,
            std::sqrt(h * h + r * r)
        };
    }

    inline Sphere conservative_bounds_sphere(const ConvexPolyhedron& hull)
    {
        if (!hull.vertices.empty())
        {
            return conservative_bounds_sphere_from_points(hull.vertices);
        }
        if (hull.planes.empty()) return Sphere{};
        return conservative_bounds_sphere_from_points(convex_vertices_from_planes(hull.planes, 1e-5f));
    }

    inline Sphere conservative_bounds_sphere(const SweptCapsule& swept)
    {
        const Sphere s0 = conservative_bounds_sphere(swept.at_t0);
        const Sphere s1 = conservative_bounds_sphere(swept.at_t1);
        const glm::vec3 c = 0.5f * (s0.center + s1.center);
        const float r = 0.5f * glm::length(s1.center - s0.center) + std::max(s0.radius, s1.radius);
        return Sphere{c, r};
    }

    inline Sphere conservative_bounds_sphere(const SweptOBB& swept)
    {
        const Sphere s0 = conservative_bounds_sphere(swept.at_t0);
        const Sphere s1 = conservative_bounds_sphere(swept.at_t1);
        const glm::vec3 c = 0.5f * (s0.center + s1.center);
        const float r = 0.5f * glm::length(s1.center - s0.center) + std::max(s0.radius, s1.radius);
        return Sphere{c, r};
    }

    inline Sphere conservative_bounds_sphere(const MeshletHull& meshlet)
    {
        return conservative_bounds_sphere(meshlet.hull);
    }

    inline Sphere conservative_bounds_sphere(const ClusterHull& cluster)
    {
        return conservative_bounds_sphere(cluster.hull);
    }

    inline Sphere conservative_bounds_sphere(const KDOP18& kdop)
    {
        if (kdop.bounds_sphere.radius > 0.0f) return conservative_bounds_sphere(kdop.bounds_sphere);
        if (aabb_has_valid_extents(kdop.bounds_aabb))
        {
            return conservative_bounds_sphere(kdop.bounds_aabb);
        }
        return conservative_bounds_sphere_from_points(kdop18_vertices(kdop, 1e-5f));
    }

    inline Sphere conservative_bounds_sphere(const KDOP26& kdop)
    {
        if (kdop.bounds_sphere.radius > 0.0f) return conservative_bounds_sphere(kdop.bounds_sphere);
        if (aabb_has_valid_extents(kdop.bounds_aabb))
        {
            return conservative_bounds_sphere(kdop.bounds_aabb);
        }
        return conservative_bounds_sphere_from_points(kdop26_vertices(kdop, 1e-5f));
    }

    inline Sphere conservative_bounds_sphere(const ShapeVolumeVariant& shape)
    {
        return std::visit([](const auto& s) -> Sphere {
            return conservative_bounds_sphere(s);
        }, shape);
    }

    inline Sphere conservative_bounds_sphere(const ShapeVolume& shape)
    {
        return conservative_bounds_sphere(shape.value);
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

    inline std::vector<glm::vec3> convex_polyhedron_vertices(
        const ConvexPolyhedron& hull,
        float eps = 1e-5f)
    {
        if (!hull.vertices.empty()) return hull.vertices;
        if (hull.planes.empty()) return {};
        return convex_vertices_from_planes(hull.planes, eps);
    }

    inline std::vector<Plane> kdop18_planes(const KDOP18& kdop)
    {
        std::vector<Plane> out{};
        out.reserve(18);
        for (size_t i = 0; i < kdop.axes.size(); ++i)
        {
            const glm::vec3 axis = kdop.axes[i];
            const float len2 = glm::dot(axis, axis);
            if (len2 <= 1e-12f) continue;
            out.push_back(Plane{axis, -kdop.min_proj[i]});
            out.push_back(Plane{-axis, kdop.max_proj[i]});
        }
        return out;
    }

    inline std::vector<Plane> kdop26_planes(const KDOP26& kdop)
    {
        std::vector<Plane> out{};
        out.reserve(26);
        for (size_t i = 0; i < kdop.axes.size(); ++i)
        {
            const glm::vec3 axis = kdop.axes[i];
            const float len2 = glm::dot(axis, axis);
            if (len2 <= 1e-12f) continue;
            out.push_back(Plane{axis, -kdop.min_proj[i]});
            out.push_back(Plane{-axis, kdop.max_proj[i]});
        }
        return out;
    }

    inline std::vector<glm::vec3> kdop18_vertices(const KDOP18& kdop, float eps = 1e-5f)
    {
        return convex_vertices_from_planes(kdop18_planes(kdop), eps);
    }

    inline std::vector<glm::vec3> kdop26_vertices(const KDOP26& kdop, float eps = 1e-5f)
    {
        return convex_vertices_from_planes(kdop26_planes(kdop), eps);
    }

    inline std::array<glm::vec3, 8> obb_corners(const OBB& obb)
    {
        const glm::vec3 ex = glm::max(obb.half_extents, glm::vec3(0.0f));
        const glm::vec3 x = obb.axis_x * ex.x;
        const glm::vec3 y = obb.axis_y * ex.y;
        const glm::vec3 z = obb.axis_z * ex.z;
        return std::array<glm::vec3, 8>{
            obb.center - x - y - z,
            obb.center + x - y - z,
            obb.center - x + y - z,
            obb.center + x + y - z,
            obb.center - x - y + z,
            obb.center + x - y + z,
            obb.center - x + y + z,
            obb.center + x + y + z
        };
    }

    inline std::vector<glm::vec3> swept_obb_vertices(const SweptOBB& swept)
    {
        std::vector<glm::vec3> out{};
        out.reserve(16);
        const auto c0 = obb_corners(swept.at_t0);
        const auto c1 = obb_corners(swept.at_t1);
        out.insert(out.end(), c0.begin(), c0.end());
        out.insert(out.end(), c1.begin(), c1.end());
        return out;
    }

    inline EndpointHullCapsule make_endpoint_hull_capsule(
        const Capsule& at_t0,
        const Capsule& at_t1,
        float t0 = 0.0f,
        float t1 = 1.0f)
    {
        EndpointHullCapsule out{};
        out.at_t0 = at_t0;
        out.at_t1 = at_t1;
        out.t0 = t0;
        out.t1 = t1;
        return out;
    }

    inline EndpointHullOBB make_endpoint_hull_obb(
        const OBB& at_t0,
        const OBB& at_t1,
        float t0 = 0.0f,
        float t1 = 1.0f)
    {
        EndpointHullOBB out{};
        out.at_t0 = at_t0;
        out.at_t1 = at_t1;
        out.t0 = t0;
        out.t1 = t1;
        return out;
    }

    inline void prepare_convex_polyhedron_for_culling(ConvexPolyhedron& hull, float eps = 1e-5f)
    {
        if (!hull.vertices.empty()) return;
        if (hull.planes.empty()) return;
        hull.vertices = convex_vertices_from_planes(hull.planes, eps);
    }

    inline void prepare_kdop18_for_culling(KDOP18& kdop, float eps = 1e-5f)
    {
        if (kdop.bounds_sphere.radius > 0.0f) return;
        if (aabb_has_valid_extents(kdop.bounds_aabb))
        {
            kdop.bounds_sphere = conservative_bounds_sphere(kdop.bounds_aabb);
            return;
        }
        const std::vector<glm::vec3> verts = kdop18_vertices(kdop, eps);
        if (verts.empty()) return;
        AABB bounds{};
        for (const glm::vec3& v : verts) bounds.expand(v);
        kdop.bounds_aabb = bounds;
        kdop.bounds_sphere = sphere_from_aabb(bounds);
    }

    inline void prepare_kdop26_for_culling(KDOP26& kdop, float eps = 1e-5f)
    {
        if (kdop.bounds_sphere.radius > 0.0f) return;
        if (aabb_has_valid_extents(kdop.bounds_aabb))
        {
            kdop.bounds_sphere = conservative_bounds_sphere(kdop.bounds_aabb);
            return;
        }
        const std::vector<glm::vec3> verts = kdop26_vertices(kdop, eps);
        if (verts.empty()) return;
        AABB bounds{};
        for (const glm::vec3& v : verts) bounds.expand(v);
        kdop.bounds_aabb = bounds;
        kdop.bounds_sphere = sphere_from_aabb(bounds);
    }

    inline void prepare_shape_volume_for_culling(ShapeVolume& shape, float eps = 1e-5f)
    {
        std::visit([&](auto& s) {
            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, ConvexPolyhedron>)
            {
                prepare_convex_polyhedron_for_culling(s, eps);
            }
            else if constexpr (std::is_same_v<T, KDOP18>)
            {
                prepare_kdop18_for_culling(s, eps);
            }
            else if constexpr (std::is_same_v<T, KDOP26>)
            {
                prepare_kdop26_for_culling(s, eps);
            }
            else if constexpr (std::is_same_v<T, MeshletHull>)
            {
                prepare_convex_polyhedron_for_culling(s.hull, eps);
            }
            else if constexpr (std::is_same_v<T, ClusterHull>)
            {
                prepare_convex_polyhedron_for_culling(s.hull, eps);
            }
        }, shape.value);
    }

    inline void prepare_shape_volumes_for_culling(std::vector<ShapeVolume>& shapes, float eps = 1e-5f)
    {
        for (ShapeVolume& shape : shapes)
        {
            prepare_shape_volume_for_culling(shape, eps);
        }
    }
}
