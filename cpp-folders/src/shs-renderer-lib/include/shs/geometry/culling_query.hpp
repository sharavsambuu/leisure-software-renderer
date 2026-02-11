#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: culling_query.hpp
    МОДУЛЬ: geometry
    ЗОРИЛГО: ShapeVolume vs ConvexCell нийтлэг classify API
            (Outside / Intersecting / Inside).
*/

#include <algorithm>
#include <cmath>
#include <variant>
#include <vector>

#include "shs/geometry/convex_cell.hpp"
#include "shs/geometry/volumes.hpp"

namespace shs
{
    enum class CullClass : uint8_t
    {
        Outside = 0,
        Intersecting = 1,
        Inside = 2
    };

    struct CullTolerance
    {
        float outside_epsilon = 1e-5f;
        float inside_epsilon = 1e-5f;
    };

    inline CullClass classify_convex_vertices(
        const std::vector<glm::vec3>& vertices,
        const ConvexCell& cell,
        const CullTolerance& tol = {})
    {
        if (!convex_cell_valid(cell)) return CullClass::Intersecting;
        if (vertices.empty()) return CullClass::Intersecting;

        bool fully_inside = true;
        for (uint32_t i = 0; i < cell.plane_count; ++i)
        {
            const Plane& p = cell.planes[i];
            bool any_inside = false;
            bool all_inside = true;
            for (const glm::vec3& v : vertices)
            {
                const float d = p.signed_distance(v);
                if (d >= -tol.outside_epsilon) any_inside = true;
                if (d < tol.inside_epsilon) all_inside = false;
            }
            if (!any_inside) return CullClass::Outside;
            if (!all_inside) fully_inside = false;
        }
        return fully_inside ? CullClass::Inside : CullClass::Intersecting;
    }

    inline float support_max_dot(const Sphere& sphere, const glm::vec3& dir)
    {
        return glm::dot(dir, sphere.center) + std::max(sphere.radius, 0.0f) * glm::length(dir);
    }

    inline float support_max_dot(const AABB& box, const glm::vec3& dir)
    {
        const glm::vec3 p = glm::vec3(
            (dir.x >= 0.0f) ? box.maxv.x : box.minv.x,
            (dir.y >= 0.0f) ? box.maxv.y : box.minv.y,
            (dir.z >= 0.0f) ? box.maxv.z : box.minv.z);
        return glm::dot(dir, p);
    }

    inline float support_max_dot(const OBB& obb, const glm::vec3& dir)
    {
        const glm::vec3 ex = glm::max(obb.half_extents, glm::vec3(0.0f));
        return
            glm::dot(dir, obb.center) +
            ex.x * std::abs(glm::dot(dir, obb.axis_x)) +
            ex.y * std::abs(glm::dot(dir, obb.axis_y)) +
            ex.z * std::abs(glm::dot(dir, obb.axis_z));
    }

    inline float support_max_dot(const Capsule& capsule, const glm::vec3& dir)
    {
        const float end = std::max(glm::dot(dir, capsule.a), glm::dot(dir, capsule.b));
        return end + std::max(capsule.radius, 0.0f) * glm::length(dir);
    }

    inline float support_max_dot(const Cone& cone, const glm::vec3& dir)
    {
        const glm::vec3 axis = normalize_or(cone.axis, glm::vec3(0.0f, -1.0f, 0.0f));
        const float h = std::max(cone.height, 0.0f);
        const float r = std::max(cone.radius, 0.0f);
        const glm::vec3 base_center = cone.apex + axis * h;
        const float axis_dot = glm::dot(dir, axis);
        const glm::vec3 perp = dir - axis * axis_dot;
        const float disk_support = glm::dot(dir, base_center) + r * glm::length(perp);
        return std::max(glm::dot(dir, cone.apex), disk_support);
    }

    inline float support_max_dot(const ConeFrustum& cone, const glm::vec3& dir)
    {
        const glm::vec3 axis = normalize_or(cone.axis, glm::vec3(0.0f, -1.0f, 0.0f));
        const float near_d = std::max(cone.near_distance, 0.0f);
        const float far_d = std::max(cone.far_distance, near_d);
        const float near_r = std::max(cone.near_radius, 0.0f);
        const float far_r = std::max(cone.far_radius, 0.0f);

        const glm::vec3 near_c = cone.apex + axis * near_d;
        const glm::vec3 far_c = cone.apex + axis * far_d;
        const float axis_dot = glm::dot(dir, axis);
        const glm::vec3 perp = dir - axis * axis_dot;
        const float perp_len = glm::length(perp);
        const float near_support = glm::dot(dir, near_c) + near_r * perp_len;
        const float far_support = glm::dot(dir, far_c) + far_r * perp_len;
        return std::max(near_support, far_support);
    }

    inline float support_max_dot(const Cylinder& cylinder, const glm::vec3& dir)
    {
        const glm::vec3 axis = normalize_or(cylinder.axis, glm::vec3(0.0f, 1.0f, 0.0f));
        const float half_h = std::max(cylinder.half_height, 0.0f);
        const float r = std::max(cylinder.radius, 0.0f);
        const float axis_dot = glm::dot(dir, axis);
        const glm::vec3 perp = dir - axis * axis_dot;
        return
            glm::dot(dir, cylinder.center) +
            half_h * std::abs(axis_dot) +
            r * glm::length(perp);
    }

    inline float support_max_dot(const SweptCapsule& swept, const glm::vec3& dir)
    {
        const float r0 = std::max(swept.at_t0.radius, 0.0f) * glm::length(dir);
        const float r1 = std::max(swept.at_t1.radius, 0.0f) * glm::length(dir);
        float best = glm::dot(dir, swept.at_t0.a) + r0;
        best = std::max(best, glm::dot(dir, swept.at_t0.b) + r0);
        best = std::max(best, glm::dot(dir, swept.at_t1.a) + r1);
        best = std::max(best, glm::dot(dir, swept.at_t1.b) + r1);
        return best;
    }

    inline float support_max_dot(const SweptOBB& swept, const glm::vec3& dir)
    {
        // SweptOBB-ийг endpoint OBB хоёрын convex hull гэж семантикчилбал support нь max(h0, h1).
        return std::max(support_max_dot(swept.at_t0, dir), support_max_dot(swept.at_t1, dir));
    }

    template<typename ShapeT>
    inline CullClass classify_support_shape(
        const ShapeT& shape,
        const ConvexCell& cell,
        const CullTolerance& tol = {})
    {
        if (!convex_cell_valid(cell)) return CullClass::Intersecting;

        bool fully_inside = true;
        for (uint32_t i = 0; i < cell.plane_count; ++i)
        {
            const Plane& p = cell.planes[i];
            const float max_d = support_max_dot(shape, p.normal) + p.d;
            if (max_d < -tol.outside_epsilon) return CullClass::Outside;

            const float min_d = -support_max_dot(shape, -p.normal) + p.d;
            if (min_d < tol.inside_epsilon) fully_inside = false;
        }
        return fully_inside ? CullClass::Inside : CullClass::Intersecting;
    }

    inline CullClass classify(
        const Sphere& sphere,
        const ConvexCell& cell,
        const CullTolerance& tol = {})
    {
        return classify_support_shape(sphere, cell, tol);
    }

    inline CullClass classify(
        const AABB& box,
        const ConvexCell& cell,
        const CullTolerance& tol = {})
    {
        return classify_support_shape(box, cell, tol);
    }

    inline CullClass classify(
        const OBB& obb,
        const ConvexCell& cell,
        const CullTolerance& tol = {})
    {
        return classify_support_shape(obb, cell, tol);
    }

    inline CullClass classify(
        const Capsule& capsule,
        const ConvexCell& cell,
        const CullTolerance& tol = {})
    {
        return classify_support_shape(capsule, cell, tol);
    }

    inline CullClass classify(
        const Cone& cone,
        const ConvexCell& cell,
        const CullTolerance& tol = {})
    {
        return classify_support_shape(cone, cell, tol);
    }

    inline CullClass classify(
        const ConeFrustum& cone,
        const ConvexCell& cell,
        const CullTolerance& tol = {})
    {
        return classify_support_shape(cone, cell, tol);
    }

    inline CullClass classify(
        const Cylinder& cylinder,
        const ConvexCell& cell,
        const CullTolerance& tol = {})
    {
        return classify_support_shape(cylinder, cell, tol);
    }

    inline CullClass classify(
        const ConvexPolyhedron& hull,
        const ConvexCell& cell,
        const CullTolerance& tol = {})
    {
        const std::vector<glm::vec3> verts = convex_polyhedron_vertices(hull);
        if (verts.empty()) return classify(conservative_bounds_sphere(hull), cell, tol);
        return classify_convex_vertices(verts, cell, tol);
    }

    inline CullClass classify(
        const KDOP18& kdop,
        const ConvexCell& cell,
        const CullTolerance& tol = {})
    {
        const std::vector<glm::vec3> verts = kdop18_vertices(kdop);
        if (verts.empty()) return classify(conservative_bounds_sphere(kdop), cell, tol);
        return classify_convex_vertices(verts, cell, tol);
    }

    inline CullClass classify(
        const KDOP26& kdop,
        const ConvexCell& cell,
        const CullTolerance& tol = {})
    {
        const std::vector<glm::vec3> verts = kdop26_vertices(kdop);
        if (verts.empty()) return classify(conservative_bounds_sphere(kdop), cell, tol);
        return classify_convex_vertices(verts, cell, tol);
    }

    inline CullClass classify(
        const SweptCapsule& swept,
        const ConvexCell& cell,
        const CullTolerance& tol = {})
    {
        return classify_support_shape(swept, cell, tol);
    }

    inline CullClass classify(
        const SweptOBB& swept,
        const ConvexCell& cell,
        const CullTolerance& tol = {})
    {
        return classify_support_shape(swept, cell, tol);
    }

    inline CullClass classify(
        const MeshletHull& meshlet,
        const ConvexCell& cell,
        const CullTolerance& tol = {})
    {
        return classify(meshlet.hull, cell, tol);
    }

    inline CullClass classify(
        const ClusterHull& cluster,
        const ConvexCell& cell,
        const CullTolerance& tol = {})
    {
        return classify(cluster.hull, cell, tol);
    }

    inline CullClass classify(
        const ShapeVolumeVariant& shape,
        const ConvexCell& cell,
        const CullTolerance& tol = {})
    {
        return std::visit([&](const auto& s) -> CullClass {
            return classify(s, cell, tol);
        }, shape);
    }

    inline CullClass classify(
        const ShapeVolume& shape,
        const ConvexCell& cell,
        const CullTolerance& tol = {})
    {
        return classify(shape.value, cell, tol);
    }
}

