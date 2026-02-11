#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: jolt_culling.hpp
    МОДУЛЬ: geometry
    ЗОРИЛГО: Jolt shape-уудад суурилсан нийтлэг culling API.
            ConvexCell (tile/cluster/cascade) болон Frustum дотор
            shape classify хийх C++20 concept-constrained функцүүд.

    CONVENTION:
        Бүх coordinate-ууд SHS LH space дотор ажиллана.
        Jolt shape-ийн world bounds-ийг LH руу хөрвүүлж тест хийнэ.
*/

#if defined(SHS_HAS_JOLT) && ((SHS_HAS_JOLT + 0) == 1)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include <glm/glm.hpp>

#include <Jolt/Jolt.h>
#include <Jolt/Geometry/AABox.h>
#include <Jolt/Physics/Collision/Shape/Shape.h>

#include "shs/geometry/aabb.hpp"
#include "shs/geometry/frustum_culling.hpp"
#include "shs/geometry/volumes.hpp"
#include "shs/geometry/jolt_adapter.hpp"
#include "shs/geometry/jolt_shape_traits.hpp"
#include "shs/geometry/scene_shape.hpp"

namespace shs
{
    // =========================================================================
    //  CullingCell — lightweight tile/cluster/cascade cell
    //  Equivalent to the old ConvexCell but designed for Jolt-based pipeline.
    // =========================================================================

    inline constexpr uint32_t k_culling_cell_max_planes = 16u;

    enum class CullingCellKind : uint8_t
    {
        CameraFrustumPerspective = 0,
        CameraFrustumOrthographic = 1,
        CascadeFrustum = 2,
        SpotShadowFrustum = 3,
        PointShadowFaceFrustum = 4,
        ScreenTileCell = 5,
        TileDepthCell = 6,
        ClusterCellPerspective = 7,
        ClusterCellOrthographic = 8,
        ClusterDepthCell = 9,
        PortalClippedCell = 10,
        CustomPlaneSetCell = 11
    };

    struct CullingCell
    {
        CullingCellKind kind = CullingCellKind::CustomPlaneSetCell;
        uint32_t plane_count = 0;
        std::array<Plane, k_culling_cell_max_planes> planes{};
        AABB bounds_aabb{};
        Sphere bounds_sphere{};
        glm::uvec4 user_data{0u, 0u, 0u, 0u};
    };

    inline bool culling_cell_valid(const CullingCell& cell) noexcept
    {
        return cell.plane_count > 0 && cell.plane_count <= k_culling_cell_max_planes;
    }

    inline bool culling_cell_add_plane(CullingCell& cell, const Plane& plane) noexcept
    {
        if (cell.plane_count >= k_culling_cell_max_planes) return false;
        cell.planes[cell.plane_count] = plane;
        ++cell.plane_count;
        return true;
    }

    inline CullingCell make_culling_cell_from_frustum(
        const Frustum& frustum,
        CullingCellKind kind = CullingCellKind::CameraFrustumPerspective)
    {
        CullingCell out{};
        out.kind = kind;
        out.plane_count = 6;
        for (size_t i = 0; i < 6; ++i) out.planes[i] = frustum.planes[i];
        return out;
    }

    inline CullingCell extract_frustum_cell(
        const glm::mat4& view_proj,
        CullingCellKind kind = CullingCellKind::CameraFrustumPerspective)
    {
        const Frustum frustum = extract_frustum_planes(view_proj);
        return make_culling_cell_from_frustum(frustum, kind);
    }


    // =========================================================================
    //  CullClass — tri-state classification
    // =========================================================================

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


    // =========================================================================
    //  Sphere vs Cell classification (SHS LH space)
    // =========================================================================

    inline CullClass classify_sphere_vs_cell(
        const Sphere& sphere,
        const CullingCell& cell,
        const CullTolerance& tol = {}) noexcept
    {
        if (!culling_cell_valid(cell)) return CullClass::Intersecting;

        const float r = std::max(sphere.radius, 0.0f);
        bool fully_inside = true;
        for (uint32_t i = 0; i < cell.plane_count; ++i)
        {
            const float dist = cell.planes[i].signed_distance(sphere.center);
            if (dist < -(r + tol.outside_epsilon)) return CullClass::Outside;
            if (dist < (r + tol.inside_epsilon)) fully_inside = false;
        }
        return fully_inside ? CullClass::Inside : CullClass::Intersecting;
    }


    // =========================================================================
    //  AABB vs Cell classification (SHS LH space)
    // =========================================================================

    inline CullClass classify_aabb_vs_cell(
        const AABB& aabb,
        const CullingCell& cell,
        const CullTolerance& tol = {}) noexcept
    {
        if (!culling_cell_valid(cell)) return CullClass::Intersecting;

        bool fully_inside = true;
        for (uint32_t i = 0; i < cell.plane_count; ++i)
        {
            const Plane& p = cell.planes[i];

            // P-vertex: the vertex most "inside" the plane direction.
            const glm::vec3 p_vert(
                (p.normal.x >= 0.0f) ? aabb.maxv.x : aabb.minv.x,
                (p.normal.y >= 0.0f) ? aabb.maxv.y : aabb.minv.y,
                (p.normal.z >= 0.0f) ? aabb.maxv.z : aabb.minv.z);
            if (p.signed_distance(p_vert) < -tol.outside_epsilon)
                return CullClass::Outside;

            // N-vertex: the vertex most "outside" the plane direction.
            const glm::vec3 n_vert(
                (p.normal.x >= 0.0f) ? aabb.minv.x : aabb.maxv.x,
                (p.normal.y >= 0.0f) ? aabb.minv.y : aabb.maxv.y,
                (p.normal.z >= 0.0f) ? aabb.minv.z : aabb.maxv.z);
            if (p.signed_distance(n_vert) < tol.inside_epsilon)
                fully_inside = false;
        }
        return fully_inside ? CullClass::Inside : CullClass::Intersecting;
    }


    // =========================================================================
    //  Sphere vs Frustum classification (SHS LH space)
    // =========================================================================

    inline CullClass classify_sphere_vs_frustum(
        const Sphere& sphere,
        const Frustum& frustum,
        const CullTolerance& tol = {}) noexcept
    {
        const float r = std::max(sphere.radius, 0.0f);
        bool fully_inside = true;
        for (const Plane& p : frustum.planes)
        {
            const float dist = p.signed_distance(sphere.center);
            if (dist < -(r + tol.outside_epsilon)) return CullClass::Outside;
            if (dist < (r + tol.inside_epsilon)) fully_inside = false;
        }
        return fully_inside ? CullClass::Inside : CullClass::Intersecting;
    }


    // =========================================================================
    //  AABB vs Frustum classification (SHS LH space)
    // =========================================================================

    inline CullClass classify_aabb_vs_frustum(
        const AABB& aabb,
        const Frustum& frustum,
        const CullTolerance& tol = {}) noexcept
    {
        bool fully_inside = true;
        for (const Plane& p : frustum.planes)
        {
            const glm::vec3 p_vert(
                (p.normal.x >= 0.0f) ? aabb.maxv.x : aabb.minv.x,
                (p.normal.y >= 0.0f) ? aabb.maxv.y : aabb.minv.y,
                (p.normal.z >= 0.0f) ? aabb.maxv.z : aabb.minv.z);
            if (p.signed_distance(p_vert) < -tol.outside_epsilon)
                return CullClass::Outside;

            const glm::vec3 n_vert(
                (p.normal.x >= 0.0f) ? aabb.minv.x : aabb.maxv.x,
                (p.normal.y >= 0.0f) ? aabb.minv.y : aabb.maxv.y,
                (p.normal.z >= 0.0f) ? aabb.minv.z : aabb.maxv.z);
            if (p.signed_distance(n_vert) < tol.inside_epsilon)
                fully_inside = false;
        }
        return fully_inside ? CullClass::Inside : CullClass::Intersecting;
    }


    // =========================================================================
    //  Concept-constrained: Cullable/FastCullable vs Cell
    // =========================================================================

    template<FastCullable T>
    inline CullClass classify_vs_cell(
        const T& obj,
        const CullingCell& cell,
        const CullTolerance& tol = {})
    {
        // Fast path: bounding sphere test first.
        const Sphere broad = obj.bounding_sphere();
        const CullClass broad_class = classify_sphere_vs_cell(broad, cell, tol);
        if (broad_class == CullClass::Outside) return CullClass::Outside;
        if (broad_class == CullClass::Inside)  return CullClass::Inside;

        // Refine with world AABB.
        if constexpr (HasWorldAABB<T>)
        {
            return classify_aabb_vs_cell(obj.world_aabb(), cell, tol);
        }
        return CullClass::Intersecting;
    }

    template<FastCullable T>
    inline CullClass classify_vs_frustum(
        const T& obj,
        const Frustum& frustum,
        const CullTolerance& tol = {})
    {
        const Sphere broad = obj.bounding_sphere();
        const CullClass broad_class = classify_sphere_vs_frustum(broad, frustum, tol);
        if (broad_class == CullClass::Outside) return CullClass::Outside;
        if (broad_class == CullClass::Inside)  return CullClass::Inside;

        if constexpr (HasWorldAABB<T>)
        {
            return classify_aabb_vs_frustum(obj.world_aabb(), frustum, tol);
        }
        return CullClass::Intersecting;
    }


    // =========================================================================
    //  Batch culling result
    // =========================================================================

    struct CullResult
    {
        std::vector<CullClass> classes{};
        std::vector<size_t>    visible_indices{};
        uint64_t tested = 0;
        uint64_t outside = 0;
        uint64_t intersecting = 0;
        uint64_t inside = 0;
    };


    // =========================================================================
    //  Batch cull vs Frustum
    // =========================================================================

    template<FastCullable T>
    inline CullResult cull_vs_frustum(
        std::span<const T> objects,
        const Frustum& frustum,
        const CullTolerance& tol = {})
    {
        CullResult out{};
        const size_t n = objects.size();
        out.classes.resize(n, CullClass::Intersecting);
        out.visible_indices.reserve(n);
        out.tested = n;

        for (size_t i = 0; i < n; ++i)
        {
            const CullClass c = classify_vs_frustum(objects[i], frustum, tol);
            out.classes[i] = c;
            switch (c)
            {
                case CullClass::Outside:       ++out.outside;       break;
                case CullClass::Inside:         ++out.inside;
                    out.visible_indices.push_back(i); break;
                case CullClass::Intersecting:   ++out.intersecting;
                    out.visible_indices.push_back(i); break;
            }
        }
        return out;
    }


    // =========================================================================
    //  Batch cull vs Cell
    // =========================================================================

    template<FastCullable T>
    inline CullResult cull_vs_cell(
        std::span<const T> objects,
        const CullingCell& cell,
        const CullTolerance& tol = {})
    {
        CullResult out{};
        const size_t n = objects.size();
        out.classes.resize(n, CullClass::Intersecting);
        out.visible_indices.reserve(n);
        out.tested = n;

        for (size_t i = 0; i < n; ++i)
        {
            const CullClass c = classify_vs_cell(objects[i], cell, tol);
            out.classes[i] = c;
            switch (c)
            {
                case CullClass::Outside:       ++out.outside;       break;
                case CullClass::Inside:         ++out.inside;
                    out.visible_indices.push_back(i); break;
                case CullClass::Intersecting:   ++out.intersecting;
                    out.visible_indices.push_back(i); break;
            }
        }
        return out;
    }


    // =========================================================================
    //  Helper: is a CullClass visible?
    // =========================================================================

    inline bool cull_class_is_visible(CullClass c, bool include_intersecting = true) noexcept
    {
        if (c == CullClass::Inside) return true;
        if (include_intersecting && c == CullClass::Intersecting) return true;
        return false;
    }
}

#endif // SHS_HAS_JOLT
