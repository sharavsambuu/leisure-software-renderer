#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: convex_cell.hpp
    МОДУЛЬ: geometry
    ЗОРИЛГО: Frustum/tile/cluster/cascade зэрэг бүх convex culling cell-ийн
            нийтлэг Plane-set семантик.
*/

#include <array>
#include <cstdint>

#include <glm/glm.hpp>

#include "shs/geometry/aabb.hpp"
#include "shs/geometry/frustum_culling.hpp"
#include "shs/geometry/volumes.hpp"

namespace shs
{
    inline constexpr uint32_t k_convex_cell_max_planes = 16u;

    enum class ConvexCellKind : uint8_t
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

    struct ConvexCell
    {
        ConvexCellKind kind = ConvexCellKind::CustomPlaneSetCell;
        uint32_t plane_count = 0;
        std::array<Plane, k_convex_cell_max_planes> planes{};

        // Conservative bounds for fast reject.
        AABB bounds_aabb{};
        Sphere bounds_sphere{};

        // Optional metadata for backends (tile coord, cascade id, view id, etc.).
        glm::uvec4 user_data{0u, 0u, 0u, 0u};
    };

    inline bool convex_cell_valid(const ConvexCell& cell)
    {
        return cell.plane_count > 0 && cell.plane_count <= k_convex_cell_max_planes;
    }

    inline bool convex_cell_add_plane(ConvexCell& cell, const Plane& plane)
    {
        if (cell.plane_count >= k_convex_cell_max_planes) return false;
        cell.planes[cell.plane_count] = plane;
        ++cell.plane_count;
        return true;
    }

    inline ConvexCell make_convex_cell_from_frustum(
        const Frustum& frustum,
        ConvexCellKind kind = ConvexCellKind::CameraFrustumPerspective)
    {
        ConvexCell out{};
        out.kind = kind;
        out.plane_count = 6;
        for (size_t i = 0; i < 6; ++i) out.planes[i] = frustum.planes[i];
        return out;
    }

    inline ConvexCell extract_frustum_cell(
        const glm::mat4& view_proj,
        ConvexCellKind kind = ConvexCellKind::CameraFrustumPerspective)
    {
        const Frustum frustum = extract_frustum_planes(view_proj);
        return make_convex_cell_from_frustum(frustum, kind);
    }
}

