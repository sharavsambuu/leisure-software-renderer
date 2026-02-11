#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: jolt_debug_draw.hpp
    МОДУЛЬ: geometry
    ЗОРИЛГО: Jolt shape-аас wireframe/triangle mesh үүсгэх debug draw utility.
            Дурын JPH::Shape-ийг SHS LH space дотор визуалчлах.
*/

#if defined(SHS_HAS_JOLT) && ((SHS_HAS_JOLT + 0) == 1)

#include <algorithm>
#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

#include <Jolt/Jolt.h>
#include <Jolt/Math/Mat44.h>
#include <Jolt/Physics/Collision/Shape/Shape.h>
#include <Jolt/Physics/Collision/Shape/SubShapeID.h>

#include "shs/geometry/aabb.hpp"
#include "shs/geometry/volumes.hpp"
#include "shs/geometry/jolt_adapter.hpp"
#include "shs/geometry/jolt_culling.hpp"

namespace shs
{
    // =========================================================================
    //  Debug mesh — simple indexed triangle mesh for visualization
    // =========================================================================

    struct DebugMesh
    {
        std::vector<glm::vec3> vertices{};
        std::vector<uint32_t>  indices{};

        void clear()
        {
            vertices.clear();
            indices.clear();
        }

        bool empty() const noexcept
        {
            return vertices.empty();
        }
    };


    // =========================================================================
    //  Generate triangle mesh from Jolt shape
    //  Uses JPH::Shape::GetTrianglesStart/GetTrianglesNext to extract geometry,
    //  then converts all vertices from Jolt RH to SHS LH space.
    // =========================================================================

    inline DebugMesh debug_mesh_from_shape(
        const JPH::Shape& shape,
        const JPH::Mat44& transform = JPH::Mat44::sIdentity())
    {
        DebugMesh mesh{};

        // Get the world-space bounding box for the shape.
        const JPH::AABox shape_bounds = shape.GetWorldSpaceBounds(transform, JPH::Vec3::sReplicate(1.0f));

        // Jolt's triangle extraction context.
        JPH::Shape::GetTrianglesContext context;
        shape.GetTrianglesStart(
            context,
            shape_bounds,
            transform.GetTranslation(),
            transform.GetQuaternion(),
            JPH::Vec3::sReplicate(1.0f));

        // Extract triangles in batches.
        constexpr int k_batch_size = 256;
        JPH::Float3 tri_verts[k_batch_size * 3];

        for (;;)
        {
            const int tri_count = shape.GetTrianglesNext(
                context,
                k_batch_size,
                tri_verts,
                nullptr); // No material needed for debug draw.

            if (tri_count == 0) break;

            const uint32_t base = static_cast<uint32_t>(mesh.vertices.size());
            for (int t = 0; t < tri_count; ++t)
            {
                for (int v = 0; v < 3; ++v)
                {
                    const JPH::Float3& fv = tri_verts[t * 3 + v];
                    // Convert from Jolt RH to SHS LH (negate Z).
                    mesh.vertices.push_back(glm::vec3(fv.x, fv.y, -fv.z));
                    mesh.indices.push_back(base + static_cast<uint32_t>(t * 3 + v));
                }
            }
        }
        return mesh;
    }


    // =========================================================================
    //  Debug mesh from SceneShape
    // =========================================================================

    inline DebugMesh debug_mesh_from_scene_shape(const SceneShape& scene_shape)
    {
        if (!scene_shape.shape) return DebugMesh{};
        return debug_mesh_from_shape(*scene_shape.shape, scene_shape.transform);
    }


    // =========================================================================
    //  Debug mesh from AABB (wireframe box → 12 edges → 12 line-quads)
    // =========================================================================

    inline DebugMesh debug_mesh_from_aabb(const AABB& aabb)
    {
        DebugMesh mesh{};
        const glm::vec3& lo = aabb.minv;
        const glm::vec3& hi = aabb.maxv;

        mesh.vertices = {
            {lo.x, lo.y, lo.z}, // 0
            {hi.x, lo.y, lo.z}, // 1
            {hi.x, hi.y, lo.z}, // 2
            {lo.x, hi.y, lo.z}, // 3
            {lo.x, lo.y, hi.z}, // 4
            {hi.x, lo.y, hi.z}, // 5
            {hi.x, hi.y, hi.z}, // 6
            {lo.x, hi.y, hi.z}, // 7
        };

        // 12 triangles forming 6 faces (2 tris per face).
        mesh.indices = {
            0,1,2, 0,2,3,  // -Z face
            4,6,5, 4,7,6,  // +Z face
            0,4,5, 0,5,1,  // -Y face
            2,6,7, 2,7,3,  // +Y face
            0,3,7, 0,7,4,  // -X face
            1,5,6, 1,6,2,  // +X face
        };
        return mesh;
    }


    // =========================================================================
    //  Geometry Helpers (formerly in shape_volume.hpp)
    // =========================================================================




    // =========================================================================
    //  Debug mesh from Frustum (6-plane frustum → 8 corner vertices)
    // =========================================================================

    inline DebugMesh debug_mesh_from_frustum(const Frustum& frustum)
    {
        DebugMesh mesh{};

        // Intersect plane triplets to find the 8 frustum corners.
        // Plane order: Left(0), Right(1), Bottom(2), Top(3), Near(4), Far(5).
        auto intersect = [&](size_t i0, size_t i1, size_t i2) -> glm::vec3 {
            glm::vec3 p{0.0f};
            intersect_three_planes(frustum.planes[i0], frustum.planes[i1], frustum.planes[i2], p);
            return p;
        };

        // Near face corners.
        mesh.vertices.push_back(intersect(0, 2, 4)); // near-bottom-left
        mesh.vertices.push_back(intersect(1, 2, 4)); // near-bottom-right
        mesh.vertices.push_back(intersect(1, 3, 4)); // near-top-right
        mesh.vertices.push_back(intersect(0, 3, 4)); // near-top-left

        // Far face corners.
        mesh.vertices.push_back(intersect(0, 2, 5)); // far-bottom-left
        mesh.vertices.push_back(intersect(1, 2, 5)); // far-bottom-right
        mesh.vertices.push_back(intersect(1, 3, 5)); // far-top-right
        mesh.vertices.push_back(intersect(0, 3, 5)); // far-top-left

        // 12 triangles forming 6 faces.
        mesh.indices = {
            0,1,2, 0,2,3,  // near
            4,6,5, 4,7,6,  // far
            0,4,5, 0,5,1,  // bottom
            2,6,7, 2,7,3,  // top
            0,3,7, 0,7,4,  // left
            1,5,6, 1,6,2,  // right
        };
        return mesh;
    }


    // =========================================================================
    //  Debug mesh from CullingCell
    // =========================================================================

    inline DebugMesh debug_mesh_from_culling_cell(const CullingCell& cell)
    {
        if (cell.plane_count < 4) return DebugMesh{};

        // Convert CullingCell planes to a vector for vertex extraction.
        std::vector<Plane> planes{};
        planes.reserve(cell.plane_count);
        for (uint32_t i = 0; i < cell.plane_count; ++i)
        {
            planes.push_back(cell.planes[i]);
        }

        // Find vertices by intersecting plane triplets.
        const std::vector<glm::vec3> verts = convex_vertices_from_planes(planes, 1e-5f);
        if (verts.empty()) return DebugMesh{};

        // Build a simple triangle fan from centroid (approximate convex hull visualization).
        DebugMesh mesh{};
        mesh.vertices = verts;

        // Centroid.
        glm::vec3 centroid{0.0f};
        for (const glm::vec3& v : verts) centroid += v;
        centroid /= static_cast<float>(verts.size());
        const uint32_t centroid_idx = static_cast<uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back(centroid);

        // Fan triangles from centroid to each pair of adjacent vertices.
        // This is a rough visualization; for precise convex hull rendering,
        // a proper face enumeration would be needed.
        for (uint32_t i = 0; i < static_cast<uint32_t>(verts.size()); ++i)
        {
            const uint32_t next = (i + 1) % static_cast<uint32_t>(verts.size());
            mesh.indices.push_back(centroid_idx);
            mesh.indices.push_back(i);
            mesh.indices.push_back(next);
        }
        return mesh;
    }
}

#endif // SHS_HAS_JOLT
