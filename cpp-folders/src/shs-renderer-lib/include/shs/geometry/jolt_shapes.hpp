#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: jolt_shapes.hpp
    МОДУЛЬ: geometry
    ЗОРИЛГО: Jolt shape-ууд үүсгэх factory функцүүд.
            SHS LH space дээрх параметрүүдийг хүлээн авч,
            Jolt RH space дээрх shape руу хөрвүүлнэ.
            Гэрлийн volume shape builder-уудыг мөн агуулна.
*/

#if defined(SHS_HAS_JOLT) && ((SHS_HAS_JOLT + 0) == 1)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/CylinderShape.h>
#include <Jolt/Physics/Collision/Shape/TaperedCapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>

#include "shs/geometry/jolt_adapter.hpp"

// Forward declare light types to avoid circular dependency.
namespace shs { struct SpotLight; struct RectAreaLight; struct TubeAreaLight; }

namespace shs::jolt
{
    // =========================================================================
    //  Basic shape factories
    //  Parameters are in SHS LH space. Shape intrinsic geometry is
    //  symmetric, so no coordinate flip needed for radius/half_extents.
    // =========================================================================

    inline JPH::ShapeRefC make_sphere(float radius)
    {
        return new JPH::SphereShape(std::max(radius, 0.001f));
    }

    inline JPH::ShapeRefC make_box(const glm::vec3& half_extents)
    {
        return new JPH::BoxShape(JPH::Vec3(
            std::max(half_extents.x, 0.001f),
            std::max(half_extents.y, 0.001f),
            std::max(half_extents.z, 0.001f)));
    }

    inline JPH::ShapeRefC make_capsule(float half_height, float radius)
    {
        return new JPH::CapsuleShape(
            std::max(half_height, 0.001f),
            std::max(radius, 0.001f));
    }

    inline JPH::ShapeRefC make_cylinder(float half_height, float radius)
    {
        return new JPH::CylinderShape(
            std::max(half_height, 0.001f),
            std::max(radius, 0.001f));
    }

    inline JPH::ShapeRefC make_tapered_capsule(float half_height, float top_radius, float bottom_radius)
    {
        JPH::TaperedCapsuleShapeSettings settings(
            std::max(half_height, 0.001f),
            std::max(top_radius, 0.001f),
            std::max(bottom_radius, 0.001f));
        auto result = settings.Create();
        if (result.HasError())
        {
            // Fallback to regular capsule with max radius.
            return make_capsule(half_height, std::max(top_radius, bottom_radius));
        }
        return result.Get();
    }

    inline JPH::ShapeRefC make_convex_hull(std::span<const glm::vec3> vertices_shs)
    {
        // Convert all vertices from SHS LH to Jolt RH.
        std::vector<JPH::Vec3> jph_verts{};
        jph_verts.reserve(vertices_shs.size());
        for (const glm::vec3& v : vertices_shs)
        {
            jph_verts.push_back(to_jph(v));
        }
        JPH::ConvexHullShapeSettings settings(
            jph_verts.data(),
            static_cast<int>(jph_verts.size()));
        auto result = settings.Create();
        if (result.HasError())
        {
            // Fallback: bounding sphere around vertices.
            float max_dist = 0.0f;
            for (const JPH::Vec3& v : jph_verts)
            {
                max_dist = std::max(max_dist, v.Length());
            }
            return make_sphere(max_dist);
        }
        return result.Get();
    }


    // =========================================================================
    //  Light volume shape builders
    //  All produce shapes centered at origin. The caller provides
    //  the world transform separately (via SceneShape).
    // =========================================================================

    /// Point light → sphere of given range.
    inline JPH::ShapeRefC make_point_light_volume(float range)
    {
        return make_sphere(std::max(range, 0.001f));
    }

    /// Spot light → cone-like convex hull approximation.
    /// Builds a discretized cone pointing along -Y (Jolt space),
    /// to be rotated by the light's orientation at the call site.
    inline JPH::ShapeRefC make_spot_light_volume(float range, float outer_angle_rad, uint32_t segments = 12)
    {
        const float r = std::max(range, 0.001f);
        const float half_angle = std::clamp(outer_angle_rad, 0.01f, glm::half_pi<float>() - 0.01f);
        const float base_radius = r * std::tan(half_angle);

        std::vector<glm::vec3> verts{};
        verts.reserve(segments + 1);

        // Apex at origin (in SHS LH space, light is at origin).
        verts.push_back(glm::vec3(0.0f));

        // Base circle at distance `range` along the light direction.
        // The direction will be handled by the transform, so we build
        // along +Z (SHS forward) here.
        for (uint32_t i = 0; i < segments; ++i)
        {
            const float theta = (2.0f * glm::pi<float>() * static_cast<float>(i)) / static_cast<float>(segments);
            verts.push_back(glm::vec3(
                base_radius * std::cos(theta),
                base_radius * std::sin(theta),
                r));
        }

        return make_convex_hull(verts);
    }

    /// Rect area light → oriented box.
    inline JPH::ShapeRefC make_rect_area_light_volume(const glm::vec2& half_extents, float range)
    {
        return make_box(glm::vec3(
            std::max(half_extents.x, 0.001f),
            std::max(half_extents.y, 0.001f),
            std::max(range, 0.001f) * 0.5f));
    }

    /// Tube area light → capsule.
    inline JPH::ShapeRefC make_tube_area_light_volume(float half_length, float radius)
    {
        return make_capsule(
            std::max(half_length, 0.001f),
            std::max(radius, 0.001f));
    }
}

#endif // SHS_HAS_JOLT
