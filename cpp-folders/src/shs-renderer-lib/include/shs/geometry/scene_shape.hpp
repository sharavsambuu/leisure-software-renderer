#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: scene_shape.hpp
    МОДУЛЬ: geometry
    ЗОРИЛГО: Нэг scene object-ийн culling-д хэрэглэх Jolt shape + transform wrapper.
            C++20 Cullable / FastCullable concept-уудыг хангана.
            Хуучин ShapeVolume-ийг бүрэн орлоно.
*/

#if defined(SHS_HAS_JOLT) && ((SHS_HAS_JOLT + 0) == 1)

#include <cstdint>

#include <glm/glm.hpp>

#include <Jolt/Jolt.h>
#include <Jolt/Math/Mat44.h>
#include <Jolt/Geometry/AABox.h>
#include <Jolt/Physics/Collision/Shape/Shape.h>

#include "shs/geometry/aabb.hpp"
#include "shs/geometry/volumes.hpp"
#include "shs/geometry/jolt_adapter.hpp"
#include "shs/geometry/jolt_shape_traits.hpp"

namespace shs
{
    struct SceneShape
    {
        JPH::ShapeRefC  shape{};
        JPH::Mat44      transform = JPH::Mat44::sIdentity();
        uint32_t        stable_id = 0;

        // -----------------------------------------------------------------
        //  Concept satisfaction: Cullable
        // -----------------------------------------------------------------

        JPH::ShapeRefC jolt_shape() const
        {
            return shape;
        }

        JPH::Mat44 world_transform() const
        {
            return transform;
        }

        // -----------------------------------------------------------------
        //  Concept satisfaction: FastCullable (HasBoundingSphere)
        // -----------------------------------------------------------------

        /// Returns conservative bounding sphere in SHS LH space.
        Sphere bounding_sphere() const
        {
            if (!shape) return Sphere{};

            const JPH::AABox jph_aabb = shape->GetWorldSpaceBounds(transform, JPH::Vec3::sReplicate(1.0f));
            const JPH::Vec3 jph_center = jph_aabb.GetCenter();
            const float radius = jph_aabb.GetExtent().Length();

            return Sphere{
                jolt::to_glm(jph_center),
                radius
            };
        }

        // -----------------------------------------------------------------
        //  Concept satisfaction: HasWorldAABB
        // -----------------------------------------------------------------

        /// Returns world-space AABB in SHS LH space.
        AABB world_aabb() const
        {
            if (!shape) return AABB{};

            const JPH::AABox jph_aabb = shape->GetWorldSpaceBounds(transform, JPH::Vec3::sReplicate(1.0f));
            return jolt::to_glm(jph_aabb);
        }
    };

    // Static assertions that SceneShape satisfies our concepts.
    static_assert(Cullable<SceneShape>, "SceneShape must satisfy Cullable");
    static_assert(FastCullable<SceneShape>, "SceneShape must satisfy FastCullable");
    static_assert(HasWorldAABB<SceneShape>, "SceneShape must satisfy HasWorldAABB");
}

#endif // SHS_HAS_JOLT
