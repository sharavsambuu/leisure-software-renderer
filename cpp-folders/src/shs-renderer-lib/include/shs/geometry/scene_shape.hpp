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

        /// SHS LH орон зай дахь аюулгүй хамарсан бөмбөрцгийг (bounding sphere) буцаана.
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

        /// SHS LH орон зай дахь ертөнцийн хязгаарын хайрцгийг (world-space AABB) буцаана.
        AABB world_aabb() const
        {
            if (!shape) return AABB{};

            const JPH::AABox jph_aabb = shape->GetWorldSpaceBounds(transform, JPH::Vec3::sReplicate(1.0f));
            return jolt::to_glm(jph_aabb);
        }
    };

    // SceneShape нь Cullable болон FastCullable дүрмүүдийг (concepts) хангасан байхыг батлах статик шалгалт.
    static_assert(Cullable<SceneShape>, "SceneShape нь Cullable концептийг хангасан байх ёстой");
    static_assert(FastCullable<SceneShape>, "SceneShape нь FastCullable концептийг хангасан байх ёстой");
    static_assert(HasWorldAABB<SceneShape>, "SceneShape нь HasWorldAABB концептийг хангасан байх ёстой");
}

#endif // SHS_HAS_JOLT
