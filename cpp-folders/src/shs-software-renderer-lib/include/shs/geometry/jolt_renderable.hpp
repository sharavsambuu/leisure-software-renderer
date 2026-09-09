#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: jolt_renderable.hpp
    МОДУЛЬ: geometry
    ЗОРИЛГО: Jolt shape-ийг материалтай холбож, рендерерт бэлэн болгох.
            SceneObject-ийн оронд ашиглагдах ирээдүйн суурь.
*/

#if defined(SHS_HAS_JOLT) && ((SHS_HAS_JOLT + 0) == 1)

#include <cstdint>
#include <string>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <Jolt/Jolt.h>
#include <Jolt/Math/Mat44.h>

#include "shs/geometry/scene_shape.hpp"
#include "shs/resources/resource_registry.hpp"
#include "shs/scene/scene_bindings.hpp"

namespace shs
{
    /**
     * @brief High-level renderable object based on Jolt Physics shape.
     * Integrates geometry (JPH::Shape), transform (JPH::Mat44), and material.
     */
    struct JoltRenderable
    {
        SceneShape          geometry{};
        MaterialAssetHandle material = 0;
        MeshAssetHandle     visual_mesh = 0; // Optional: separate high-poly mesh. If 0, use debug/proxy mesh.
        std::string         name{};
        bool                visible = true;
        bool                casts_shadow = true;

        inline uint64_t object_id() const
        {
            return geometry.stable_id;
        }

        /**
         * @brief Convert this JoltRenderable to a low-level RenderItem.
         * If visual_mesh is 0, this might need external logic (e.g. from debug draw)
         * to produce a renderable mesh from the Jolt shape.
         */
        inline RenderItem to_render_item() const
        {
            RenderItem ri{};
            ri.mesh = visual_mesh;
            ri.mat = material;

            const glm::mat4 m_shs = jolt::to_glm(geometry.transform);
            ri.tr.pos = glm::vec3(m_shs[3]);

            glm::vec3 axis_x = glm::vec3(m_shs[0]);
            glm::vec3 axis_y = glm::vec3(m_shs[1]);
            glm::vec3 axis_z = glm::vec3(m_shs[2]);

            glm::vec3 scale{
                glm::length(axis_x),
                glm::length(axis_y),
                glm::length(axis_z)
            };

            if (scale.x <= 1e-6f) scale.x = 1.0f;
            if (scale.y <= 1e-6f) scale.y = 1.0f;
            if (scale.z <= 1e-6f) scale.z = 1.0f;

            axis_x /= scale.x;
            axis_y /= scale.y;
            axis_z /= scale.z;

            glm::mat3 rot_m{};
            rot_m[0] = axis_x;
            rot_m[1] = axis_y;
            rot_m[2] = axis_z;
            if (glm::determinant(rot_m) < 0.0f)
            {
                // Keep a proper rotation matrix and preserve a signed scale component.
                scale.z = -scale.z;
                rot_m[2] = -rot_m[2];
            }

            const glm::quat rot_q = glm::normalize(glm::quat_cast(rot_m));
            ri.tr.rot_euler = glm::eulerAngles(rot_q);
            ri.tr.scl = scale;
            
            ri.object_id = object_id();
            ri.visible = visible;
            ri.casts_shadow = casts_shadow;
            
            return ri;
        }
    };
}

#endif // SHS_HAS_JOLT
