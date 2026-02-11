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
            
            const JPH::Vec3 pos = geometry.transform.GetTranslation();
            const JPH::Quat rot_q = geometry.transform.GetRotation().GetQuaternion();
            const JPH::Vec3 rot = rot_q.GetEulerAngles();
            
            const glm::vec3 pos_shs = jolt::to_glm(pos);
            const glm::vec3 rot_shs = glm::vec3(rot.GetX(), rot.GetY(), -rot.GetZ());
            
            ri.tr.pos = pos_shs;
            ri.tr.rot_euler = rot_shs;
            ri.tr.scl = glm::vec3(1.0f);
            
            ri.object_id = object_id();
            ri.visible = visible;
            ri.casts_shadow = casts_shadow;
            
            return ri;
        }
    };
}

#endif // SHS_HAS_JOLT
