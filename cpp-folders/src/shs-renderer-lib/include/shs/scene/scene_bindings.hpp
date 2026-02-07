#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: scene_bindings.hpp
    МОДУЛЬ: scene
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн scene модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <glm/glm.hpp>

#include "shs/scene/scene_types.hpp"

namespace shs
{
    inline RenderItem make_render_item(
        MeshHandle mesh,
        MaterialHandle mat,
        const glm::vec3& pos = glm::vec3(0.0f),
        const glm::vec3& scl = glm::vec3(1.0f),
        const glm::vec3& rot_euler = glm::vec3(0.0f)
    )
    {
        RenderItem it{};
        it.mesh = mesh;
        it.mat = mat;
        it.tr.pos = pos;
        it.tr.scl = scl;
        it.tr.rot_euler = rot_euler;
        return it;
    }

    inline void bind_material(RenderItem& it, MaterialHandle mat)
    {
        it.mat = mat;
    }

    inline void bind_mesh(RenderItem& it, MeshHandle mesh)
    {
        it.mesh = mesh;
    }
}

