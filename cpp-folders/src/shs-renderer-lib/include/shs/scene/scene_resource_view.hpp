#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: scene_resource_view.hpp
    МОДУЛЬ: scene
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн scene модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include "shs/resources/material.hpp"
#include "shs/resources/storage/resource_registry.hpp"
#include "shs/scene/scene_types.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace scene
    {
    struct SceneResourceView
    {
        // Renderer projection (step 4.3): resolves scene asset handles into
        // registry pointers PER CALL. Returned pointers are valid only until
        // the next registry mutation (add_*/clear()) — projections must
        // never cache them across frames or across registry writes. Handle 0
        // ("unbound") resolves to nullptr by policy.
        const ResourceRegistry* resources = nullptr;

        const MeshData* mesh(const RenderItem& item) const
        {
            if (!resources) return nullptr;
            return resources->get_mesh((MeshAssetHandle)item.mesh);
        }

        const MaterialData* material(const RenderItem& item) const
        {
            if (!resources) return nullptr;
            return resources->get_material((MaterialAssetHandle)item.mat);
        }
    };

    } // inline namespace scene
}
