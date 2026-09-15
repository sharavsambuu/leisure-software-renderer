#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: scene_resource_view.hpp
    МОДУЛЬ: scene
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн scene модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include "shs/domains/resources/material.hpp"
#include "shs/domains/resources/resource_registry.hpp"
#include "shs/domains/scene/scene_types.hpp"

namespace shs
{
    struct SceneResourceView
    {
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
}
