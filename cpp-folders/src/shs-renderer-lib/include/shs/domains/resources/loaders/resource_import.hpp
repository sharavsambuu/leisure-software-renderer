#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: resource_import.hpp
    МОДУЛЬ: resources
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн resources модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <string>

#include "shs/domains/resources/loaders/mesh_loader_assimp.hpp"
#include "shs/domains/resources/resource_registry.hpp"

namespace shs
{
    inline MeshAssetHandle import_mesh_assimp(
        ResourceRegistry& reg,
        const std::string& path,
        const std::string& key = {},
        const MeshLoadOptions& opt = {}
    )
    {
        MeshData mesh = load_mesh_assimp_first(path, opt);
        if (mesh.empty()) return 0;
        return reg.add_mesh(std::move(mesh), key.empty() ? path : key);
    }

}
