#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: resource_import.hpp
    МОДУЛЬ: resources
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн resources модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <string>

#include "shs/resources/loaders/mesh_loader_assimp.hpp"
#include "shs/resources/loaders/texture_loader_sdl.hpp"
#include "shs/resources/resource_registry.hpp"

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

    inline TextureAssetHandle import_texture_sdl(
        ResourceRegistry& reg,
        const std::string& path,
        const std::string& key = {},
        bool flip_y = true
    )
    {
        Texture2DData tex = load_texture2d_sdl_image(path, flip_y);
        if (!tex.valid()) return 0;
        return reg.add_texture(std::move(tex), key.empty() ? path : key);
    }
}
