#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: asset_manager.hpp
    МОДУЛЬ: execution/platform (edge orchestration; moved from domains/resources P0.2)
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн resources модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <string>

#include "shs/domains/resources/loaders/resource_import.hpp"
#include "shs/execution/platform/loaders/texture_loader_sdl.hpp"
#include "shs/domains/resources/resource_registry.hpp"

namespace shs
{
    class AssetManager
    {
    public:
        ResourceRegistry& registry() { return registry_; }
        const ResourceRegistry& registry() const { return registry_; }

        MeshAssetHandle load_mesh(const std::string& path, const std::string& key = {})
        {
            return import_mesh_assimp(registry_, path, key);
        }

        TextureAssetHandle load_texture(const std::string& path, const std::string& key = {}, bool flip_y = true)
        {
            return import_texture_sdl(registry_, path, key, flip_y);
        }

    private:
        ResourceRegistry registry_{};
    };
}
