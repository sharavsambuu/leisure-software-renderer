#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: asset_registry.hpp
    МОДУЛЬ: resources
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн resources модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <string>
#include <unordered_map>
#include <vector>

#include "shs/resources/material.hpp"
#include "shs/resources/mesh.hpp"
#include "shs/resources/texture.hpp"

namespace shs
{
    class AssetRegistry
    {
    public:
        MeshHandle add_mesh(MeshData mesh, const std::string& key = {})
        {
            meshes_.push_back(std::move(mesh));
            const MeshHandle h = (MeshHandle)meshes_.size();
            if (!key.empty()) mesh_by_key_[key] = h;
            return h;
        }

        TextureHandle add_texture(Texture2DData tex, const std::string& key = {})
        {
            textures_.push_back(std::move(tex));
            const TextureHandle h = (TextureHandle)textures_.size();
            if (!key.empty()) texture_by_key_[key] = h;
            return h;
        }

        MaterialHandle add_material(MaterialData mat, const std::string& key = {})
        {
            materials_.push_back(std::move(mat));
            const MaterialHandle h = (MaterialHandle)materials_.size();
            if (!key.empty()) material_by_key_[key] = h;
            return h;
        }

        const MeshData* get_mesh(MeshHandle h) const
        {
            if (h == 0 || h > (MeshHandle)meshes_.size()) return nullptr;
            return &meshes_[(size_t)h - 1];
        }

        MeshData* get_mesh(MeshHandle h)
        {
            if (h == 0 || h > (MeshHandle)meshes_.size()) return nullptr;
            return &meshes_[(size_t)h - 1];
        }

        const Texture2DData* get_texture(TextureHandle h) const
        {
            if (h == 0 || h > (TextureHandle)textures_.size()) return nullptr;
            return &textures_[(size_t)h - 1];
        }

        Texture2DData* get_texture(TextureHandle h)
        {
            if (h == 0 || h > (TextureHandle)textures_.size()) return nullptr;
            return &textures_[(size_t)h - 1];
        }

        const MaterialData* get_material(MaterialHandle h) const
        {
            if (h == 0 || h > (MaterialHandle)materials_.size()) return nullptr;
            return &materials_[(size_t)h - 1];
        }

        MaterialData* get_material(MaterialHandle h)
        {
            if (h == 0 || h > (MaterialHandle)materials_.size()) return nullptr;
            return &materials_[(size_t)h - 1];
        }

        MeshHandle find_mesh(const std::string& key) const
        {
            const auto it = mesh_by_key_.find(key);
            return it == mesh_by_key_.end() ? 0u : it->second;
        }

        TextureHandle find_texture(const std::string& key) const
        {
            const auto it = texture_by_key_.find(key);
            return it == texture_by_key_.end() ? 0u : it->second;
        }

        MaterialHandle find_material(const std::string& key) const
        {
            const auto it = material_by_key_.find(key);
            return it == material_by_key_.end() ? 0u : it->second;
        }

        size_t mesh_count() const noexcept { return meshes_.size(); }
        size_t texture_count() const noexcept { return textures_.size(); }
        size_t material_count() const noexcept { return materials_.size(); }

    private:
        std::vector<MeshData> meshes_{};
        std::vector<Texture2DData> textures_{};
        std::vector<MaterialData> materials_{};
        std::unordered_map<std::string, MeshHandle> mesh_by_key_{};
        std::unordered_map<std::string, TextureHandle> texture_by_key_{};
        std::unordered_map<std::string, MaterialHandle> material_by_key_{};
    };
}
