#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: resource_registry.hpp
    МОДУЛЬ: resources
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн resources модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <memory_resource>
#include <string>
#include <vector>

#include "shs/containers/flat_map.hpp"
#include "shs/resources/material.hpp"
#include "shs/resources/mesh.hpp"
#include "shs/resources/texture.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace resources
    {
    /*
        IDENTITY POLICY (step 4.3, engine_domain_separation_migration.md):
        - Handles are 1-based indices; 0 = "unbound"; every getter returns
          nullptr for 0 or out-of-range handles (stale-handle null rule).
        - Append-only: add_* never invalidates existing handles. A duplicate
          key deliberately REBINDS the key lookup to the newest asset
          (last-wins); the previously returned handle keeps resolving to the
          older asset, so already-bound scene items stay intact.
        - Per-asset deletion is NOT supported (index handles would silently
          invalidate sibling handles). The only reset is clear(), which
          bumps generation(): handles minted before clear() are stale —
          re-derive them via find_* after clearing (a stale handle may alias
          a re-added asset's slot, so generation must be checked).
        - Pointers handed out by get_* are valid only until the next
          registry mutation (vector reallocation); renderer projections must
          re-resolve per frame (SceneResourceView) and never cache them.
    */
    class ResourceRegistry
    {
    public:
        // Cold-registry container migration (W-E, 2026-09-17): keyed state is
        // node-free (shs::containers::FlatMap, §7.2 rule 5/6 — shared lib
        // utility, no private copies). Defaults to the default pmr resource;
        // pass an arena for arena-scoped lifetimes.
        explicit ResourceRegistry(
            std::pmr::memory_resource* resource = std::pmr::get_default_resource())
            : mesh_by_key_{resource}, texture_by_key_{resource},
              material_by_key_{resource} {}

        // Identity epoch (step 4.3): bumped by every clear(). Handles minted
        // in an older generation are stale and must be re-derived via find_*.
        uint64_t generation() const { return generation_; }
        MeshAssetHandle add_mesh(MeshData mesh, const std::string& key = {})
        {
            meshes_.push_back(std::move(mesh));
            const MeshAssetHandle h = (MeshAssetHandle)meshes_.size();
            if (!key.empty()) mesh_by_key_.insert_or_assign(key, h);
            return h;
        }

        TextureAssetHandle add_texture(Texture2DData tex, const std::string& key = {})
        {
            textures_.push_back(std::move(tex));
            const TextureAssetHandle h = (TextureAssetHandle)textures_.size();
            if (!key.empty()) texture_by_key_.insert_or_assign(key, h);
            return h;
        }

        MaterialAssetHandle add_material(MaterialData mat, const std::string& key = {})
        {
            materials_.push_back(std::move(mat));
            const MaterialAssetHandle h = (MaterialAssetHandle)materials_.size();
            if (!key.empty()) material_by_key_.insert_or_assign(key, h);
            return h;
        }

        const MeshData* get_mesh(MeshAssetHandle h) const
        {
            if (h == 0 || h > (MeshAssetHandle)meshes_.size()) return nullptr;
            return &meshes_[(size_t)h - 1];
        }

        MeshData* get_mesh(MeshAssetHandle h)
        {
            if (h == 0 || h > (MeshAssetHandle)meshes_.size()) return nullptr;
            return &meshes_[(size_t)h - 1];
        }

        const Texture2DData* get_texture(TextureAssetHandle h) const
        {
            if (h == 0 || h > (TextureAssetHandle)textures_.size()) return nullptr;
            return &textures_[(size_t)h - 1];
        }

        Texture2DData* get_texture(TextureAssetHandle h)
        {
            if (h == 0 || h > (TextureAssetHandle)textures_.size()) return nullptr;
            return &textures_[(size_t)h - 1];
        }

        const MaterialData* get_material(MaterialAssetHandle h) const
        {
            if (h == 0 || h > (MaterialAssetHandle)materials_.size()) return nullptr;
            return &materials_[(size_t)h - 1];
        }

        MaterialData* get_material(MaterialAssetHandle h)
        {
            if (h == 0 || h > (MaterialAssetHandle)materials_.size()) return nullptr;
            return &materials_[(size_t)h - 1];
        }

        MeshAssetHandle find_mesh(const std::string& key) const
        {
            const MeshAssetHandle* v = mesh_by_key_.find(key);
            return v ? *v : 0u;
        }

        TextureAssetHandle find_texture(const std::string& key) const
        {
            const TextureAssetHandle* v = texture_by_key_.find(key);
            return v ? *v : 0u;
        }

        MaterialAssetHandle find_material(const std::string& key) const
        {
            const MaterialAssetHandle* v = material_by_key_.find(key);
            return v ? *v : 0u;
        }

        size_t mesh_count() const { return meshes_.size(); }
        size_t texture_count() const { return textures_.size(); }
        size_t material_count() const { return materials_.size(); }

        void clear()
        {
            meshes_.clear();
            textures_.clear();
            materials_.clear();
            mesh_by_key_.clear();
            texture_by_key_.clear();
            material_by_key_.clear();
            generation_ += 1; // identity epoch: pre-clear handles are stale
        }

    private:
        uint64_t generation_ = 0;
        std::vector<MeshData> meshes_{};
        std::vector<Texture2DData> textures_{};
        std::vector<MaterialData> materials_{};
        containers::FlatMap<std::string, MeshAssetHandle> mesh_by_key_;
        containers::FlatMap<std::string, TextureAssetHandle> texture_by_key_;
        containers::FlatMap<std::string, MaterialAssetHandle> material_by_key_;
    };

    } // inline namespace resources
}
