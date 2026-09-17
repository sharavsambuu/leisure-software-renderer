#pragma once

/*
    SHS RENDERER SAN

    FILE: resources.contract.hpp
    MODULE: domains/resources
    PURPOSE: CORE 1. TYPES — the resources pod's discoverability seam (R5a P3.7).
             Asset DATA (mesh/texture/material + IBL values) is the pod spine.
             Registry CLASSES (ResourceRegistry/AssetRegistry: unordered_map
             keyed stores mutated via methods) are edge candidates — they stay
             visible here but migrate to an edge subfolder in R5b (same ruling
             as gfx rt_registry). The assimp loader is edge IO (same class as
             the evicted SDL loaders); primitive_import stays (pure builders).
             R5b convergence executed: the assimp adapter re-export
             (adapters/resource_import.hpp) is no longer re-exported here —
             include shs/resources/adapters/resource_import.hpp (old path:
             shs/domains/resources/edge/resource_import.hpp) directly. The
             contract now compiles with zero optional SDKs.
*/

#include "shs/resources/ibl.hpp"
#include "shs/resources/loaders/primitive_import.hpp"
#include "shs/resources/material.hpp"
#include "shs/resources/mesh.hpp"
#include "shs/resources/storage/resource_registry.hpp"
#include "shs/resources/texture.hpp"

namespace shs::resources
{
    // --- asset data (pod spine) ---
    using shs::TextureAssetHandle;
    using shs::Texture2DData;
    using shs::MeshAssetHandle;
    using shs::MeshData;
    using shs::MaterialAssetHandle;
    using shs::MaterialData;

    // --- IBL values ---
    using shs::CubeMapLinear;
    using shs::PrefilteredSpecular;
    using shs::EnvIBL;

    // --- edge candidates (visible, migrating in R5b) ---
    using shs::ResourceRegistry;
    // NOTE: AssetRegistry (parallel fork: MeshHandle/TextureHandle names that
    // do not exist) does not compile and has zero consumers; R5b converges
    // or deletes it. Deliberately NOT included here.
} // namespace shs::resources
