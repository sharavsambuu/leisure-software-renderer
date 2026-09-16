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
*/

#include "shs/domains/resources/ibl.hpp"
#include "shs/domains/resources/loaders/primitive_import.hpp"
#include "shs/domains/resources/loaders/resource_import.hpp"
#include "shs/domains/resources/material.hpp"
#include "shs/domains/resources/mesh.hpp"
#include "shs/domains/resources/resource_registry.hpp"
#include "shs/domains/resources/texture.hpp"

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
