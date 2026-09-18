#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: render_path_recipe.hpp
    МОДУЛЬ: pipeline
    ЗОРИЛГО: Динамик рендер замыг (render path) бүрдүүлэх өгөгдөл төвтэй (data-first) жор ба загвар.
*/


#include <cstdint>
#include <string>
#include <vector>

#include "shs/render/frame/technique_mode.hpp"
#include "shs/render/frame/backend_type.hpp"
#include "shs/renderpath/planning/pass_id.hpp"
#include "shs/renderpath/execution/render_path_runtime_state.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace renderpath
    {
    enum class RenderPathLightVolumeProvider : uint8_t
    {
        Default = 0,
        JoltShapeVolumes = 1,
        ClusteredGrid = 2
    };

    inline const char* render_path_light_volume_provider_name(RenderPathLightVolumeProvider p)
    {
        switch (p)
        {
            case RenderPathLightVolumeProvider::Default: return "default";
            case RenderPathLightVolumeProvider::JoltShapeVolumes: return "jolt_shape_volumes";
            case RenderPathLightVolumeProvider::ClusteredGrid: return "clustered_grid";
        }
        return "unknown";
    }

    enum class RenderPathCullingMode : uint8_t
    {
        None = 0,
        Frustum = 1,
        FrustumAndOcclusion = 2,
        FrustumAndOptionalOcclusion = 3
    };

    inline const char* render_path_culling_mode_name(RenderPathCullingMode mode)
    {
        switch (mode)
        {
            case RenderPathCullingMode::None: return "none";
            case RenderPathCullingMode::Frustum: return "frustum";
            case RenderPathCullingMode::FrustumAndOcclusion: return "frustum+occlusion";
            case RenderPathCullingMode::FrustumAndOptionalOcclusion: return "frustum+optional_occlusion";
        }
        return "unknown";
    }

    inline bool render_path_culling_requires_occlusion(RenderPathCullingMode mode)
    {
        return mode == RenderPathCullingMode::FrustumAndOcclusion;
    }

    inline bool render_path_culling_allows_occlusion(RenderPathCullingMode mode)
    {
        return mode == RenderPathCullingMode::FrustumAndOcclusion ||
               mode == RenderPathCullingMode::FrustumAndOptionalOcclusion;
    }

    enum class RenderPathRenderingTechnique : uint8_t
    {
        ForwardLit = 0,
        ForwardPlus = 1,
        Deferred = 2
    };

    inline const char* render_path_rendering_technique_name(RenderPathRenderingTechnique t)
    {
        switch (t)
        {
            case RenderPathRenderingTechnique::ForwardLit: return "forward_lit";
            case RenderPathRenderingTechnique::ForwardPlus: return "forward_plus";
            case RenderPathRenderingTechnique::Deferred: return "deferred";
        }
        return "unknown";
    }

    struct RenderPathPassEntry
    {
        std::string id{};
        PassId pass_id = PassId::Unknown;
        bool required = true;

        // Value semantics (pod test kit requires snapshot equality).
        bool operator==(const RenderPathPassEntry&) const = default;
    };

    inline RenderPathPassEntry make_render_path_pass_entry(PassId pass_id, bool required)
    {
        RenderPathPassEntry out{};
        out.id = pass_id_string(pass_id);
        out.pass_id = pass_id;
        out.required = required;
        return out;
    }

    // Consumer-owned (open-id) pass entry: the name is the registration key, so
    // it must be the exact name interned through `PassIdRegistry`
    // (`PassFactoryRegistry::intern_pass_id`). Builtin passes may use this
    // overload too; the name must then be the builtin spelling.
    inline RenderPathPassEntry make_render_path_pass_entry(std::string id, PassId pass_id, bool required)
    {
        RenderPathPassEntry out{};
        out.id = std::move(id);
        out.pass_id = pass_id;
        out.required = required;
        return out;
    }

    struct RenderPathRecipe
    {
        std::string name{};

        RenderBackendType backend = RenderBackendType::Software;
        RenderPathLightVolumeProvider light_volume_provider = RenderPathLightVolumeProvider::Default;
        RenderPathCullingMode view_culling = RenderPathCullingMode::Frustum;
        RenderPathCullingMode shadow_culling = RenderPathCullingMode::FrustumAndOptionalOcclusion;
        RenderPathRenderingTechnique render_technique = RenderPathRenderingTechnique::ForwardLit;
        TechniqueMode technique_mode = TechniqueMode::Forward;

        std::vector<RenderPathPassEntry> pass_chain{};
        RenderPathRuntimeState runtime_defaults{};

        // Resource-layout knobs that should come from recipe/preset instead of demo constants.
        uint32_t light_tile_size = 16u;
        uint32_t cluster_z_slices = 16u;

        bool wants_shadows = true;
        bool strict_validation = true;

        // Value semantics (pod test kit requires snapshot equality).
        bool operator==(const RenderPathRecipe&) const = default;
    };

    // Technique-mode transition table (C2.2, P1 pure-leaf placement): the
    // closed-enum image of a technique mode in rendering-technique space.
    // The compiler asserts a compiled plan's technique is exactly this table
    // image of its mode (Rule 17, value invariant); presets and pod arrows
    // derive their pairs through this table, so the assert never fires on
    // house input. Moved here (was render_path_presets.hpp) so the pure leaf
    // compiler header can state the invariant without an include cycle.
    inline RenderPathRenderingTechnique render_path_rendering_technique_for_mode(TechniqueMode mode)
    {
        switch (mode)
        {
            case TechniqueMode::Forward:
                return RenderPathRenderingTechnique::ForwardLit;
            case TechniqueMode::ForwardPlus:
            case TechniqueMode::ClusteredForward:
                return RenderPathRenderingTechnique::ForwardPlus;
            case TechniqueMode::Deferred:
            case TechniqueMode::TiledDeferred:
                return RenderPathRenderingTechnique::Deferred;
        }
        return RenderPathRenderingTechnique::ForwardPlus;
    }

    inline RenderPathRecipe make_default_soft_shadow_culling_recipe(RenderBackendType backend)
    {
        RenderPathRecipe recipe{};
        recipe.backend = backend;
        recipe.light_volume_provider = RenderPathLightVolumeProvider::JoltShapeVolumes;
        recipe.view_culling = RenderPathCullingMode::FrustumAndOcclusion;
        recipe.shadow_culling = RenderPathCullingMode::FrustumAndOptionalOcclusion;
        recipe.runtime_defaults.view_occlusion_enabled = true;
        recipe.runtime_defaults.shadow_occlusion_enabled = false;
        recipe.runtime_defaults.debug_aabb = false;
        recipe.runtime_defaults.lit_mode = true;
        recipe.runtime_defaults.enable_shadows = true;
        recipe.wants_shadows = true;
        recipe.strict_validation = true;

        if (backend == RenderBackendType::Vulkan)
        {
            recipe.name = "soft_shadow_culling_vk_default";
            recipe.render_technique = RenderPathRenderingTechnique::ForwardPlus;
            recipe.technique_mode = TechniqueMode::ForwardPlus;
            recipe.pass_chain = {
                make_render_path_pass_entry(PassId::ShadowMap, true),
                make_render_path_pass_entry(PassId::DepthPrepass, false),
                make_render_path_pass_entry(PassId::LightCulling, false),
                make_render_path_pass_entry(PassId::PBRForwardPlus, true),
                make_render_path_pass_entry(PassId::Tonemap, true),
                make_render_path_pass_entry(PassId::MotionBlur, false)
            };
        }
        else
        {
            recipe.name = "soft_shadow_culling_sw_default";
            recipe.render_technique = RenderPathRenderingTechnique::ForwardLit;
            recipe.technique_mode = TechniqueMode::Forward;
            recipe.pass_chain = {
                make_render_path_pass_entry(PassId::ShadowMap, true),
                make_render_path_pass_entry(PassId::PBRForward, true),
                make_render_path_pass_entry(PassId::Tonemap, true),
                make_render_path_pass_entry(PassId::MotionBlur, false)
            };
        }
        return recipe;
    }

    } // inline namespace renderpath
}
