#pragma once

/*
    SHS RENDERER SAN

    FILE: render_path_recipe.hpp
    MODULE: pipeline
    PURPOSE: Data-first recipe schema for dynamic render path composition.
*/


#include <cstdint>
#include <string>
#include <vector>

#include "shs/frame/technique_mode.hpp"
#include "shs/pipeline/render_path_runtime_state.hpp"
#include "shs/rhi/core/backend.hpp"

namespace shs
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
        bool required = true;
    };

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

        bool wants_shadows = true;
        bool strict_validation = true;
    };

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
                {"shadow_map", true},
                {"depth_prepass", false},
                {"light_culling", false},
                {"pbr_forward_plus", true},
                {"tonemap", true},
                {"motion_blur", false}
            };
        }
        else
        {
            recipe.name = "soft_shadow_culling_sw_default";
            recipe.render_technique = RenderPathRenderingTechnique::ForwardLit;
            recipe.technique_mode = TechniqueMode::Forward;
            recipe.pass_chain = {
                {"shadow_map", true},
                {"pbr_forward", true},
                {"tonemap", true},
                {"motion_blur", false}
            };
        }
        return recipe;
    }
}

