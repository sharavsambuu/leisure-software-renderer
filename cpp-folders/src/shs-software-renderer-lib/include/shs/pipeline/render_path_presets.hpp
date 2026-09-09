#pragma once

/*
    SHS RENDERER SAN

    FILE: render_path_presets.hpp
    MODULE: pipeline
    PURPOSE: Built-in render-path presets and shared mode defaults.
*/


#include <array>
#include <string>
#include <string_view>
#include <vector>

#include "shs/frame/technique_mode.hpp"
#include "shs/lighting/light_culling_mode.hpp"
#include "shs/pipeline/render_path_recipe.hpp"
#include "shs/pipeline/render_path_registry.hpp"
#include "shs/pipeline/technique_profile.hpp"
#include "shs/rhi/core/backend.hpp"

namespace shs
{
    enum class RenderPathPreset : uint8_t
    {
        Forward = 0,
        ForwardPlus = 1,
        Deferred = 2,
        TiledDeferred = 3,
        ClusteredForward = 4
    };

    inline const char* render_path_preset_name(RenderPathPreset preset)
    {
        switch (preset)
        {
            case RenderPathPreset::Forward: return "forward";
            case RenderPathPreset::ForwardPlus: return "forward_plus";
            case RenderPathPreset::Deferred: return "deferred";
            case RenderPathPreset::TiledDeferred: return "tiled_deferred";
            case RenderPathPreset::ClusteredForward: return "clustered_forward";
        }
        return "forward_plus";
    }

    inline TechniqueMode render_path_preset_mode(RenderPathPreset preset)
    {
        switch (preset)
        {
            case RenderPathPreset::Forward: return TechniqueMode::Forward;
            case RenderPathPreset::ForwardPlus: return TechniqueMode::ForwardPlus;
            case RenderPathPreset::Deferred: return TechniqueMode::Deferred;
            case RenderPathPreset::TiledDeferred: return TechniqueMode::TiledDeferred;
            case RenderPathPreset::ClusteredForward: return TechniqueMode::ClusteredForward;
        }
        return TechniqueMode::ForwardPlus;
    }

    inline RenderPathPreset render_path_preset_for_mode(TechniqueMode mode)
    {
        switch (mode)
        {
            case TechniqueMode::Forward: return RenderPathPreset::Forward;
            case TechniqueMode::ForwardPlus: return RenderPathPreset::ForwardPlus;
            case TechniqueMode::Deferred: return RenderPathPreset::Deferred;
            case TechniqueMode::TiledDeferred: return RenderPathPreset::TiledDeferred;
            case TechniqueMode::ClusteredForward: return RenderPathPreset::ClusteredForward;
        }
        return RenderPathPreset::ForwardPlus;
    }

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

    inline LightCullingMode default_light_culling_mode_for_mode(TechniqueMode mode)
    {
        switch (mode)
        {
            case TechniqueMode::ForwardPlus:
                return LightCullingMode::Tiled;
            case TechniqueMode::TiledDeferred:
                return LightCullingMode::TiledDepthRange;
            case TechniqueMode::ClusteredForward:
                return LightCullingMode::Clustered;
            case TechniqueMode::Forward:
            case TechniqueMode::Deferred:
            default:
                return LightCullingMode::None;
        }
    }

    inline const std::array<RenderPathPreset, 5>& default_render_path_preset_order()
    {
        static const std::array<RenderPathPreset, 5> order = {
            RenderPathPreset::Forward,
            RenderPathPreset::ForwardPlus,
            RenderPathPreset::Deferred,
            RenderPathPreset::TiledDeferred,
            RenderPathPreset::ClusteredForward
        };
        return order;
    }

    inline RenderPathRecipe make_builtin_render_path_recipe(
        RenderPathPreset preset,
        RenderBackendType backend = RenderBackendType::Vulkan,
        std::string_view name_prefix = "path")
    {
        const TechniqueMode mode = render_path_preset_mode(preset);

        RenderPathRecipe recipe{};
        recipe.name = std::string(name_prefix) + "_" + render_path_preset_name(preset);
        recipe.backend = backend;
        recipe.light_volume_provider = RenderPathLightVolumeProvider::JoltShapeVolumes;
        recipe.view_culling = RenderPathCullingMode::FrustumAndOptionalOcclusion;
        recipe.shadow_culling = RenderPathCullingMode::FrustumAndOptionalOcclusion;
        recipe.render_technique = render_path_rendering_technique_for_mode(mode);
        recipe.technique_mode = mode;
        recipe.runtime_defaults.view_occlusion_enabled = true;
        recipe.runtime_defaults.shadow_occlusion_enabled = false;
        recipe.runtime_defaults.debug_aabb = false;
        recipe.runtime_defaults.lit_mode = true;
        recipe.runtime_defaults.enable_shadows = true;
        recipe.light_tile_size = 16u;
        recipe.cluster_z_slices = (mode == TechniqueMode::ClusteredForward) ? 24u : 16u;
        recipe.wants_shadows = true;
        recipe.strict_validation = true;

        const TechniqueProfile profile = make_default_technique_profile(mode);
        recipe.pass_chain.reserve(profile.passes.size());
        for (const auto& pass : profile.passes)
        {
            recipe.pass_chain.push_back(RenderPathPassEntry{pass.id, pass.pass_id, pass.required});
        }

        return recipe;
    }

    inline bool register_builtin_render_path_presets(
        RenderPathRegistry& registry,
        RenderBackendType backend,
        std::vector<std::string>* out_cycle_order = nullptr,
        std::string_view name_prefix = "path")
    {
        if (out_cycle_order) out_cycle_order->clear();

        bool ok = true;
        const auto& order = default_render_path_preset_order();
        for (const RenderPathPreset preset : order)
        {
            RenderPathRecipe recipe = make_builtin_render_path_recipe(preset, backend, name_prefix);
            const std::string id = recipe.name;
            const bool registered = registry.register_recipe(std::move(recipe));
            ok = ok && registered;
            if (out_cycle_order && registered)
            {
                out_cycle_order->push_back(id);
            }
        }
        return ok;
    }
}
