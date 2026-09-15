#pragma once

/*
    SHS RENDERER SAN

    FILE: render_composition_presets.hpp
    MODULE: pipeline
    PURPOSE: Compose render-path presets with rendering-technique presets into reusable recipes.
*/


#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "shs/pipeline/render_path_presets.hpp"
#include "shs/pipeline/render_technique_presets.hpp"

namespace shs
{
    enum class RenderCompositionPostStackPreset : uint8_t
    {
        Default = 0,
        Minimal = 1,
        Temporal = 2,
        Full = 3
    };

    inline const char* render_composition_post_stack_preset_name(RenderCompositionPostStackPreset preset)
    {
        switch (preset)
        {
            case RenderCompositionPostStackPreset::Default: return "default";
            case RenderCompositionPostStackPreset::Minimal: return "minimal";
            case RenderCompositionPostStackPreset::Temporal: return "temporal";
            case RenderCompositionPostStackPreset::Full: return "full";
        }
        return "default";
    }

    struct RenderCompositionPostStackState
    {
        bool enable_ssao = false;
        bool enable_taa = false;
        bool enable_motion_blur = false;
        bool enable_depth_of_field = false;
    };

    inline bool render_path_preset_supports_ssao(RenderPathPreset path)
    {
        return path == RenderPathPreset::Deferred || path == RenderPathPreset::TiledDeferred;
    }

    inline bool render_path_preset_supports_taa(RenderPathPreset path)
    {
        return path == RenderPathPreset::Deferred || path == RenderPathPreset::TiledDeferred;
    }

    inline bool render_path_preset_supports_motion_blur(RenderPathPreset path)
    {
        (void)path;
        return true;
    }

    inline bool render_path_preset_supports_depth_of_field(RenderPathPreset path)
    {
        return path == RenderPathPreset::Deferred || path == RenderPathPreset::TiledDeferred;
    }

    inline RenderCompositionPostStackState default_render_composition_post_stack_state(RenderPathPreset path)
    {
        RenderCompositionPostStackState out{};
        out.enable_ssao = render_path_preset_supports_ssao(path);
        out.enable_taa = render_path_preset_supports_taa(path);
        out.enable_motion_blur = render_path_preset_supports_motion_blur(path);
        out.enable_depth_of_field = render_path_preset_supports_depth_of_field(path);
        return out;
    }

    inline RenderCompositionPostStackState resolve_render_composition_post_stack_state(
        RenderPathPreset path,
        RenderCompositionPostStackPreset preset)
    {
        const RenderCompositionPostStackState defaults = default_render_composition_post_stack_state(path);

        switch (preset)
        {
            case RenderCompositionPostStackPreset::Default:
                return defaults;
            case RenderCompositionPostStackPreset::Minimal:
                return RenderCompositionPostStackState{};
            case RenderCompositionPostStackPreset::Temporal:
            {
                RenderCompositionPostStackState out{};
                out.enable_ssao = defaults.enable_ssao;
                out.enable_taa = defaults.enable_taa;
                return out;
            }
            case RenderCompositionPostStackPreset::Full:
                return defaults;
        }
        return defaults;
    }

    inline bool render_composition_post_stack_controls_pass(PassId pass_id)
    {
        switch (pass_id)
        {
            case PassId::SSAO:
            case PassId::TAA:
            case PassId::MotionBlur:
            case PassId::DepthOfField:
                return true;
            default:
                break;
        }
        return false;
    }

    inline bool render_composition_post_stack_pass_enabled(
        PassId pass_id,
        const RenderCompositionPostStackState& state)
    {
        switch (pass_id)
        {
            case PassId::SSAO: return state.enable_ssao;
            case PassId::TAA: return state.enable_taa;
            case PassId::MotionBlur: return state.enable_motion_blur;
            case PassId::DepthOfField: return state.enable_depth_of_field;
            default:
                break;
        }
        return true;
    }

    struct RenderCompositionRecipe
    {
        std::string name{};
        RenderPathPreset path_preset = RenderPathPreset::ForwardPlus;
        RenderTechniquePreset technique_preset = RenderTechniquePreset::PBR;
        RenderCompositionPostStackPreset post_stack = RenderCompositionPostStackPreset::Default;
    };

    struct RenderCompositionResolved
    {
        RenderCompositionRecipe composition{};
        RenderPathRecipe path_recipe{};
        RenderTechniqueRecipe technique_recipe{};
    };

    inline std::string make_render_composition_name(
        RenderPathPreset path,
        RenderTechniquePreset technique,
        std::string_view name_prefix = "composition",
        RenderCompositionPostStackPreset post_stack = RenderCompositionPostStackPreset::Default)
    {
        std::string out = std::string(name_prefix) + "_"
            + render_path_preset_name(path) + "_"
            + render_technique_preset_name(technique);
        if (post_stack != RenderCompositionPostStackPreset::Default)
        {
            out += "_";
            out += render_composition_post_stack_preset_name(post_stack);
        }
        return out;
    }

    inline RenderCompositionRecipe make_builtin_render_composition_recipe(
        RenderPathPreset path,
        RenderTechniquePreset technique,
        std::string_view name_prefix = "composition",
        RenderCompositionPostStackPreset post_stack = RenderCompositionPostStackPreset::Default)
    {
        RenderCompositionRecipe recipe{};
        recipe.path_preset = path;
        recipe.technique_preset = technique;
        recipe.post_stack = post_stack;
        recipe.name = make_render_composition_name(path, technique, name_prefix, post_stack);
        return recipe;
    }

    inline RenderCompositionResolved resolve_builtin_render_composition_recipe(
        const RenderCompositionRecipe& composition,
        RenderBackendType backend = RenderBackendType::Vulkan,
        std::string_view path_name_prefix = "path",
        std::string_view technique_name_prefix = "technique")
    {
        RenderCompositionResolved resolved{};
        resolved.composition = composition;
        resolved.path_recipe = make_builtin_render_path_recipe(
            composition.path_preset,
            backend,
            path_name_prefix);
        const RenderCompositionPostStackState stack = resolve_render_composition_post_stack_state(
            composition.path_preset,
            composition.post_stack);
        std::vector<RenderPathPassEntry> filtered_pass_chain{};
        filtered_pass_chain.reserve(resolved.path_recipe.pass_chain.size());
        for (const auto& entry : resolved.path_recipe.pass_chain)
        {
            const PassId pass_id =
                pass_id_is_standard(entry.pass_id) ? entry.pass_id : parse_pass_id(entry.id);
            if (render_composition_post_stack_controls_pass(pass_id) &&
                !render_composition_post_stack_pass_enabled(pass_id, stack))
            {
                continue;
            }
            filtered_pass_chain.push_back(entry);
        }
        resolved.path_recipe.pass_chain = std::move(filtered_pass_chain);
        if (!composition.name.empty())
        {
            resolved.path_recipe.name = composition.name + "__path";
        }
        resolved.technique_recipe = make_builtin_render_technique_recipe(
            composition.technique_preset,
            technique_name_prefix);
        return resolved;
    }

    inline std::vector<RenderCompositionRecipe> make_default_render_composition_recipes(
        std::string_view name_prefix = "composition")
    {
        std::vector<RenderCompositionRecipe> out{};
        const auto& path_order = default_render_path_preset_order();
        const auto& technique_order = default_render_technique_preset_order();
        out.reserve(path_order.size() * technique_order.size());
        for (const RenderPathPreset path : path_order)
        {
            for (const RenderTechniquePreset technique : technique_order)
            {
                out.push_back(make_builtin_render_composition_recipe(
                    path,
                    technique,
                    name_prefix,
                    RenderCompositionPostStackPreset::Default));
            }
        }
        return out;
    }

    inline std::vector<RenderCompositionRecipe> make_phase_d_render_composition_recipes(
        std::string_view name_prefix = "composition")
    {
        std::vector<RenderCompositionRecipe> out = make_default_render_composition_recipes(name_prefix);

        // Curated post-stack variants for Phase-D coverage/testing.
        out.push_back(make_builtin_render_composition_recipe(
            RenderPathPreset::ForwardPlus,
            RenderTechniquePreset::PBR,
            name_prefix,
            RenderCompositionPostStackPreset::Minimal));
        out.push_back(make_builtin_render_composition_recipe(
            RenderPathPreset::Deferred,
            RenderTechniquePreset::PBR,
            name_prefix,
            RenderCompositionPostStackPreset::Temporal));
        out.push_back(make_builtin_render_composition_recipe(
            RenderPathPreset::Deferred,
            RenderTechniquePreset::PBR,
            name_prefix,
            RenderCompositionPostStackPreset::Full));
        out.push_back(make_builtin_render_composition_recipe(
            RenderPathPreset::Deferred,
            RenderTechniquePreset::BlinnPhong,
            name_prefix,
            RenderCompositionPostStackPreset::Full));
        out.push_back(make_builtin_render_composition_recipe(
            RenderPathPreset::TiledDeferred,
            RenderTechniquePreset::PBR,
            name_prefix,
            RenderCompositionPostStackPreset::Full));

        return out;
    }
}
