#pragma once

/*
    SHS RENDERER SAN

    FILE: render_technique_presets.hpp
    MODULE: pipeline
    PURPOSE: Built-in rendering-technique presets (PBR/Blinn) and frame-param application helpers.
*/


#include <array>
#include <cstdint>
#include <string>
#include <string_view>

#include "shs/frame/frame_params.hpp"

namespace shs
{
    enum class RenderTechniquePreset : uint8_t
    {
        PBR = 0,
        BlinnPhong = 1
    };

    inline const char* render_technique_preset_name(RenderTechniquePreset preset)
    {
        switch (preset)
        {
            case RenderTechniquePreset::PBR: return "pbr";
            case RenderTechniquePreset::BlinnPhong: return "blinn";
        }
        return "pbr";
    }

    inline ShadingModel render_technique_shading_model(RenderTechniquePreset preset)
    {
        switch (preset)
        {
            case RenderTechniquePreset::BlinnPhong:
                return ShadingModel::BlinnPhong;
            case RenderTechniquePreset::PBR:
            default:
                return ShadingModel::PBRMetalRough;
        }
    }

    inline RenderTechniquePreset render_technique_preset_from_shading_model(ShadingModel model)
    {
        switch (model)
        {
            case ShadingModel::BlinnPhong:
                return RenderTechniquePreset::BlinnPhong;
            case ShadingModel::PBRMetalRough:
            default:
                return RenderTechniquePreset::PBR;
        }
    }

    inline uint32_t render_technique_shader_variant(RenderTechniquePreset preset)
    {
        // Shader-side contract:
        //   0 = PBR
        //   1 = Blinn-Phong
        return (preset == RenderTechniquePreset::BlinnPhong) ? 1u : 0u;
    }

    inline RenderTechniquePreset next_render_technique_preset(RenderTechniquePreset preset)
    {
        switch (preset)
        {
            case RenderTechniquePreset::PBR: return RenderTechniquePreset::BlinnPhong;
            case RenderTechniquePreset::BlinnPhong:
            default:
                return RenderTechniquePreset::PBR;
        }
    }

    inline const std::array<RenderTechniquePreset, 2>& default_render_technique_preset_order()
    {
        static const std::array<RenderTechniquePreset, 2> order = {
            RenderTechniquePreset::PBR,
            RenderTechniquePreset::BlinnPhong
        };
        return order;
    }

    struct RenderTechniqueRecipe
    {
        std::string name{};
        ShadingModel shading_model = ShadingModel::PBRMetalRough;
        bool enable_light_shafts = false;
        bool enable_motion_blur = false;
        float tonemap_exposure = 1.4f;
        float tonemap_gamma = 2.2f;
    };

    inline RenderTechniqueRecipe make_builtin_render_technique_recipe(
        RenderTechniquePreset preset,
        std::string_view name_prefix = "technique")
    {
        RenderTechniqueRecipe recipe{};
        recipe.name = std::string(name_prefix) + "_" + render_technique_preset_name(preset);
        recipe.shading_model = render_technique_shading_model(preset);
        recipe.enable_light_shafts = false;
        recipe.enable_motion_blur = false;

        if (preset == RenderTechniquePreset::BlinnPhong)
        {
            recipe.tonemap_exposure = 1.32f;
            recipe.tonemap_gamma = 2.2f;
        }
        else
        {
            recipe.tonemap_exposure = 1.40f;
            recipe.tonemap_gamma = 2.2f;
        }
        return recipe;
    }

    inline void apply_render_technique_recipe_to_frame_params(
        const RenderTechniqueRecipe& recipe,
        FrameParams& fp)
    {
        fp.shading_model = recipe.shading_model;
        fp.pass.tonemap.exposure = recipe.tonemap_exposure;
        fp.pass.tonemap.gamma = recipe.tonemap_gamma;
        fp.pass.light_shafts.enable = recipe.enable_light_shafts;
        fp.pass.motion_blur.enable = recipe.enable_motion_blur;
    }
}
