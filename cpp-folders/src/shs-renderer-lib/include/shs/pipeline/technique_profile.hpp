#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: technique_profile.hpp
    МОДУЛЬ: pipeline
    ЗОРИЛГО: Technique mode бүрийн default pass chain profile.
*/


#include <string>
#include <vector>

#include "shs/frame/technique_mode.hpp"

namespace shs
{
    struct TechniquePassEntry
    {
        std::string id{};
        bool required = true;
    };

    struct TechniqueProfile
    {
        TechniqueMode mode = TechniqueMode::Forward;
        std::vector<TechniquePassEntry> passes{};
    };

    inline TechniqueProfile make_default_technique_profile(TechniqueMode mode)
    {
        TechniqueProfile p{};
        p.mode = mode;

        // Эдгээр id нь PassFactoryRegistry-д бүртгэгдсэн үед автоматаар assemble хийнэ.
        switch (mode)
        {
            case TechniqueMode::Forward:
                p.passes = {
                    {"shadow_map", false},
                    {"pbr_forward", true},
                    {"tonemap", true},
                    {"motion_blur", false}
                };
                break;
            case TechniqueMode::ForwardPlus:
                p.passes = {
                    {"shadow_map", false},
                    {"depth_prepass", false},
                    {"light_culling", false},
                    {"pbr_forward_plus", true},
                    {"tonemap", true},
                    {"motion_blur", false}
                };
                break;
            case TechniqueMode::Deferred:
                p.passes = {
                    {"shadow_map", false},
                    {"gbuffer", false},
                    {"deferred_lighting", false},
                    {"tonemap", true},
                    {"motion_blur", false}
                };
                break;
            case TechniqueMode::TiledDeferred:
                p.passes = {
                    {"shadow_map", false},
                    {"depth_prepass", false},
                    {"gbuffer", false},
                    {"light_culling", false},
                    {"deferred_lighting_tiled", false},
                    {"tonemap", true},
                    {"motion_blur", false}
                };
                break;
            case TechniqueMode::ClusteredForward:
                p.passes = {
                    {"shadow_map", false},
                    {"depth_prepass", false},
                    {"cluster_build", false},
                    {"cluster_light_assign", false},
                    {"pbr_forward_clustered", false},
                    {"tonemap", true},
                    {"motion_blur", false}
                };
                break;
        }
        return p;
    }
}
