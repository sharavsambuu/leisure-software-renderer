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
#include "shs/pipeline/pass_id.hpp"

namespace shs
{
    struct TechniquePassEntry
    {
        std::string id{};
        PassId pass_id = PassId::Unknown;
        bool required = true;
    };

    inline TechniquePassEntry make_technique_pass_entry(PassId pass_id, bool required)
    {
        TechniquePassEntry out{};
        out.id = pass_id_string(pass_id);
        out.pass_id = pass_id;
        out.required = required;
        return out;
    }

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
                    make_technique_pass_entry(PassId::ShadowMap, false),
                    make_technique_pass_entry(PassId::PBRForward, true),
                    make_technique_pass_entry(PassId::Tonemap, true),
                    make_technique_pass_entry(PassId::MotionBlur, false)
                };
                break;
            case TechniqueMode::ForwardPlus:
                p.passes = {
                    make_technique_pass_entry(PassId::ShadowMap, false),
                    make_technique_pass_entry(PassId::DepthPrepass, false),
                    make_technique_pass_entry(PassId::LightCulling, false),
                    make_technique_pass_entry(PassId::PBRForwardPlus, true),
                    make_technique_pass_entry(PassId::Tonemap, true),
                    make_technique_pass_entry(PassId::MotionBlur, false)
                };
                break;
            case TechniqueMode::Deferred:
                p.passes = {
                    make_technique_pass_entry(PassId::ShadowMap, false),
                    make_technique_pass_entry(PassId::GBuffer, false),
                    make_technique_pass_entry(PassId::SSAO, false),
                    make_technique_pass_entry(PassId::DeferredLighting, false),
                    make_technique_pass_entry(PassId::Tonemap, true),
                    make_technique_pass_entry(PassId::TAA, false),
                    make_technique_pass_entry(PassId::MotionBlur, false),
                    make_technique_pass_entry(PassId::DepthOfField, false)
                };
                break;
            case TechniqueMode::TiledDeferred:
                p.passes = {
                    make_technique_pass_entry(PassId::ShadowMap, false),
                    make_technique_pass_entry(PassId::DepthPrepass, false),
                    make_technique_pass_entry(PassId::GBuffer, false),
                    make_technique_pass_entry(PassId::SSAO, false),
                    make_technique_pass_entry(PassId::LightCulling, false),
                    make_technique_pass_entry(PassId::DeferredLightingTiled, false),
                    make_technique_pass_entry(PassId::Tonemap, true),
                    make_technique_pass_entry(PassId::TAA, false),
                    make_technique_pass_entry(PassId::MotionBlur, false),
                    make_technique_pass_entry(PassId::DepthOfField, false)
                };
                break;
            case TechniqueMode::ClusteredForward:
                p.passes = {
                    make_technique_pass_entry(PassId::ShadowMap, false),
                    make_technique_pass_entry(PassId::DepthPrepass, false),
                    make_technique_pass_entry(PassId::ClusterBuild, false),
                    make_technique_pass_entry(PassId::ClusterLightAssign, false),
                    make_technique_pass_entry(PassId::PBRForwardClustered, false),
                    make_technique_pass_entry(PassId::Tonemap, true),
                    make_technique_pass_entry(PassId::MotionBlur, false)
                };
                break;
        }
        return p;
    }
}
