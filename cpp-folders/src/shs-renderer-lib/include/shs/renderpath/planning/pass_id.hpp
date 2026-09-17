#pragma once

/*
    SHS RENDERER SAN

    FILE: pass_id.hpp
    MODULE: pipeline
    PURPOSE: Canonical typed identifiers for standard render-path passes.
*/


#include <cstdint>
#include <string>
#include <string_view>

namespace shs
{
    enum class PassId : uint16_t
    {
        Unknown = 0,
        ShadowMap = 1,
        DepthPrepass = 2,
        LightCulling = 3,
        ClusterBuild = 4,
        ClusterLightAssign = 5,
        GBuffer = 6,
        DeferredLighting = 7,
        DeferredLightingTiled = 8,
        PBRForward = 9,
        PBRForwardPlus = 10,
        PBRForwardClustered = 11,
        Tonemap = 12,
        MotionBlur = 13,
        TAA = 14,
        SSAO = 15,
        DepthOfField = 16
    };

    inline const char* pass_id_name(PassId id)
    {
        switch (id)
        {
            case PassId::ShadowMap: return "shadow_map";
            case PassId::DepthPrepass: return "depth_prepass";
            case PassId::LightCulling: return "light_culling";
            case PassId::ClusterBuild: return "cluster_build";
            case PassId::ClusterLightAssign: return "cluster_light_assign";
            case PassId::GBuffer: return "gbuffer";
            case PassId::DeferredLighting: return "deferred_lighting";
            case PassId::DeferredLightingTiled: return "deferred_lighting_tiled";
            case PassId::PBRForward: return "pbr_forward";
            case PassId::PBRForwardPlus: return "pbr_forward_plus";
            case PassId::PBRForwardClustered: return "pbr_forward_clustered";
            case PassId::Tonemap: return "tonemap";
            case PassId::MotionBlur: return "motion_blur";
            case PassId::TAA: return "taa";
            case PassId::SSAO: return "ssao";
            case PassId::DepthOfField: return "depth_of_field";
            case PassId::Unknown:
            default:
                return "unknown";
        }
    }

    inline PassId parse_pass_id(std::string_view id)
    {
        if (id == "shadow_map") return PassId::ShadowMap;
        if (id == "depth_prepass") return PassId::DepthPrepass;
        if (id == "light_culling") return PassId::LightCulling;
        if (id == "cluster_build") return PassId::ClusterBuild;
        if (id == "cluster_light_assign") return PassId::ClusterLightAssign;
        if (id == "gbuffer") return PassId::GBuffer;
        if (id == "deferred_lighting") return PassId::DeferredLighting;
        if (id == "deferred_lighting_tiled") return PassId::DeferredLightingTiled;
        if (id == "pbr_forward") return PassId::PBRForward;
        if (id == "pbr_forward_plus") return PassId::PBRForwardPlus;
        if (id == "pbr_forward_clustered") return PassId::PBRForwardClustered;
        if (id == "tonemap") return PassId::Tonemap;
        if (id == "motion_blur") return PassId::MotionBlur;
        if (id == "taa") return PassId::TAA;
        if (id == "ssao") return PassId::SSAO;
        if (id == "depth_of_field") return PassId::DepthOfField;
        return PassId::Unknown;
    }

    inline bool pass_id_is_standard(PassId id)
    {
        return id != PassId::Unknown;
    }

    inline std::string pass_id_string(PassId id)
    {
        return std::string(pass_id_name(id));
    }
}
