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
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace renderpath
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

    // --- Pass-id range law (Constitution I §7; arch §4 graduation req 1) -----
    // The builtin vocabulary is the small closed enum above; consumer/demo-owned
    // passes live in an open *registered* range, so a demo adds a pass without a
    // core edit. Ids in the open range are minted only by `PassIdRegistry`
    // (`shs/renderpath/execution/pass_id_registry.hpp`) in registration order —
    // never by hashing a name at the call site, because id assignment must not
    // depend on hash/iteration order (replay/determinism law).
    inline constexpr uint16_t kPassIdUnknown = 0u;
    inline constexpr uint16_t kPassIdBuiltinMin = 1u;
    inline constexpr uint16_t kPassIdBuiltinMax = 1023u;
    inline constexpr uint16_t kPassIdOpenBase = 1024u;
    inline constexpr uint16_t kPassIdOpenMax = 65534u;
    inline constexpr uint16_t kPassIdReserved = 65535u;

    // Vocabulary pins: the null id stays null and the builtin enum can never
    // silently grow into the open range (which would alias consumer passes).
    static_assert(static_cast<uint16_t>(PassId::Unknown) == kPassIdUnknown,
                  "PassId::Unknown must stay the null id");
    static_assert(static_cast<uint16_t>(PassId::DepthOfField) <= kPassIdBuiltinMax,
                  "builtin PassId vocabulary overflowed into the open registered range");

    inline constexpr bool pass_id_is_builtin(PassId id)
    {
        const uint16_t raw = static_cast<uint16_t>(id);
        return raw >= kPassIdBuiltinMin && raw <= kPassIdBuiltinMax;
    }

    inline constexpr bool pass_id_is_open(PassId id)
    {
        const uint16_t raw = static_cast<uint16_t>(id);
        return raw >= kPassIdOpenBase && raw <= kPassIdOpenMax;
    }

    // A typed id a plan or a registry may legitimately carry: builtin or minted.
    inline constexpr bool pass_id_in_valid_range(PassId id)
    {
        return pass_id_is_builtin(id) || pass_id_is_open(id);
    }

    // Honest, deterministic spelling for an open-range id that carries no
    // resolvable registry here. It is deliberately NOT a registration key: the
    // owning registry holds the real name — resolve it through
    // `PassIdRegistry::try_name`. Callers building factory keys or plan
    // canonical ids must go through the registry, never through this spelling.
    inline constexpr const char* kPassIdOpenSpelling = "open_pass";

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
                return "unknown";
            default:
                // New values can only be open-range (the builtin pins above hold).
                return pass_id_is_open(id) ? kPassIdOpenSpelling : "unknown";
        }
    }

    // Static name for a builtin id, nullptr when the name lives in a registry.
    inline const char* pass_id_name_or_null(PassId id)
    {
        if (pass_id_is_builtin(id)) return pass_id_name(id);
        return nullptr;
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

    // Legacy spelling for "the builtin vocabulary" — behavior-identical to
    // `pass_id_is_builtin` for every value the enum can hold. It reads as
    // "is a core pass", which is exactly the builtin test; prefer the explicit
    // `pass_id_is_builtin` in new code, and use `pass_id_in_valid_range` when a
    // consumer-minted open id must also pass.
    inline bool pass_id_is_standard(PassId id)
    {
        return pass_id_is_builtin(id);
    }

    inline std::string pass_id_string(PassId id)
    {
        return std::string(pass_id_name(id));
    }

    } // inline namespace renderpath
}
