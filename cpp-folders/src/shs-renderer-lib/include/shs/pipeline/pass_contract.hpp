#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: pass_contract.hpp
    МОДУЛЬ: pipeline
    ЗОРИЛГО: Render pass-ийн technique-level contract/semantic metadata.
*/


#include <cstdint>
#include <string>
#include <vector>

#include "shs/frame/technique_mode.hpp"

namespace shs
{
    enum class TechniquePassRole : uint8_t
    {
        Custom = 0,
        Visibility = 1,
        LightCulling = 2,
        GBuffer = 3,
        Lighting = 4,
        ForwardOpaque = 5,
        ForwardTransparent = 6,
        PostProcess = 7,
        Composite = 8,
        Present = 9
    };

    enum class PassSemantic : uint16_t
    {
        Unknown = 0,
        Depth = 1,
        ShadowMap = 2,
        ColorHDR = 3,
        ColorLDR = 4,
        MotionVectors = 5,
        GBufferA = 6,
        GBufferB = 7,
        GBufferC = 8,
        LightGrid = 9,
        LightIndexList = 10,
        LightClusters = 11
    };

    enum class ContractAccess : uint8_t
    {
        Read = 1,
        Write = 2,
        ReadWrite = 3
    };

    enum class ContractDomain : uint8_t
    {
        Any = 0,
        CPU = 1,
        GPU = 2,
        Software = 3,
        OpenGL = 4,
        Vulkan = 5
    };

    inline const char* pass_semantic_name(PassSemantic s)
    {
        switch (s)
        {
            case PassSemantic::Unknown: return "unknown";
            case PassSemantic::Depth: return "depth";
            case PassSemantic::ShadowMap: return "shadow_map";
            case PassSemantic::ColorHDR: return "color_hdr";
            case PassSemantic::ColorLDR: return "color_ldr";
            case PassSemantic::MotionVectors: return "motion_vectors";
            case PassSemantic::GBufferA: return "gbuffer_a";
            case PassSemantic::GBufferB: return "gbuffer_b";
            case PassSemantic::GBufferC: return "gbuffer_c";
            case PassSemantic::LightGrid: return "light_grid";
            case PassSemantic::LightIndexList: return "light_index_list";
            case PassSemantic::LightClusters: return "light_clusters";
        }
        return "unknown";
    }

    struct PassSemanticRef
    {
        PassSemantic semantic = PassSemantic::Unknown;
        ContractAccess access = ContractAccess::Read;
        ContractDomain domain = ContractDomain::Any;
        std::string alias{};
    };

    struct TechniquePassContract
    {
        TechniquePassRole role = TechniquePassRole::Custom;
        uint32_t supported_modes_mask = technique_mode_mask_all();
        std::vector<PassSemanticRef> semantics{};
        bool requires_depth_prepass = false;
        bool requires_light_culling = false;
        bool prefer_async_compute = false;
    };

    inline PassSemanticRef read_semantic(PassSemantic s, ContractDomain d = ContractDomain::Any, const char* alias = nullptr)
    {
        PassSemanticRef out{};
        out.semantic = s;
        out.access = ContractAccess::Read;
        out.domain = d;
        if (alias) out.alias = alias;
        return out;
    }

    inline PassSemanticRef write_semantic(PassSemantic s, ContractDomain d = ContractDomain::Any, const char* alias = nullptr)
    {
        PassSemanticRef out{};
        out.semantic = s;
        out.access = ContractAccess::Write;
        out.domain = d;
        if (alias) out.alias = alias;
        return out;
    }

    inline PassSemanticRef read_write_semantic(PassSemantic s, ContractDomain d = ContractDomain::Any, const char* alias = nullptr)
    {
        PassSemanticRef out{};
        out.semantic = s;
        out.access = ContractAccess::ReadWrite;
        out.domain = d;
        if (alias) out.alias = alias;
        return out;
    }
}
