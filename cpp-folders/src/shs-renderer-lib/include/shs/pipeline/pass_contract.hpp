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
        LightGrid = 6,
        LightIndexList = 7,
        LightClusters = 8,
        Albedo = 9,
        Normal = 10,
        Material = 11,
        AmbientOcclusion = 12,
        HistoryColor = 13,
        HistoryDepth = 14,
        HistoryMotion = 15
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

    enum class PassSemanticSpace : uint8_t
    {
        Auto = 0,
        None = 1,
        World = 2,
        View = 3,
        Clip = 4,
        Screen = 5,
        Light = 6,
        Tile = 7
    };

    enum class PassSemanticEncoding : uint8_t
    {
        Auto = 0,
        Unknown = 1,
        Linear = 2,
        SRGB = 3,
        DeviceDepth = 4,
        LinearDepth = 5,
        UnitVector01 = 6,
        SignedVector = 7,
        VelocityScreen = 8,
        UIntIndices = 9,
        UIntCounts = 10
    };

    enum class PassSemanticLifetime : uint8_t
    {
        Auto = 0,
        Transient = 1,
        Persistent = 2,
        History = 3
    };

    enum class PassSemanticTemporalRole : uint8_t
    {
        None = 0,
        CurrentFrame = 1,
        HistoryRead = 2,
        HistoryWrite = 3
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
            case PassSemantic::LightGrid: return "light_grid";
            case PassSemantic::LightIndexList: return "light_index_list";
            case PassSemantic::LightClusters: return "light_clusters";
            case PassSemantic::Albedo: return "albedo";
            case PassSemantic::Normal: return "normal";
            case PassSemantic::Material: return "material";
            case PassSemantic::AmbientOcclusion: return "ambient_occlusion";
            case PassSemantic::HistoryColor: return "history_color";
            case PassSemantic::HistoryDepth: return "history_depth";
            case PassSemantic::HistoryMotion: return "history_motion";
        }
        return "unknown";
    }

    inline bool contract_access_has_read(ContractAccess a)
    {
        return a == ContractAccess::Read || a == ContractAccess::ReadWrite;
    }

    inline bool contract_access_has_write(ContractAccess a)
    {
        return a == ContractAccess::Write || a == ContractAccess::ReadWrite;
    }

    inline const char* pass_semantic_space_name(PassSemanticSpace s)
    {
        switch (s)
        {
            case PassSemanticSpace::Auto: return "auto";
            case PassSemanticSpace::None: return "none";
            case PassSemanticSpace::World: return "world";
            case PassSemanticSpace::View: return "view";
            case PassSemanticSpace::Clip: return "clip";
            case PassSemanticSpace::Screen: return "screen";
            case PassSemanticSpace::Light: return "light";
            case PassSemanticSpace::Tile: return "tile";
        }
        return "auto";
    }

    inline const char* pass_semantic_encoding_name(PassSemanticEncoding e)
    {
        switch (e)
        {
            case PassSemanticEncoding::Auto: return "auto";
            case PassSemanticEncoding::Unknown: return "unknown";
            case PassSemanticEncoding::Linear: return "linear";
            case PassSemanticEncoding::SRGB: return "srgb";
            case PassSemanticEncoding::DeviceDepth: return "device_depth";
            case PassSemanticEncoding::LinearDepth: return "linear_depth";
            case PassSemanticEncoding::UnitVector01: return "unit_vector_01";
            case PassSemanticEncoding::SignedVector: return "signed_vector";
            case PassSemanticEncoding::VelocityScreen: return "velocity_screen";
            case PassSemanticEncoding::UIntIndices: return "uint_indices";
            case PassSemanticEncoding::UIntCounts: return "uint_counts";
        }
        return "auto";
    }

    inline const char* pass_semantic_lifetime_name(PassSemanticLifetime l)
    {
        switch (l)
        {
            case PassSemanticLifetime::Auto: return "auto";
            case PassSemanticLifetime::Transient: return "transient";
            case PassSemanticLifetime::Persistent: return "persistent";
            case PassSemanticLifetime::History: return "history";
        }
        return "auto";
    }

    inline const char* pass_semantic_temporal_role_name(PassSemanticTemporalRole r)
    {
        switch (r)
        {
            case PassSemanticTemporalRole::None: return "none";
            case PassSemanticTemporalRole::CurrentFrame: return "current";
            case PassSemanticTemporalRole::HistoryRead: return "history_read";
            case PassSemanticTemporalRole::HistoryWrite: return "history_write";
        }
        return "none";
    }

    struct PassSemanticDescriptor
    {
        PassSemantic semantic = PassSemantic::Unknown;
        PassSemanticSpace space = PassSemanticSpace::Auto;
        PassSemanticEncoding encoding = PassSemanticEncoding::Auto;
        PassSemanticLifetime lifetime = PassSemanticLifetime::Auto;
        PassSemanticTemporalRole temporal_role = PassSemanticTemporalRole::CurrentFrame;
        bool sampled = true;
        bool storage = false;
    };

    inline PassSemanticDescriptor default_pass_semantic_descriptor(PassSemantic semantic)
    {
        PassSemanticDescriptor out{};
        out.semantic = semantic;
        out.space = PassSemanticSpace::Screen;
        out.encoding = PassSemanticEncoding::Linear;
        out.lifetime = PassSemanticLifetime::Transient;
        out.temporal_role = PassSemanticTemporalRole::CurrentFrame;
        out.sampled = true;
        out.storage = false;

        switch (semantic)
        {
            case PassSemantic::Unknown:
                out.space = PassSemanticSpace::None;
                out.encoding = PassSemanticEncoding::Unknown;
                out.lifetime = PassSemanticLifetime::Transient;
                out.sampled = false;
                out.storage = false;
                break;
            case PassSemantic::Depth:
                out.space = PassSemanticSpace::Screen;
                out.encoding = PassSemanticEncoding::DeviceDepth;
                out.lifetime = PassSemanticLifetime::Transient;
                out.sampled = true;
                break;
            case PassSemantic::ShadowMap:
                out.space = PassSemanticSpace::Light;
                out.encoding = PassSemanticEncoding::DeviceDepth;
                out.lifetime = PassSemanticLifetime::Persistent;
                out.sampled = true;
                break;
            case PassSemantic::ColorHDR:
                out.space = PassSemanticSpace::Screen;
                out.encoding = PassSemanticEncoding::Linear;
                out.lifetime = PassSemanticLifetime::Transient;
                out.sampled = true;
                break;
            case PassSemantic::ColorLDR:
                out.space = PassSemanticSpace::Screen;
                out.encoding = PassSemanticEncoding::SRGB;
                out.lifetime = PassSemanticLifetime::Persistent;
                out.sampled = true;
                break;
            case PassSemantic::MotionVectors:
                out.space = PassSemanticSpace::Screen;
                out.encoding = PassSemanticEncoding::VelocityScreen;
                out.lifetime = PassSemanticLifetime::Persistent;
                out.sampled = true;
                break;
            case PassSemantic::Albedo:
                out.space = PassSemanticSpace::Screen;
                out.encoding = PassSemanticEncoding::SRGB;
                out.lifetime = PassSemanticLifetime::Transient;
                out.sampled = true;
                break;
            case PassSemantic::Normal:
                out.space = PassSemanticSpace::View;
                out.encoding = PassSemanticEncoding::SignedVector;
                out.lifetime = PassSemanticLifetime::Transient;
                out.sampled = true;
                break;
            case PassSemantic::Material:
                out.space = PassSemanticSpace::Screen;
                out.encoding = PassSemanticEncoding::Linear;
                out.lifetime = PassSemanticLifetime::Transient;
                out.sampled = true;
                break;
            case PassSemantic::AmbientOcclusion:
                out.space = PassSemanticSpace::Screen;
                out.encoding = PassSemanticEncoding::Linear;
                out.lifetime = PassSemanticLifetime::Transient;
                out.sampled = true;
                break;
            case PassSemantic::LightGrid:
                out.space = PassSemanticSpace::Tile;
                out.encoding = PassSemanticEncoding::UIntCounts;
                out.lifetime = PassSemanticLifetime::Transient;
                out.sampled = false;
                out.storage = true;
                break;
            case PassSemantic::LightIndexList:
                out.space = PassSemanticSpace::Tile;
                out.encoding = PassSemanticEncoding::UIntIndices;
                out.lifetime = PassSemanticLifetime::Transient;
                out.sampled = false;
                out.storage = true;
                break;
            case PassSemantic::LightClusters:
                out.space = PassSemanticSpace::View;
                out.encoding = PassSemanticEncoding::UIntCounts;
                out.lifetime = PassSemanticLifetime::Transient;
                out.sampled = false;
                out.storage = true;
                break;
            case PassSemantic::HistoryColor:
                out.space = PassSemanticSpace::Screen;
                out.encoding = PassSemanticEncoding::Linear;
                out.lifetime = PassSemanticLifetime::History;
                out.temporal_role = PassSemanticTemporalRole::HistoryWrite;
                out.sampled = true;
                out.storage = false;
                break;
            case PassSemantic::HistoryDepth:
                out.space = PassSemanticSpace::Screen;
                out.encoding = PassSemanticEncoding::DeviceDepth;
                out.lifetime = PassSemanticLifetime::History;
                out.temporal_role = PassSemanticTemporalRole::HistoryWrite;
                out.sampled = true;
                out.storage = false;
                break;
            case PassSemantic::HistoryMotion:
                out.space = PassSemanticSpace::Screen;
                out.encoding = PassSemanticEncoding::VelocityScreen;
                out.lifetime = PassSemanticLifetime::History;
                out.temporal_role = PassSemanticTemporalRole::HistoryWrite;
                out.sampled = true;
                out.storage = false;
                break;
        }

        return out;
    }

    struct PassSemanticRef
    {
        PassSemantic semantic = PassSemantic::Unknown;
        ContractAccess access = ContractAccess::Read;
        ContractDomain domain = ContractDomain::Any;
        PassSemanticSpace space = PassSemanticSpace::Auto;
        PassSemanticEncoding encoding = PassSemanticEncoding::Auto;
        PassSemanticLifetime lifetime = PassSemanticLifetime::Auto;
        PassSemanticTemporalRole temporal_role = PassSemanticTemporalRole::CurrentFrame;
        bool sampled = true;
        bool storage = false;
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

    inline PassSemanticRef make_semantic_ref(
        PassSemantic s,
        ContractAccess access,
        ContractDomain d = ContractDomain::Any,
        const char* alias = nullptr,
        PassSemanticSpace space_override = PassSemanticSpace::Auto,
        PassSemanticEncoding encoding_override = PassSemanticEncoding::Auto,
        PassSemanticLifetime lifetime_override = PassSemanticLifetime::Auto,
        PassSemanticTemporalRole temporal_role_override = PassSemanticTemporalRole::None)
    {
        const PassSemanticDescriptor desc = default_pass_semantic_descriptor(s);
        PassSemanticRef out{};
        out.semantic = desc.semantic;
        out.access = access;
        out.domain = d;
        out.space = (space_override == PassSemanticSpace::Auto) ? desc.space : space_override;
        out.encoding = (encoding_override == PassSemanticEncoding::Auto) ? desc.encoding : encoding_override;
        out.lifetime = (lifetime_override == PassSemanticLifetime::Auto) ? desc.lifetime : lifetime_override;
        out.temporal_role =
            (temporal_role_override == PassSemanticTemporalRole::None)
                ? ((out.lifetime == PassSemanticLifetime::History)
                    ? (contract_access_has_read(access)
                        ? PassSemanticTemporalRole::HistoryRead
                        : PassSemanticTemporalRole::HistoryWrite)
                    : PassSemanticTemporalRole::CurrentFrame)
                : temporal_role_override;
        out.sampled = desc.sampled;
        out.storage = desc.storage;
        if (alias) out.alias = alias;
        return out;
    }

    inline PassSemanticRef read_semantic(PassSemantic s, ContractDomain d = ContractDomain::Any, const char* alias = nullptr)
    {
        return make_semantic_ref(s, ContractAccess::Read, d, alias);
    }

    inline PassSemanticRef write_semantic(PassSemantic s, ContractDomain d = ContractDomain::Any, const char* alias = nullptr)
    {
        return make_semantic_ref(s, ContractAccess::Write, d, alias);
    }

    inline PassSemanticRef read_write_semantic(PassSemantic s, ContractDomain d = ContractDomain::Any, const char* alias = nullptr)
    {
        return make_semantic_ref(s, ContractAccess::ReadWrite, d, alias);
    }
}
