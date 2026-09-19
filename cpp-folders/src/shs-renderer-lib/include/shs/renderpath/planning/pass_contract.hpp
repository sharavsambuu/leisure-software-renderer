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

#include "shs/render/frame/technique_mode.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace renderpath
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

    // --- Pass-semantic range law (Constitution I §7; arch §4 graduation req 7) -
    // Same shape and the same reasoning as the pass-id range law in
    // planning/pass_id.hpp, and deliberately so: an author who understands one
    // namespace's open range understands both.
    //
    // The builtin vocabulary is the closed enum above; consumer-owned semantics
    // (an engine's own G-buffer channels, a technique's private intermediate)
    // live in an open *registered* range, so a consumer names its own semantic
    // without a core edit. Ids in the open range are minted only by
    // `PassSemanticRegistry`
    // (`shs/renderpath/planning/semantic_registry.hpp`) from a registered NAME
    // via the shared content-addressing law (core/open_id_hash.hpp) -- never by
    // hashing a name at the call site, because id assignment must not depend on
    // hash/iteration order (replay/determinism law).
    //
    // The bounds are intentionally identical to the pass-id bounds; the two
    // namespaces are typed separately (a PassId can never be passed where a
    // PassSemantic is expected), so the shared numerics are not aliasing.
    inline constexpr uint16_t kPassSemanticUnknown = 0u;
    inline constexpr uint16_t kPassSemanticBuiltinMin = 1u;
    inline constexpr uint16_t kPassSemanticBuiltinMax = 1023u;
    inline constexpr uint16_t kPassSemanticOpenBase = 1024u;
    inline constexpr uint16_t kPassSemanticOpenMax = 65534u;
    inline constexpr uint16_t kPassSemanticReserved = 65535u;

    // Vocabulary pins: the null semantic stays null and the builtin enum can
    // never silently grow into the open range (which would alias consumer
    // semantics -- the collision would be silent and load-bearing, since a
    // resource spec keys off the semantic).
    static_assert(static_cast<uint16_t>(PassSemantic::Unknown) == kPassSemanticUnknown,
                  "PassSemantic::Unknown must stay the null semantic");
    static_assert(static_cast<uint16_t>(PassSemantic::HistoryMotion) <= kPassSemanticBuiltinMax,
                  "builtin PassSemantic vocabulary overflowed into the open registered range");

    inline constexpr bool pass_semantic_is_builtin(PassSemantic s)
    {
        const uint16_t raw = static_cast<uint16_t>(s);
        return raw >= kPassSemanticBuiltinMin && raw <= kPassSemanticBuiltinMax;
    }

    inline constexpr bool pass_semantic_is_open(PassSemantic s)
    {
        const uint16_t raw = static_cast<uint16_t>(s);
        return raw >= kPassSemanticOpenBase && raw <= kPassSemanticOpenMax;
    }

    // A typed semantic a plan or a registry may legitimately carry: builtin or
    // minted.
    inline constexpr bool pass_semantic_in_valid_range(PassSemantic s)
    {
        return pass_semantic_is_builtin(s) || pass_semantic_is_open(s);
    }

    // Honest, deterministic spelling for an open-range semantic that carries no
    // resolvable registry here. Deliberately NOT a registration key: the owning
    // registry holds the real name -- resolve it through
    // `PassSemanticRegistry::try_name`. Callers building resource ids or contract
    // canonical keys must go through the registry, never through this spelling.
    inline constexpr const char* kPassSemanticOpenSpelling = "open_semantic";

    enum class ContractAccess : uint8_t
    {
        Read = 1,
        Write = 2,
        ReadWrite = 3
    };

    // ---------------------------------------------------------------------
    // Render-path target vocabulary (RP-2; owner ruling 2026-09-18).
    //
    // Two INDEPENDENT axes, deliberately not collapsed:
    //   ExecutionUnit - where a pass executes (host CPU vs device).
    //   Substrate     - which renderer realizes it.
    //
    // The retired ContractDomain / PassResourceDomain pair collapsed these
    // ("host == software rasterizer", "device == {GL,Vulkan}") into one
    // six-value enum duplicated across two headers. That made host-assisted
    // *device* work inexpressible and kept three dead values (`CPU`,
    // `OpenGL`, `Vulkan` - zero uses each) in circulation.
    //
    // Hybrid legality: mixing execution units in one chain is legal only where
    // the crossing passes declare an interop boundary (`is_interop_pass()`)
    // AND share a declared staging resource. The relation below is what makes
    // an undeclared crossing a *rejection* rather than a warning.
    // ---------------------------------------------------------------------
    enum class ExecutionUnit : uint8_t
    {
        Host = 0,
        Device = 1
    };

    enum class Substrate : uint8_t
    {
        SoftwareRaster = 0,
        OpenGL = 1,
        Vulkan = 2
    };

    // The retired sentinel did double duty: "the author did not state a target"
    // (default argument) and "matches any target" (compatibility check). Those
    // two meanings are now named separately - neither inherits the other's.
    enum class RenderDomainKind : uint8_t
    {
        Unspecified = 0,
        Any = 1,
        Pinned = 2
    };

    struct RenderDomain
    {
        RenderDomainKind kind = RenderDomainKind::Unspecified;
        ExecutionUnit unit = ExecutionUnit::Host;
        Substrate substrate = Substrate::SoftwareRaster;
        bool substrate_pinned = false;

        // Value semantics (pod test kit requires snapshot equality): a declared
        // intent is data, and RP-1 made it a plan input, so it must compare by
        // value like every other plan input.
        bool operator==(const RenderDomain&) const = default;
    };

    inline constexpr ExecutionUnit execution_unit_of(Substrate s)
    {
        return (s == Substrate::SoftwareRaster) ? ExecutionUnit::Host : ExecutionUnit::Device;
    }

    inline constexpr RenderDomain render_domain_unspecified()
    {
        return RenderDomain{};
    }

    inline constexpr RenderDomain render_domain_any()
    {
        return RenderDomain{RenderDomainKind::Any, ExecutionUnit::Host, Substrate::SoftwareRaster, false};
    }

    inline constexpr RenderDomain render_domain_host()
    {
        return RenderDomain{RenderDomainKind::Pinned, ExecutionUnit::Host, Substrate::SoftwareRaster, true};
    }

    // Device execution with no substrate pin - "any device substrate". This is
    // exactly what the retired `GPU` value meant.
    inline constexpr RenderDomain render_domain_device()
    {
        return RenderDomain{RenderDomainKind::Pinned, ExecutionUnit::Device, Substrate::OpenGL, false};
    }

    inline constexpr RenderDomain render_domain_substrate(Substrate s)
    {
        return RenderDomain{RenderDomainKind::Pinned, execution_unit_of(s), s, true};
    }

    inline const char* execution_unit_name(ExecutionUnit u)
    {
        switch (u)
        {
            case ExecutionUnit::Host: return "host";
            case ExecutionUnit::Device: return "device";
        }
        return "unknown";
    }

    inline const char* substrate_name(Substrate s)
    {
        switch (s)
        {
            case Substrate::SoftwareRaster: return "software_raster";
            case Substrate::OpenGL: return "opengl";
            case Substrate::Vulkan: return "vulkan";
        }
        return "unknown";
    }

    inline const char* render_domain_kind_name(RenderDomainKind k)
    {
        switch (k)
        {
            case RenderDomainKind::Unspecified: return "unspecified";
            case RenderDomainKind::Any: return "any";
            case RenderDomainKind::Pinned: return "pinned";
        }
        return "unknown";
    }

    inline std::string render_domain_name(const RenderDomain& d)
    {
        if (d.kind == RenderDomainKind::Unspecified) return "unspecified";
        if (d.kind == RenderDomainKind::Any) return "any";
        std::string out = execution_unit_name(d.unit);
        if (d.substrate_pinned)
        {
            out += ':';
            out += substrate_name(d.substrate);
        }
        return out;
    }

    // Resource-level compatibility across two declared domains.
    //
    // Note what falls out of the two axes rather than being special-cased: an
    // unpinned device resource is compatible with a pinned GL/Vulkan one, while
    // host vs device is FALSE - i.e. an undeclared cross-unit crossing is a
    // rejection, which is the hybrid legality rule.
    inline bool render_domains_compatible(const RenderDomain& a, const RenderDomain& b)
    {
        if (a.kind == RenderDomainKind::Unspecified || b.kind == RenderDomainKind::Unspecified) return true;
        if (a.kind == RenderDomainKind::Any || b.kind == RenderDomainKind::Any) return true;
        if (a.unit != b.unit) return false;
        if (!a.substrate_pinned || !b.substrate_pinned) return true;
        return a.substrate == b.substrate;
    }

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
            default:
                // New values can only be open-range (the builtin pins above hold).
                return pass_semantic_is_open(s) ? kPassSemanticOpenSpelling : "unknown";
        }
    }

    // Static name for a builtin semantic, nullptr when the name lives in a
    // registry. The honest counterpart of `pass_id_name_or_null`: a caller that
    // must have a *real* name for an open semantic has to resolve it through the
    // owning PassSemanticRegistry rather than accept the open spelling.
    inline const char* pass_semantic_name_or_null(PassSemantic s)
    {
        if (pass_semantic_is_builtin(s)) return pass_semantic_name(s);
        return nullptr;
    }

    // Parse a builtin semantic name. Open names are NOT parseable here by
    // construction -- their ids are minted by a registry from the same name, and
    // parsing one back would be a second, drifting definition of that mapping.
    inline PassSemantic parse_pass_semantic(std::string_view s)
    {
        if (s == "depth") return PassSemantic::Depth;
        if (s == "shadow_map") return PassSemantic::ShadowMap;
        if (s == "color_hdr") return PassSemantic::ColorHDR;
        if (s == "color_ldr") return PassSemantic::ColorLDR;
        if (s == "motion_vectors") return PassSemantic::MotionVectors;
        if (s == "light_grid") return PassSemantic::LightGrid;
        if (s == "light_index_list") return PassSemantic::LightIndexList;
        if (s == "light_clusters") return PassSemantic::LightClusters;
        if (s == "albedo") return PassSemantic::Albedo;
        if (s == "normal") return PassSemantic::Normal;
        if (s == "material") return PassSemantic::Material;
        if (s == "ambient_occlusion") return PassSemantic::AmbientOcclusion;
        if (s == "history_color") return PassSemantic::HistoryColor;
        if (s == "history_depth") return PassSemantic::HistoryDepth;
        if (s == "history_motion") return PassSemantic::HistoryMotion;
        return PassSemantic::Unknown;
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

        // Open-registered semantics take these neutral defaults, deliberately and
        // by early return. The core cannot know a consumer's channel intent, and
        // it must NOT guess: an author that needs a specific space/encoding/
        // lifetime states it through `make_semantic_ref`'s overrides, which are
        // applied ON TOP of this descriptor. Returning here keeps the builtin
        // switch exhaustive over the builtin vocabulary instead of relying on
        // fall-through, and leaves Unknown and every builtin path untouched.
        if (pass_semantic_is_open(semantic)) return out;

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
        RenderDomain domain = render_domain_unspecified();
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
        RenderDomain d = render_domain_unspecified(),
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

    inline PassSemanticRef read_semantic(PassSemantic s, RenderDomain d = render_domain_unspecified(), const char* alias = nullptr)
    {
        return make_semantic_ref(s, ContractAccess::Read, d, alias);
    }

    inline PassSemanticRef write_semantic(PassSemantic s, RenderDomain d = render_domain_unspecified(), const char* alias = nullptr)
    {
        return make_semantic_ref(s, ContractAccess::Write, d, alias);
    }

    inline PassSemanticRef read_write_semantic(PassSemantic s, RenderDomain d = render_domain_unspecified(), const char* alias = nullptr)
    {
        return make_semantic_ref(s, ContractAccess::ReadWrite, d, alias);
    }

    } // inline namespace renderpath
}
