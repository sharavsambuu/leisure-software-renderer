#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: render_path_recipe.hpp
    МОДУЛЬ: pipeline
    ЗОРИЛГО: Динамик рендер замыг (render path) бүрдүүлэх өгөгдөл төвтэй (data-first) жор ба загвар.
*/


#include <cstdint>
#include <string>
#include <vector>

#include "shs/render/frame/technique_mode.hpp"
#include "shs/render/frame/backend_type.hpp"
#include "shs/renderpath/planning/pass_id.hpp"
#include "shs/renderpath/planning/substrate_resolution.hpp"
#include "shs/renderpath/execution/render_path_runtime_state.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace renderpath
    {
    enum class RenderPathLightVolumeProvider : uint8_t
    {
        Default = 0,
        JoltShapeVolumes = 1,
        ClusteredGrid = 2
    };

    inline const char* render_path_light_volume_provider_name(RenderPathLightVolumeProvider p)
    {
        switch (p)
        {
            case RenderPathLightVolumeProvider::Default: return "default";
            case RenderPathLightVolumeProvider::JoltShapeVolumes: return "jolt_shape_volumes";
            case RenderPathLightVolumeProvider::ClusteredGrid: return "clustered_grid";
        }
        return "unknown";
    }

    enum class RenderPathCullingMode : uint8_t
    {
        None = 0,
        Frustum = 1,
        FrustumAndOcclusion = 2,
        FrustumAndOptionalOcclusion = 3
    };

    inline const char* render_path_culling_mode_name(RenderPathCullingMode mode)
    {
        switch (mode)
        {
            case RenderPathCullingMode::None: return "none";
            case RenderPathCullingMode::Frustum: return "frustum";
            case RenderPathCullingMode::FrustumAndOcclusion: return "frustum+occlusion";
            case RenderPathCullingMode::FrustumAndOptionalOcclusion: return "frustum+optional_occlusion";
        }
        return "unknown";
    }

    inline bool render_path_culling_requires_occlusion(RenderPathCullingMode mode)
    {
        return mode == RenderPathCullingMode::FrustumAndOcclusion;
    }

    inline bool render_path_culling_allows_occlusion(RenderPathCullingMode mode)
    {
        return mode == RenderPathCullingMode::FrustumAndOcclusion ||
               mode == RenderPathCullingMode::FrustumAndOptionalOcclusion;
    }

    enum class RenderPathRenderingTechnique : uint8_t
    {
        ForwardLit = 0,
        ForwardPlus = 1,
        Deferred = 2
    };

    inline const char* render_path_rendering_technique_name(RenderPathRenderingTechnique t)
    {
        switch (t)
        {
            case RenderPathRenderingTechnique::ForwardLit: return "forward_lit";
            case RenderPathRenderingTechnique::ForwardPlus: return "forward_plus";
            case RenderPathRenderingTechnique::Deferred: return "deferred";
        }
        return "unknown";
    }

    struct RenderPathPassEntry
    {
        std::string id{};
        PassId pass_id = PassId::Unknown;
        bool required = true;

        // RP-1: the entry's *intent* — what this pass needs, stated by the
        // author. It narrows which substrates the compiler may resolve this pass
        // to; it no longer names one. `Unspecified` (the default) narrows
        // nothing, so every pre-RP-1 recipe keeps its written chain verbatim.
        RenderDomain domain = render_domain_unspecified();

        // Value semantics (pod test kit requires snapshot equality).
        bool operator==(const RenderPathPassEntry&) const = default;
    };

    inline RenderPathPassEntry make_render_path_pass_entry(PassId pass_id, bool required)
    {
        RenderPathPassEntry out{};
        out.id = pass_id_string(pass_id);
        out.pass_id = pass_id;
        out.required = required;
        return out;
    }

    // RP-1: the same entry, with an explicit intent. Non-mutating, so a recipe
    // can be derived from another without aliasing it (plans and recipes are
    // values).
    inline RenderPathPassEntry with_domain(RenderPathPassEntry entry, RenderDomain domain)
    {
        entry.domain = domain;
        return entry;
    }

    // Consumer-owned (open-id) pass entry: the name is the registration key, so
    // it must be the exact name interned through `PassIdRegistry`
    // (`PassFactoryRegistry::intern_pass_id`). Builtin passes may use this
    // overload too; the name must then be the builtin spelling.
    inline RenderPathPassEntry make_render_path_pass_entry(std::string id, PassId pass_id, bool required)
    {
        RenderPathPassEntry out{};
        out.id = std::move(id);
        out.pass_id = pass_id;
        out.required = required;
        return out;
    }

    struct RenderPathRecipe
    {
        std::string name{};

        // RP-1: the *declared* substrate — the `ExactMatch` target, and the
        // fallback when the host's available-substrate set is unknown. It is no
        // longer the plan's per-pass substrate: that is a resolution output
        // (`RenderPathCompiledPass::substrate`), so ONE recipe can resolve to
        // software, device, or a hybrid. Kept on the recipe because a consumer
        // still needs a substrate to route to a context backend with.
        RenderBackendType backend = RenderBackendType::Software;

        // How the compiler should choose among the substrates that could
        // realize each pass. `ExactMatch` is the default and the historical
        // behaviour: a recipe that states no policy resolves where it always
        // did.
        SubstratePolicy substrate_policy = SubstratePolicy::ExactMatch;
        RenderPathLightVolumeProvider light_volume_provider = RenderPathLightVolumeProvider::Default;
        RenderPathCullingMode view_culling = RenderPathCullingMode::Frustum;
        RenderPathCullingMode shadow_culling = RenderPathCullingMode::FrustumAndOptionalOcclusion;
        RenderPathRenderingTechnique render_technique = RenderPathRenderingTechnique::ForwardLit;
        TechniqueMode technique_mode = TechniqueMode::Forward;

        std::vector<RenderPathPassEntry> pass_chain{};
        RenderPathRuntimeState runtime_defaults{};

        // Resource-layout knobs that should come from recipe/preset instead of demo constants.
        uint32_t light_tile_size = 16u;
        uint32_t cluster_z_slices = 16u;

        bool wants_shadows = true;
        bool strict_validation = true;

        // Value semantics (pod test kit requires snapshot equality).
        bool operator==(const RenderPathRecipe&) const = default;
    };

    // Technique-mode transition table (C2.2, P1 pure-leaf placement): the
    // closed-enum image of a technique mode in rendering-technique space.
    // The compiler asserts a compiled plan's technique is exactly this table
    // image of its mode (Rule 17, value invariant); presets and pod arrows
    // derive their pairs through this table, so the assert never fires on
    // house input. Moved here (was render_path_presets.hpp) so the pure leaf
    // compiler header can state the invariant without an include cycle.
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

    // RP-1 (graduation req 4): ONE recipe, resolved two ways, and now the only
    // soft-shadow-culling entry point. The
    // `make_default_soft_shadow_culling_recipe(backend)` fork that used to sit
    // here is REMOVED (2026-09-18). It authored two *different* recipes —
    // different pass chains AND different technique modes — for one intent,
    // which is precisely the authoring-time substrate choice req 4 removes; it
    // was the last place in the tree where the substrate selected the *shape* of
    // the path rather than the runtime of a pass. All four of its references
    // migrated: the two `render_path_registry.hpp` callers now register this one
    // recipe, `renderpath.contract.hpp` re-exports this name, and
    // `hello_soft_shadow_culling_vk.cpp` declares its substrate and lets the
    // policy resolve it.
    //
    // No pass carries a pinned `RenderDomain`: intent that is not a real
    // requirement would pre-empt the policy. What each pass *can* realize comes
    // from the registry, and the policy picks. `DepthPrepass` / `LightCulling` /
    // `MotionBlur` stay `required == false`, so a host that cannot realize them
    // resolves the rest instead of failing the whole plan.
    //
    // `policy` is the *resolution* input. `recipe.backend` is only the DECLARED
    // substrate — the `ExactMatch` target, and the fallback when the host
    // advertises no substrate set — so a caller that must not substitute passes
    // `ExactMatch` and its declared backend; a caller on a Vulkan host that
    // leaves `available_substrate_mask` empty gets Vulkan because the declared
    // substrate is the admissible set.
    inline RenderPathRecipe make_soft_shadow_culling_recipe(
        SubstratePolicy policy = SubstratePolicy::DevicePreferred)
    {
        RenderPathRecipe recipe{};
        recipe.name = "soft_shadow_culling";
        recipe.substrate_policy = policy;
        recipe.render_technique = RenderPathRenderingTechnique::ForwardPlus;
        recipe.technique_mode = TechniqueMode::ForwardPlus;
        recipe.light_volume_provider = RenderPathLightVolumeProvider::JoltShapeVolumes;
        recipe.view_culling = RenderPathCullingMode::FrustumAndOcclusion;
        recipe.shadow_culling = RenderPathCullingMode::FrustumAndOptionalOcclusion;
        recipe.runtime_defaults.view_occlusion_enabled = true;
        recipe.runtime_defaults.shadow_occlusion_enabled = false;
        recipe.runtime_defaults.debug_aabb = false;
        recipe.runtime_defaults.lit_mode = true;
        recipe.runtime_defaults.enable_shadows = true;
        recipe.wants_shadows = true;
        recipe.strict_validation = true;
        recipe.pass_chain = {
            make_render_path_pass_entry(PassId::ShadowMap, true),
            make_render_path_pass_entry(PassId::DepthPrepass, false),
            make_render_path_pass_entry(PassId::LightCulling, false),
            make_render_path_pass_entry(PassId::PBRForwardPlus, true),
            make_render_path_pass_entry(PassId::Tonemap, true),
            make_render_path_pass_entry(PassId::MotionBlur, false)
        };
        return recipe;
    }

    } // inline namespace renderpath
}
