#pragma once

/*
    SHS RENDERER SAN

    FILE: pass_contract_registry.hpp
    MODULE: pipeline
    PURPOSE: Shared pass-contract lookup + lightweight contract-only pass registry.
*/


#include <string>
#include <string_view>
#include <utility>

#include "shs/pipeline/pass_contract.hpp"
#include "shs/pipeline/pass_id.hpp"
#include "shs/pipeline/pass_registry.hpp"

namespace shs
{
    inline bool lookup_standard_pass_contract(PassId pass_id, TechniquePassContract& out)
    {
        out = TechniquePassContract{};
        out.supported_modes_mask = technique_mode_mask_all();

        if (pass_id == PassId::ShadowMap)
        {
            out.role = TechniquePassRole::Visibility;
            out.semantics = {
                write_semantic(PassSemantic::ShadowMap, ContractDomain::GPU, "shadow")
            };
            return true;
        }
        if (pass_id == PassId::DepthPrepass)
        {
            out.role = TechniquePassRole::Visibility;
            out.semantics = {
                write_semantic(PassSemantic::Depth, ContractDomain::GPU, "depth")
            };
            return true;
        }
        if (pass_id == PassId::LightCulling)
        {
            out.role = TechniquePassRole::LightCulling;
            out.requires_depth_prepass = true;
            out.semantics = {
                read_semantic(PassSemantic::Depth, ContractDomain::GPU, "depth"),
                write_semantic(PassSemantic::LightGrid, ContractDomain::GPU, "light_grid"),
                write_semantic(PassSemantic::LightIndexList, ContractDomain::GPU, "light_index_list")
            };
            return true;
        }
        if (pass_id == PassId::ClusterBuild)
        {
            out.role = TechniquePassRole::LightCulling;
            out.requires_depth_prepass = true;
            out.semantics = {
                read_semantic(PassSemantic::Depth, ContractDomain::GPU, "depth"),
                write_semantic(PassSemantic::LightClusters, ContractDomain::GPU, "clusters")
            };
            return true;
        }
        if (pass_id == PassId::ClusterLightAssign)
        {
            out.role = TechniquePassRole::LightCulling;
            out.requires_depth_prepass = true;
            out.semantics = {
                read_semantic(PassSemantic::Depth, ContractDomain::GPU, "depth"),
                read_semantic(PassSemantic::LightClusters, ContractDomain::GPU, "clusters"),
                write_semantic(PassSemantic::LightGrid, ContractDomain::GPU, "light_grid"),
                write_semantic(PassSemantic::LightIndexList, ContractDomain::GPU, "light_index_list")
            };
            return true;
        }
        if (pass_id == PassId::GBuffer)
        {
            out.role = TechniquePassRole::GBuffer;
            out.semantics = {
                write_semantic(PassSemantic::Depth, ContractDomain::GPU, "depth"),
                write_semantic(PassSemantic::Albedo, ContractDomain::GPU, "albedo"),
                write_semantic(PassSemantic::Normal, ContractDomain::GPU, "normal"),
                write_semantic(PassSemantic::Material, ContractDomain::GPU, "material")
            };
            return true;
        }
        if (pass_id == PassId::SSAO)
        {
            out.role = TechniquePassRole::PostProcess;
            out.semantics = {
                read_semantic(PassSemantic::Depth, ContractDomain::GPU, "depth"),
                read_semantic(PassSemantic::Normal, ContractDomain::GPU, "normal"),
                write_semantic(PassSemantic::AmbientOcclusion, ContractDomain::GPU, "ao")
            };
            return true;
        }
        if (pass_id == PassId::DeferredLighting)
        {
            out.role = TechniquePassRole::Lighting;
            out.semantics = {
                read_semantic(PassSemantic::ShadowMap, ContractDomain::GPU, "shadow"),
                read_semantic(PassSemantic::Albedo, ContractDomain::GPU, "albedo"),
                read_semantic(PassSemantic::Normal, ContractDomain::GPU, "normal"),
                read_semantic(PassSemantic::Material, ContractDomain::GPU, "material"),
                read_semantic(PassSemantic::AmbientOcclusion, ContractDomain::GPU, "ao"),
                write_semantic(PassSemantic::ColorHDR, ContractDomain::GPU, "hdr"),
                write_semantic(PassSemantic::MotionVectors, ContractDomain::GPU, "motion")
            };
            return true;
        }
        if (pass_id == PassId::DeferredLightingTiled)
        {
            out.role = TechniquePassRole::Lighting;
            out.requires_light_culling = true;
            out.semantics = {
                read_semantic(PassSemantic::ShadowMap, ContractDomain::GPU, "shadow"),
                read_semantic(PassSemantic::Albedo, ContractDomain::GPU, "albedo"),
                read_semantic(PassSemantic::Normal, ContractDomain::GPU, "normal"),
                read_semantic(PassSemantic::Material, ContractDomain::GPU, "material"),
                read_semantic(PassSemantic::AmbientOcclusion, ContractDomain::GPU, "ao"),
                read_semantic(PassSemantic::Depth, ContractDomain::GPU, "depth"),
                read_semantic(PassSemantic::LightGrid, ContractDomain::GPU, "light_grid"),
                read_semantic(PassSemantic::LightIndexList, ContractDomain::GPU, "light_index_list"),
                write_semantic(PassSemantic::ColorHDR, ContractDomain::GPU, "hdr"),
                write_semantic(PassSemantic::MotionVectors, ContractDomain::GPU, "motion")
            };
            return true;
        }
        if (pass_id == PassId::PBRForward)
        {
            out.role = TechniquePassRole::ForwardOpaque;
            out.semantics = {
                read_semantic(PassSemantic::ShadowMap, ContractDomain::GPU, "shadow"),
                write_semantic(PassSemantic::ColorHDR, ContractDomain::GPU, "hdr"),
                write_semantic(PassSemantic::MotionVectors, ContractDomain::GPU, "motion")
            };
            return true;
        }
        if (pass_id == PassId::PBRForwardPlus)
        {
            out.role = TechniquePassRole::ForwardOpaque;
            out.requires_light_culling = true;
            out.semantics = {
                read_semantic(PassSemantic::ShadowMap, ContractDomain::GPU, "shadow"),
                read_semantic(PassSemantic::Depth, ContractDomain::GPU, "depth"),
                read_semantic(PassSemantic::LightGrid, ContractDomain::GPU, "light_grid"),
                read_semantic(PassSemantic::LightIndexList, ContractDomain::GPU, "light_index_list"),
                write_semantic(PassSemantic::ColorHDR, ContractDomain::GPU, "hdr"),
                write_semantic(PassSemantic::MotionVectors, ContractDomain::GPU, "motion")
            };
            return true;
        }
        if (pass_id == PassId::PBRForwardClustered)
        {
            out.role = TechniquePassRole::ForwardOpaque;
            out.requires_light_culling = true;
            out.semantics = {
                read_semantic(PassSemantic::ShadowMap, ContractDomain::GPU, "shadow"),
                read_semantic(PassSemantic::Depth, ContractDomain::GPU, "depth"),
                read_semantic(PassSemantic::LightGrid, ContractDomain::GPU, "light_grid"),
                read_semantic(PassSemantic::LightIndexList, ContractDomain::GPU, "light_index_list"),
                write_semantic(PassSemantic::ColorHDR, ContractDomain::GPU, "hdr"),
                write_semantic(PassSemantic::MotionVectors, ContractDomain::GPU, "motion")
            };
            return true;
        }
        if (pass_id == PassId::Tonemap)
        {
            out.role = TechniquePassRole::PostProcess;
            out.semantics = {
                read_semantic(PassSemantic::ColorHDR, ContractDomain::GPU, "hdr"),
                write_semantic(PassSemantic::ColorLDR, ContractDomain::GPU, "ldr")
            };
            return true;
        }
        if (pass_id == PassId::MotionBlur)
        {
            out.role = TechniquePassRole::PostProcess;
            out.semantics = {
                read_write_semantic(PassSemantic::ColorLDR, ContractDomain::GPU, "ldr"),
                read_semantic(PassSemantic::MotionVectors, ContractDomain::GPU, "motion")
            };
            return true;
        }
        if (pass_id == PassId::DepthOfField)
        {
            out.role = TechniquePassRole::PostProcess;
            out.semantics = {
                read_write_semantic(PassSemantic::ColorLDR, ContractDomain::GPU, "ldr"),
                read_semantic(PassSemantic::Depth, ContractDomain::GPU, "depth")
            };
            return true;
        }
        if (pass_id == PassId::TAA)
        {
            out.role = TechniquePassRole::PostProcess;
            out.semantics = {
                read_write_semantic(PassSemantic::ColorLDR, ContractDomain::GPU, "ldr"),
                read_semantic(PassSemantic::HistoryColor, ContractDomain::GPU, "history_in"),
                write_semantic(PassSemantic::HistoryColor, ContractDomain::GPU, "history_out")
            };
            return true;
        }
        return false;
    }

    inline bool lookup_standard_pass_contract(std::string_view pass_id, TechniquePassContract& out)
    {
        return lookup_standard_pass_contract(parse_pass_id(pass_id), out);
    }

    class ContractOnlyRenderPass final : public IRenderPass
    {
    public:
        ContractOnlyRenderPass(PassId pass_id, TechniquePassContract contract)
            : pass_id_(pass_id),
              contract_(std::move(contract))
        {
        }

        ContractOnlyRenderPass(
            PassId pass_id,
            TechniquePassContract contract,
            RenderBackendType constrained_backend)
            : pass_id_(pass_id),
              contract_(std::move(contract)),
              backend_constrained_(true),
              constrained_backend_(constrained_backend)
        {
        }

        const char* id() const override
        {
            return pass_id_name(pass_id_);
        }

        bool supports_backend(RenderBackendType backend) const override
        {
            if (!backend_constrained_) return true;
            return backend == constrained_backend_;
        }

        TechniquePassContract describe_contract() const override
        {
            return contract_;
        }

        void execute(Context& ctx, const Scene& scene, const FrameParams& fp, RTRegistry& rtr) override
        {
            (void)ctx;
            (void)scene;
            (void)fp;
            (void)rtr;
        }

    private:
        PassId pass_id_ = PassId::Unknown;
        TechniquePassContract contract_{};
        bool backend_constrained_ = false;
        RenderBackendType constrained_backend_ = RenderBackendType::Software;
    };

    inline PassFactoryRegistry make_standard_pass_contract_registry()
    {
        PassFactoryRegistry registry{};
        const PassId known_pass_ids[] = {
            PassId::ShadowMap,
            PassId::DepthPrepass,
            PassId::LightCulling,
            PassId::ClusterBuild,
            PassId::ClusterLightAssign,
            PassId::GBuffer,
            PassId::SSAO,
            PassId::DeferredLighting,
            PassId::DeferredLightingTiled,
            PassId::PBRForward,
            PassId::PBRForwardPlus,
            PassId::PBRForwardClustered,
            PassId::Tonemap,
            PassId::MotionBlur,
            PassId::DepthOfField,
            PassId::TAA
        };

        for (const PassId pass_id : known_pass_ids)
        {
            TechniquePassContract contract{};
            if (!lookup_standard_pass_contract(pass_id, contract)) continue;
            registry.register_factory(pass_id, [pass_id, contract]() {
                return std::make_unique<ContractOnlyRenderPass>(pass_id, contract);
            });
        }
        return registry;
    }

    inline PassFactoryRegistry make_standard_pass_contract_registry_for_backend(RenderBackendType backend)
    {
        PassFactoryRegistry registry{};
        const PassId known_pass_ids[] = {
            PassId::ShadowMap,
            PassId::DepthPrepass,
            PassId::LightCulling,
            PassId::ClusterBuild,
            PassId::ClusterLightAssign,
            PassId::GBuffer,
            PassId::SSAO,
            PassId::DeferredLighting,
            PassId::DeferredLightingTiled,
            PassId::PBRForward,
            PassId::PBRForwardPlus,
            PassId::PBRForwardClustered,
            PassId::Tonemap,
            PassId::MotionBlur,
            PassId::DepthOfField,
            PassId::TAA
        };

        for (const PassId pass_id : known_pass_ids)
        {
            TechniquePassContract contract{};
            if (!lookup_standard_pass_contract(pass_id, contract)) continue;
            registry.register_factory(pass_id, [pass_id, contract, backend]() {
                return std::make_unique<ContractOnlyRenderPass>(pass_id, contract, backend);
            });
        }
        return registry;
    }
}
