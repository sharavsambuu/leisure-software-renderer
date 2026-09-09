#pragma once

/*
    SHS RENDERER SAN

    FILE: render_path_resource_plan.hpp
    MODULE: pipeline
    PURPOSE: Compile render-path plans into resource/binding layout plans.
*/


#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "shs/pipeline/pass_contract.hpp"
#include "shs/pipeline/pass_contract_registry.hpp"
#include "shs/pipeline/pass_id.hpp"
#include "shs/pipeline/pass_registry.hpp"
#include "shs/pipeline/render_path_compiler.hpp"
#include "shs/pipeline/render_path_recipe.hpp"

namespace shs
{
    enum class RenderPathResourceKind : uint8_t
    {
        Texture2D = 0,
        StorageBuffer = 1
    };

    enum class RenderPathResolutionClass : uint8_t
    {
        Full = 0,
        Half = 1,
        Quarter = 2,
        Tile = 3,
        Absolute = 4
    };

    struct RenderPathResourceSpec
    {
        std::string id{};
        PassSemantic semantic = PassSemantic::Unknown;
        PassSemanticSpace semantic_space = PassSemanticSpace::Auto;
        PassSemanticEncoding semantic_encoding = PassSemanticEncoding::Auto;
        PassSemanticLifetime semantic_lifetime = PassSemanticLifetime::Auto;
        PassSemanticTemporalRole semantic_temporal_role = PassSemanticTemporalRole::CurrentFrame;
        RenderPathResourceKind kind = RenderPathResourceKind::Texture2D;
        RenderPathResolutionClass resolution = RenderPathResolutionClass::Full;
        uint32_t tile_size = 16u;
        uint32_t width = 0u;
        uint32_t height = 0u;
        uint32_t layers = 1u;
        bool transient = true;
        bool history = false;
        bool sampled = true;
        bool storage = false;
    };

    struct RenderPathPassResourceBinding
    {
        std::string pass_id{};
        std::vector<std::string> reads{};
        std::vector<std::string> writes{};
    };

    struct RenderPathResourcePlan
    {
        std::vector<RenderPathResourceSpec> resources{};
        std::vector<RenderPathPassResourceBinding> pass_bindings{};
        std::vector<std::string> warnings{};
        std::vector<std::string> errors{};
        bool valid = false;
    };

    struct RenderPathResourceExtent
    {
        uint32_t width = 0u;
        uint32_t height = 0u;
        uint32_t layers = 1u;
    };

    inline const char* render_path_resource_kind_name(RenderPathResourceKind k)
    {
        switch (k)
        {
            case RenderPathResourceKind::Texture2D: return "tex2d";
            case RenderPathResourceKind::StorageBuffer: return "ssbo";
        }
        return "tex2d";
    }

    inline const char* render_path_resolution_class_name(RenderPathResolutionClass c)
    {
        switch (c)
        {
            case RenderPathResolutionClass::Full: return "full";
            case RenderPathResolutionClass::Half: return "half";
            case RenderPathResolutionClass::Quarter: return "quarter";
            case RenderPathResolutionClass::Tile: return "tile";
            case RenderPathResolutionClass::Absolute: return "absolute";
        }
        return "full";
    }

    inline std::string render_path_resource_id_for_semantic(PassSemantic semantic)
    {
        switch (semantic)
        {
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
            case PassSemantic::AmbientOcclusion: return "ao";
            case PassSemantic::HistoryColor: return "history_color";
            case PassSemantic::HistoryDepth: return "history_depth";
            case PassSemantic::HistoryMotion: return "history_motion";
            case PassSemantic::Unknown:
            default:
                return "unknown";
        }
    }

    inline RenderPathResourceSpec make_default_resource_spec_for_semantic(
        PassSemantic semantic,
        const RenderPathRecipe& recipe)
    {
        const PassSemanticDescriptor desc = default_pass_semantic_descriptor(semantic);
        RenderPathResourceSpec spec{};
        spec.semantic = desc.semantic;
        spec.semantic_space = desc.space;
        spec.semantic_encoding = desc.encoding;
        spec.semantic_lifetime = desc.lifetime;
        spec.semantic_temporal_role = desc.temporal_role;
        spec.id = render_path_resource_id_for_semantic(desc.semantic);
        spec.layers = 1u;
        spec.sampled = desc.sampled;
        spec.storage = desc.storage;
        spec.transient = (desc.lifetime == PassSemanticLifetime::Transient);
        spec.history = (desc.lifetime == PassSemanticLifetime::History) ||
            (desc.temporal_role == PassSemanticTemporalRole::HistoryRead) ||
            (desc.temporal_role == PassSemanticTemporalRole::HistoryWrite);
        spec.tile_size = std::max(1u, recipe.light_tile_size);
        spec.kind = RenderPathResourceKind::Texture2D;
        spec.resolution = RenderPathResolutionClass::Full;

        switch (desc.semantic)
        {
            case PassSemantic::ShadowMap:
                spec.resolution = RenderPathResolutionClass::Absolute;
                spec.width = 2048u;
                spec.height = 2048u;
                spec.transient = false;
                break;
            case PassSemantic::LightGrid:
            case PassSemantic::LightIndexList:
                spec.kind = RenderPathResourceKind::StorageBuffer;
                spec.resolution = RenderPathResolutionClass::Tile;
                spec.storage = true;
                break;
            case PassSemantic::LightClusters:
                spec.kind = RenderPathResourceKind::StorageBuffer;
                spec.resolution = RenderPathResolutionClass::Tile;
                spec.storage = true;
                spec.layers = std::max(1u, recipe.cluster_z_slices);
                break;
            case PassSemantic::ColorLDR:
                spec.transient = false;
                break;
            case PassSemantic::MotionVectors:
                spec.transient = false;
                break;
            case PassSemantic::Depth:
            case PassSemantic::ColorHDR:
            case PassSemantic::Albedo:
            case PassSemantic::Normal:
            case PassSemantic::Material:
            case PassSemantic::AmbientOcclusion:
            case PassSemantic::HistoryColor:
            case PassSemantic::HistoryDepth:
            case PassSemantic::HistoryMotion:
            default:
                break;
        }

        return spec;
    }

    inline bool semantic_representation_mismatch(
        const RenderPathResourceSpec& spec,
        const PassSemanticRef& sref)
    {
        return spec.semantic_space != sref.space ||
               spec.semantic_encoding != sref.encoding;
    }

    inline RenderPathResourceExtent resolve_render_path_resource_extent(
        const RenderPathResourceSpec& spec,
        uint32_t frame_width,
        uint32_t frame_height)
    {
        RenderPathResourceExtent out{};
        out.layers = std::max(1u, spec.layers);

        switch (spec.resolution)
        {
            case RenderPathResolutionClass::Full:
                out.width = frame_width;
                out.height = frame_height;
                break;
            case RenderPathResolutionClass::Half:
                out.width = std::max(1u, (frame_width + 1u) / 2u);
                out.height = std::max(1u, (frame_height + 1u) / 2u);
                break;
            case RenderPathResolutionClass::Quarter:
                out.width = std::max(1u, (frame_width + 3u) / 4u);
                out.height = std::max(1u, (frame_height + 3u) / 4u);
                break;
            case RenderPathResolutionClass::Tile:
            {
                const uint32_t tile = std::max(1u, spec.tile_size);
                out.width = std::max(1u, (frame_width + tile - 1u) / tile);
                out.height = std::max(1u, (frame_height + tile - 1u) / tile);
                break;
            }
            case RenderPathResolutionClass::Absolute:
            default:
                out.width = std::max(1u, spec.width);
                out.height = std::max(1u, spec.height);
                break;
        }

        return out;
    }

    inline const RenderPathResourceSpec* find_render_path_resource_by_semantic(
        const RenderPathResourcePlan& plan,
        PassSemantic semantic)
    {
        for (const auto& spec : plan.resources)
        {
            if (spec.semantic == semantic) return &spec;
        }
        return nullptr;
    }

    inline bool pass_semantic_supports_visual_debug(PassSemantic semantic)
    {
        switch (semantic)
        {
            case PassSemantic::Depth:
            case PassSemantic::ShadowMap:
            case PassSemantic::ColorHDR:
            case PassSemantic::ColorLDR:
            case PassSemantic::MotionVectors:
            case PassSemantic::Albedo:
            case PassSemantic::Normal:
            case PassSemantic::Material:
            case PassSemantic::AmbientOcclusion:
                return true;
            case PassSemantic::Unknown:
            case PassSemantic::LightGrid:
            case PassSemantic::LightIndexList:
            case PassSemantic::LightClusters:
            case PassSemantic::HistoryColor:
            case PassSemantic::HistoryDepth:
            case PassSemantic::HistoryMotion:
            default:
                return false;
        }
    }

    inline uint32_t pass_semantic_visual_debug_priority(PassSemantic semantic)
    {
        switch (semantic)
        {
            case PassSemantic::ColorLDR: return 0u;
            case PassSemantic::ColorHDR: return 1u;
            case PassSemantic::Albedo: return 2u;
            case PassSemantic::Normal: return 3u;
            case PassSemantic::Material: return 4u;
            case PassSemantic::Depth: return 5u;
            case PassSemantic::ShadowMap: return 6u;
            case PassSemantic::MotionVectors: return 7u;
            case PassSemantic::AmbientOcclusion: return 8u;
            default: return 255u;
        }
    }

    inline std::vector<PassSemantic> collect_render_path_visual_debug_semantics(const RenderPathResourcePlan& plan)
    {
        std::vector<PassSemantic> out{};
        out.reserve(plan.resources.size());
        for (const auto& spec : plan.resources)
        {
            if (!pass_semantic_supports_visual_debug(spec.semantic)) continue;
            if (spec.kind != RenderPathResourceKind::Texture2D) continue;
            if (std::find(out.begin(), out.end(), spec.semantic) == out.end())
            {
                out.push_back(spec.semantic);
            }
        }
        std::sort(out.begin(), out.end(), [](PassSemantic a, PassSemantic b) {
            return pass_semantic_visual_debug_priority(a) < pass_semantic_visual_debug_priority(b);
        });
        return out;
    }

    inline RenderPathResourcePlan compile_render_path_resource_plan(
        const RenderPathExecutionPlan& plan,
        const RenderPathRecipe& recipe,
        const PassFactoryRegistry* pass_registry = nullptr)
    {
        RenderPathResourcePlan out{};
        out.valid = true;

        std::unordered_map<PassSemantic, std::size_t> resource_index_by_semantic{};
        std::unordered_map<PassSemantic, bool> produced_semantics{};
        resource_index_by_semantic.reserve(32);
        produced_semantics.reserve(32);

        for (const auto& pass_entry : plan.pass_chain)
        {
            const PassId pass_id =
                pass_id_is_standard(pass_entry.pass_id) ? pass_entry.pass_id : parse_pass_id(pass_entry.id);
            const std::string pass_name =
                pass_id_is_standard(pass_id) ? pass_id_string(pass_id) : pass_entry.id;

            RenderPathPassResourceBinding binding{};
            binding.pass_id = pass_name;

            TechniquePassContract contract{};
            bool have_contract = false;
            if (pass_id_is_standard(pass_id))
            {
                have_contract = lookup_standard_pass_contract(pass_id, contract);
            }
            if (!have_contract && pass_registry)
            {
                have_contract =
                    pass_id_is_standard(pass_id)
                        ? pass_registry->try_get_contract_hint(pass_id, contract)
                        : pass_registry->try_get_contract_hint(pass_entry.id, contract);
            }
            if (!have_contract)
            {
                out.warnings.push_back(
                    "No semantic contract available for pass '" + pass_name +
                    "' (descriptor hint required). Resource planning is partial.");
                out.pass_bindings.push_back(std::move(binding));
                continue;
            }

            if (contract.requires_depth_prepass &&
                produced_semantics.find(PassSemantic::Depth) == produced_semantics.end())
            {
                out.errors.push_back(
                    "Pass '" + pass_name + "' requires depth, but no prior pass writes 'depth'.");
                out.valid = false;
            }

            if (contract.requires_light_culling)
            {
                const bool has_grid = produced_semantics.find(PassSemantic::LightGrid) != produced_semantics.end();
                const bool has_list = produced_semantics.find(PassSemantic::LightIndexList) != produced_semantics.end();
                if (!has_grid || !has_list)
                {
                    out.errors.push_back(
                        "Pass '" + pass_name + "' requires light culling outputs, but they are not produced yet.");
                    out.valid = false;
                }
            }

            for (const PassSemanticRef& sref : contract.semantics)
            {
                if (sref.semantic == PassSemantic::Unknown) continue;

                const bool reads = contract_access_has_read(sref.access);
                const bool writes = contract_access_has_write(sref.access);
                const bool history_read = sref.temporal_role == PassSemanticTemporalRole::HistoryRead;
                const bool history_write = sref.temporal_role == PassSemanticTemporalRole::HistoryWrite;

                auto it = resource_index_by_semantic.find(sref.semantic);
                if (it == resource_index_by_semantic.end())
                {
                    RenderPathResourceSpec spec = make_default_resource_spec_for_semantic(sref.semantic, recipe);
                    out.resources.push_back(std::move(spec));
                    const std::size_t idx = out.resources.size() - 1u;
                    resource_index_by_semantic[sref.semantic] = idx;
                    it = resource_index_by_semantic.find(sref.semantic);
                }

                const std::string resource_id = out.resources[it->second].id;
                RenderPathResourceSpec& spec = out.resources[it->second];
                if (semantic_representation_mismatch(spec, sref))
                {
                    out.errors.push_back(
                        "Pass '" + pass_name + "' uses semantic '" + pass_semantic_name(sref.semantic) +
                        "' with mismatched representation (space=" + pass_semantic_space_name(sref.space) +
                        ", encoding=" + pass_semantic_encoding_name(sref.encoding) +
                        "), but resource spec expects (space=" + pass_semantic_space_name(spec.semantic_space) +
                        ", encoding=" + pass_semantic_encoding_name(spec.semantic_encoding) + ").");
                    out.valid = false;
                }

                const PassSemanticLifetime expected_lifetime =
                    (history_read || history_write) ? PassSemanticLifetime::History : spec.semantic_lifetime;
                if (expected_lifetime != sref.lifetime)
                {
                    out.errors.push_back(
                        "Pass '" + pass_name + "' uses semantic '" + pass_semantic_name(sref.semantic) +
                        "' with mismatched lifetime (" + pass_semantic_lifetime_name(sref.lifetime) +
                        "), but resource spec expects (" + pass_semantic_lifetime_name(expected_lifetime) + ").");
                    out.valid = false;
                }

                spec.sampled = spec.sampled || sref.sampled;
                spec.storage = spec.storage || sref.storage;
                if (history_read || history_write)
                {
                    spec.history = true;
                    spec.transient = false;
                    spec.semantic_lifetime = PassSemanticLifetime::History;
                }

                if (reads)
                {
                    if (produced_semantics.find(sref.semantic) == produced_semantics.end() &&
                        sref.semantic != PassSemantic::ShadowMap &&
                        !history_read)
                    {
                        out.errors.push_back(
                            "Pass '" + pass_name + "' reads '" + pass_semantic_name(sref.semantic) +
                            "' before it is produced.");
                        out.valid = false;
                    }
                    if (history_read && !spec.history)
                    {
                        out.errors.push_back(
                            "Pass '" + pass_name + "' marks '" + pass_semantic_name(sref.semantic) +
                            "' as history-read, but resource is not marked as history.");
                        out.valid = false;
                    }
                    binding.reads.push_back(resource_id);
                }
                if (writes)
                {
                    produced_semantics[sref.semantic] = true;
                    if (history_write && !spec.history)
                    {
                        out.errors.push_back(
                            "Pass '" + pass_name + "' marks '" + pass_semantic_name(sref.semantic) +
                            "' as history-write, but resource is not marked as history.");
                        out.valid = false;
                    }
                    binding.writes.push_back(resource_id);
                }
            }

            out.pass_bindings.push_back(std::move(binding));
        }

        return out;
    }
}
