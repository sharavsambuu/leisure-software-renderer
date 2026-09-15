#pragma once

/*
    SHS RENDERER SAN

    FILE: render_path_barrier_plan.hpp
    MODULE: pipeline
    PURPOSE: Build graph-owned barrier/access metadata and transient alias slots from compiled plans.
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
#include "shs/pipeline/render_path_compiler.hpp"
#include "shs/pipeline/render_path_resource_plan.hpp"

namespace shs
{
    struct RenderPathBarrierAccess
    {
        std::string pass_id{};
        PassId pass_kind = PassId::Unknown;
        uint32_t pass_index = 0u;
        std::string resource_id{};
        PassSemantic semantic = PassSemantic::Unknown;
        ContractAccess access = ContractAccess::Read;
        ContractDomain domain = ContractDomain::Any;
        bool sampled = true;
        bool storage = false;
    };

    struct RenderPathResourceLifetime
    {
        std::string resource_id{};
        PassSemantic semantic = PassSemantic::Unknown;
        uint32_t first_pass_index = 0u;
        uint32_t last_pass_index = 0u;
        bool transient = true;
        bool history = false;
        RenderPathResourceKind kind = RenderPathResourceKind::Texture2D;
        RenderPathResolutionClass resolution = RenderPathResolutionClass::Full;
        uint32_t alias_slot = 0u;
        std::string alias_class{};
    };

    struct RenderPathBarrierEdge
    {
        std::string resource_id{};
        PassSemantic semantic = PassSemantic::Unknown;
        std::string from_pass_id{};
        std::string to_pass_id{};
        PassId from_pass_kind = PassId::Unknown;
        PassId to_pass_kind = PassId::Unknown;
        uint32_t from_pass_index = 0u;
        uint32_t to_pass_index = 0u;
        ContractAccess from_access = ContractAccess::Read;
        ContractAccess to_access = ContractAccess::Read;
        ContractDomain from_domain = ContractDomain::Any;
        ContractDomain to_domain = ContractDomain::Any;
        bool requires_memory_barrier = false;
        bool requires_layout_transition = false;
    };

    struct RenderPathAliasClassSummary
    {
        std::string alias_class{};
        uint32_t resource_count = 0u;
        uint32_t slot_count = 0u;
    };

    struct RenderPathBarrierPlan
    {
        std::vector<RenderPathBarrierAccess> accesses{};
        std::vector<RenderPathResourceLifetime> lifetimes{};
        std::vector<RenderPathBarrierEdge> edges{};
        std::vector<RenderPathAliasClassSummary> alias_classes{};
        std::vector<std::string> warnings{};
        std::vector<std::string> errors{};
        bool valid = false;
    };

    inline bool render_path_barrier_requires_transition_between(
        const RenderPathBarrierAccess& from_access,
        const RenderPathBarrierAccess& to_access,
        const RenderPathResourceSpec& spec)
    {
        if (spec.kind != RenderPathResourceKind::Texture2D) return false;
        const bool from_writes = contract_access_has_write(from_access.access);
        const bool to_reads = contract_access_has_read(to_access.access);
        const bool to_writes = contract_access_has_write(to_access.access);
        return from_writes && (to_reads || to_writes);
    }

    inline std::string render_path_resource_alias_class_key(const RenderPathResourceSpec& spec)
    {
        std::string out{};
        out.reserve(96u);
        out += render_path_resource_kind_name(spec.kind);
        out += ".";
        out += render_path_resolution_class_name(spec.resolution);
        out += ".l";
        out += std::to_string(std::max(1u, spec.layers));
        out += ".";
        out += pass_semantic_space_name(spec.semantic_space);
        out += ".";
        out += pass_semantic_encoding_name(spec.semantic_encoding);
        out += spec.storage ? ".storage" : ".sampled";
        return out;
    }

    inline uint32_t render_path_barrier_layout_transition_count(const RenderPathBarrierPlan& plan)
    {
        uint32_t out = 0u;
        for (const auto& e : plan.edges)
        {
            if (e.requires_layout_transition) ++out;
        }
        return out;
    }

    inline uint32_t render_path_barrier_memory_edge_count(const RenderPathBarrierPlan& plan)
    {
        uint32_t out = 0u;
        for (const auto& e : plan.edges)
        {
            if (e.requires_memory_barrier) ++out;
        }
        return out;
    }

    inline uint32_t render_path_alias_slot_count(const RenderPathBarrierPlan& plan)
    {
        uint32_t out = 0u;
        for (const auto& c : plan.alias_classes)
        {
            out += c.slot_count;
        }
        return out;
    }

    inline const RenderPathBarrierEdge* find_render_path_barrier_edge(
        const RenderPathBarrierPlan& plan,
        PassSemantic semantic,
        PassId from_pass_kind = PassId::Unknown,
        PassId to_pass_kind = PassId::Unknown)
    {
        const RenderPathBarrierEdge* best = nullptr;
        for (const auto& edge : plan.edges)
        {
            if (edge.semantic != semantic) continue;
            if (from_pass_kind != PassId::Unknown && edge.from_pass_kind != from_pass_kind) continue;
            if (to_pass_kind != PassId::Unknown && edge.to_pass_kind != to_pass_kind) continue;
            if (!best || edge.to_pass_index < best->to_pass_index)
            {
                best = &edge;
            }
        }
        return best;
    }

    inline RenderPathBarrierPlan compile_render_path_barrier_plan(
        const RenderPathExecutionPlan& plan,
        const RenderPathResourcePlan& resource_plan,
        const PassFactoryRegistry* pass_registry = nullptr)
    {
        RenderPathBarrierPlan out{};
        out.valid = true;

        std::unordered_map<PassSemantic, const RenderPathResourceSpec*> spec_by_semantic{};
        spec_by_semantic.reserve(resource_plan.resources.size());
        for (const auto& spec : resource_plan.resources)
        {
            spec_by_semantic[spec.semantic] = &spec;
        }

        for (uint32_t pass_index = 0u; pass_index < plan.pass_chain.size(); ++pass_index)
        {
            const auto& pass_entry = plan.pass_chain[pass_index];
            const PassId pass_id =
                pass_id_is_standard(pass_entry.pass_id) ? pass_entry.pass_id : parse_pass_id(pass_entry.id);
            const std::string pass_name =
                pass_id_is_standard(pass_id) ? pass_id_string(pass_id) : pass_entry.id;

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
                    "' (descriptor hint required). Barrier planning is partial.");
                continue;
            }

            for (const PassSemanticRef& sref : contract.semantics)
            {
                if (sref.semantic == PassSemantic::Unknown) continue;
                const auto it = spec_by_semantic.find(sref.semantic);
                if (it == spec_by_semantic.end())
                {
                    out.warnings.push_back(
                        "Pass '" + pass_name + "' references semantic '" + pass_semantic_name(sref.semantic) +
                        "' but no resource spec exists in the active resource plan.");
                    continue;
                }

                RenderPathBarrierAccess access{};
                access.pass_id = pass_name;
                access.pass_kind = pass_id;
                access.pass_index = pass_index;
                access.resource_id = it->second->id;
                access.semantic = sref.semantic;
                access.access = sref.access;
                access.domain = sref.domain;
                access.sampled = sref.sampled;
                access.storage = sref.storage;
                out.accesses.push_back(std::move(access));
            }
        }

        std::unordered_map<std::string, std::vector<std::size_t>> access_indices_by_resource{};
        access_indices_by_resource.reserve(resource_plan.resources.size());
        for (std::size_t i = 0; i < out.accesses.size(); ++i)
        {
            access_indices_by_resource[out.accesses[i].resource_id].push_back(i);
        }

        out.lifetimes.reserve(resource_plan.resources.size());
        for (const auto& spec : resource_plan.resources)
        {
            auto it = access_indices_by_resource.find(spec.id);
            if (it == access_indices_by_resource.end() || it->second.empty()) continue;

            uint32_t first_pass = UINT32_MAX;
            uint32_t last_pass = 0u;
            for (const std::size_t access_index : it->second)
            {
                const uint32_t pass_index = out.accesses[access_index].pass_index;
                first_pass = std::min(first_pass, pass_index);
                last_pass = std::max(last_pass, pass_index);
            }

            RenderPathResourceLifetime lifetime{};
            lifetime.resource_id = spec.id;
            lifetime.semantic = spec.semantic;
            lifetime.first_pass_index = first_pass;
            lifetime.last_pass_index = last_pass;
            lifetime.transient = spec.transient;
            lifetime.history = spec.history;
            lifetime.kind = spec.kind;
            lifetime.resolution = spec.resolution;
            if (spec.transient && !spec.history)
            {
                lifetime.alias_class = render_path_resource_alias_class_key(spec);
            }

            out.lifetimes.push_back(std::move(lifetime));
        }

        out.edges.reserve(out.accesses.size());
        for (const auto& entry : access_indices_by_resource)
        {
            const auto spec_it = std::find_if(
                resource_plan.resources.begin(),
                resource_plan.resources.end(),
                [&](const RenderPathResourceSpec& spec) { return spec.id == entry.first; });
            if (spec_it == resource_plan.resources.end()) continue;

            std::vector<std::size_t> sorted_access_indices = entry.second;
            std::sort(
                sorted_access_indices.begin(),
                sorted_access_indices.end(),
                [&](std::size_t a, std::size_t b) {
                    return out.accesses[a].pass_index < out.accesses[b].pass_index;
                });

            for (std::size_t i = 1u; i < sorted_access_indices.size(); ++i)
            {
                const RenderPathBarrierAccess& prev = out.accesses[sorted_access_indices[i - 1u]];
                const RenderPathBarrierAccess& curr = out.accesses[sorted_access_indices[i]];
                if (prev.pass_index == curr.pass_index) continue;

                const bool prev_writes = contract_access_has_write(prev.access);
                const bool curr_writes = contract_access_has_write(curr.access);
                const bool requires_memory = prev_writes || curr_writes;
                const bool requires_layout =
                    render_path_barrier_requires_transition_between(prev, curr, *spec_it);
                if (!requires_memory && !requires_layout) continue;

                RenderPathBarrierEdge edge{};
                edge.resource_id = entry.first;
                edge.semantic = prev.semantic;
                edge.from_pass_id = prev.pass_id;
                edge.to_pass_id = curr.pass_id;
                edge.from_pass_kind = prev.pass_kind;
                edge.to_pass_kind = curr.pass_kind;
                edge.from_pass_index = prev.pass_index;
                edge.to_pass_index = curr.pass_index;
                edge.from_access = prev.access;
                edge.to_access = curr.access;
                edge.from_domain = prev.domain;
                edge.to_domain = curr.domain;
                edge.requires_memory_barrier = requires_memory;
                edge.requires_layout_transition = requires_layout;
                out.edges.push_back(std::move(edge));
            }
        }

        std::unordered_map<std::string, std::vector<std::size_t>> lifetimes_by_alias_class{};
        for (std::size_t i = 0; i < out.lifetimes.size(); ++i)
        {
            const auto& lifetime = out.lifetimes[i];
            if (lifetime.alias_class.empty()) continue;
            lifetimes_by_alias_class[lifetime.alias_class].push_back(i);
        }

        out.alias_classes.reserve(lifetimes_by_alias_class.size());
        for (auto& entry : lifetimes_by_alias_class)
        {
            std::vector<std::size_t>& indices = entry.second;
            std::sort(indices.begin(), indices.end(), [&](std::size_t a, std::size_t b) {
                const auto& la = out.lifetimes[a];
                const auto& lb = out.lifetimes[b];
                if (la.first_pass_index != lb.first_pass_index)
                {
                    return la.first_pass_index < lb.first_pass_index;
                }
                return la.last_pass_index < lb.last_pass_index;
            });

            std::vector<uint32_t> slot_last_pass{};
            for (const std::size_t lifetime_index : indices)
            {
                auto& lifetime = out.lifetimes[lifetime_index];
                uint32_t chosen_slot = UINT32_MAX;
                for (uint32_t slot = 0u; slot < slot_last_pass.size(); ++slot)
                {
                    if (lifetime.first_pass_index > slot_last_pass[slot])
                    {
                        chosen_slot = slot;
                        slot_last_pass[slot] = lifetime.last_pass_index;
                        break;
                    }
                }
                if (chosen_slot == UINT32_MAX)
                {
                    chosen_slot = static_cast<uint32_t>(slot_last_pass.size());
                    slot_last_pass.push_back(lifetime.last_pass_index);
                }
                lifetime.alias_slot = chosen_slot;
            }

            RenderPathAliasClassSummary summary{};
            summary.alias_class = entry.first;
            summary.resource_count = static_cast<uint32_t>(indices.size());
            summary.slot_count = static_cast<uint32_t>(slot_last_pass.size());
            out.alias_classes.push_back(std::move(summary));
        }

        std::sort(out.lifetimes.begin(), out.lifetimes.end(), [](const auto& a, const auto& b) {
            if (a.first_pass_index != b.first_pass_index) return a.first_pass_index < b.first_pass_index;
            return a.last_pass_index < b.last_pass_index;
        });
        std::sort(out.alias_classes.begin(), out.alias_classes.end(), [](const auto& a, const auto& b) {
            return a.alias_class < b.alias_class;
        });

        if (!out.errors.empty()) out.valid = false;
        return out;
    }
}
