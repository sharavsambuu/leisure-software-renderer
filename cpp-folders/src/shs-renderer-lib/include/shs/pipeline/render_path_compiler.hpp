#pragma once

/*
    SHS RENDERER SAN

    FILE: render_path_compiler.hpp
    MODULE: pipeline
    PURPOSE: Recipe validation + compilation into an executable pass chain plan.
*/


#include <string>
#include <optional>
#include <unordered_set>
#include <vector>

#include "shs/core/context.hpp"
#include "shs/pipeline/pass_registry.hpp"
#include "shs/pipeline/render_path_capabilities.hpp"
#include "shs/pipeline/render_path_recipe.hpp"
#include "shs/pipeline/technique_profile.hpp"

namespace shs
{
    struct RenderPathCompatibilityRules
    {
        bool require_shadow_map_pass_when_shadows_enabled = true;
        bool require_depth_prepass_for_occlusion = true;
        bool require_occlusion_support_for_occlusion_culling = true;
        bool require_depth_attachment_for_shadow_pass = true;
        bool reject_empty_pass_chain = true;
        bool reject_unknown_required_passes = true;
        bool reject_duplicate_pass_ids = false;
    };

    struct RenderPathCompiledPass
    {
        std::string id{};
        PassId pass_id = PassId::Unknown;
        bool required = true;
    };

    struct RenderPathExecutionPlan
    {
        std::string recipe_name{};
        RenderBackendType backend = RenderBackendType::Software;
        TechniqueMode technique_mode = TechniqueMode::Forward;
        RenderPathRenderingTechnique render_technique = RenderPathRenderingTechnique::ForwardLit;
        RenderPathRuntimeState runtime_state{};
        std::vector<RenderPathCompiledPass> pass_chain{};
        std::vector<std::string> warnings{};
        std::vector<std::string> errors{};
        bool valid = false;
    };

    inline TechniqueProfile make_technique_profile(const RenderPathExecutionPlan& plan)
    {
        TechniqueProfile profile{};
        profile.mode = plan.technique_mode;
        for (const auto& pass : plan.pass_chain)
        {
            profile.passes.push_back(TechniquePassEntry{pass.id, pass.pass_id, pass.required});
        }
        return profile;
    }

    class RenderPathCompiler
    {
    public:
        explicit RenderPathCompiler(RenderPathCompatibilityRules rules = RenderPathCompatibilityRules{})
            : rules_(rules)
        {
        }

        const RenderPathCompatibilityRules& rules() const
        {
            return rules_;
        }

        void set_rules(const RenderPathCompatibilityRules& rules)
        {
            rules_ = rules;
        }

        RenderPathExecutionPlan compile(
            const RenderPathRecipe& recipe,
            const RenderPathCapabilitySet& caps,
            const PassFactoryRegistry* pass_registry = nullptr) const
        {
            RenderPathExecutionPlan plan{};
            plan.recipe_name = recipe.name.empty() ? std::string("unnamed_recipe") : recipe.name;
            plan.backend = recipe.backend;
            plan.technique_mode = recipe.technique_mode;
            plan.render_technique = recipe.render_technique;
            plan.runtime_state = recipe.runtime_defaults;

            auto push_warning = [&plan](const std::string& msg) {
                plan.warnings.push_back(msg);
            };
            auto push_error = [&plan](const std::string& msg) {
                plan.errors.push_back(msg);
            };

            if (recipe.name.empty())
            {
                push_warning("Recipe has no name. Using 'unnamed_recipe'.");
            }
            if (recipe.pass_chain.empty() && rules_.reject_empty_pass_chain)
            {
                push_error("Recipe pass chain is empty.");
            }

            if (!caps.has_backend)
            {
                push_error("Requested backend is not registered in context.");
            }
            else if (caps.backend != recipe.backend)
            {
                push_warning("Capability snapshot backend does not match recipe backend.");
            }

            if (rules_.require_depth_attachment_for_shadow_pass &&
                recipe.wants_shadows &&
                caps.depth_attachment_known &&
                !caps.supports_depth_attachment)
            {
                push_error("Recipe requires shadows, but backend reports no depth attachment support.");
            }

            const bool view_requires_occlusion = render_path_culling_requires_occlusion(recipe.view_culling);
            const bool shadow_requires_occlusion = render_path_culling_requires_occlusion(recipe.shadow_culling);
            const bool view_allows_occlusion = render_path_culling_allows_occlusion(recipe.view_culling);
            const bool shadow_allows_occlusion = render_path_culling_allows_occlusion(recipe.shadow_culling);
            const bool requires_occlusion = view_requires_occlusion || shadow_requires_occlusion;
            const bool allows_occlusion = view_allows_occlusion || shadow_allows_occlusion;

            const auto resolve_entry_pass_id = [](const RenderPathPassEntry& entry) -> PassId
            {
                if (pass_id_is_standard(entry.pass_id)) return entry.pass_id;
                return parse_pass_id(entry.id);
            };

            if (rules_.require_occlusion_support_for_occlusion_culling)
            {
                if (requires_occlusion && !caps.supports_occlusion_query)
                {
                    push_error("Recipe requires occlusion culling, but backend does not support occlusion queries.");
                }
                else if (allows_occlusion && !caps.supports_occlusion_query)
                {
                    push_warning("Recipe allows occlusion culling, but backend does not support occlusion queries. Occlusion defaults will be forced OFF.");
                    plan.runtime_state.view_occlusion_enabled = false;
                    plan.runtime_state.shadow_occlusion_enabled = false;
                }
            }

            auto recipe_has_pass = [&recipe, &resolve_entry_pass_id](PassId pass_id) -> bool
            {
                if (!pass_id_is_standard(pass_id)) return false;
                for (const auto& entry : recipe.pass_chain)
                {
                    if (resolve_entry_pass_id(entry) == pass_id) return true;
                }
                return false;
            };

            if (rules_.require_shadow_map_pass_when_shadows_enabled &&
                recipe.wants_shadows &&
                !recipe_has_pass(PassId::ShadowMap))
            {
                push_error("Recipe enables shadows but pass chain has no 'shadow_map' pass.");
            }

            if (rules_.require_depth_prepass_for_occlusion &&
                requires_occlusion &&
                !recipe_has_pass(PassId::DepthPrepass))
            {
                push_error("Recipe requires occlusion culling but pass chain has no 'depth_prepass' pass.");
            }

            std::unordered_set<std::string> seen_pass_ids{};
            for (const auto& entry : recipe.pass_chain)
            {
                if (entry.id.empty())
                {
                    if (!pass_id_is_standard(entry.pass_id))
                    {
                        if (entry.required) push_error("Pass entry has empty id and is marked required.");
                        else push_warning("Skipping optional pass entry with empty id.");
                        continue;
                    }
                }

                const PassId entry_pass_id = resolve_entry_pass_id(entry);
                if (pass_id_is_standard(entry.pass_id) && !entry.id.empty())
                {
                    const PassId parsed_from_text = parse_pass_id(entry.id);
                    if (pass_id_is_standard(parsed_from_text) && parsed_from_text != entry.pass_id)
                    {
                        push_warning(
                            "Pass entry textual id '" + entry.id +
                            "' does not match typed id '" + pass_id_string(entry.pass_id) +
                            "'. Typed id is used.");
                    }
                }
                const std::string canonical_id =
                    pass_id_is_standard(entry_pass_id) ? pass_id_string(entry_pass_id) : entry.id;

                const auto insert_result = seen_pass_ids.insert(canonical_id);
                if (!insert_result.second)
                {
                    const std::string msg = "Duplicate pass id in recipe: '" + canonical_id + "'.";
                    if (rules_.reject_duplicate_pass_ids) push_error(msg);
                    else push_warning(msg);
                    continue;
                }

                if (!pass_registry)
                {
                    plan.pass_chain.push_back(RenderPathCompiledPass{canonical_id, entry_pass_id, entry.required});
                    continue;
                }

                const bool has_registered_pass =
                    pass_id_is_standard(entry_pass_id)
                        ? pass_registry->has(entry_pass_id)
                        : pass_registry->has(canonical_id);
                if (!has_registered_pass)
                {
                    const std::string msg = "Pass id '" + canonical_id + "' is not registered in PassFactoryRegistry.";
                    if (entry.required && rules_.reject_unknown_required_passes) push_error(msg);
                    else push_warning(msg);
                    continue;
                }

                const std::optional<bool> backend_ok_hint =
                    pass_id_is_standard(entry_pass_id)
                        ? pass_registry->supports_backend_hint(entry_pass_id, recipe.backend)
                        : pass_registry->supports_backend_hint(canonical_id, recipe.backend);
                if (backend_ok_hint.has_value() && !backend_ok_hint.value())
                {
                    const std::string msg =
                        "Pass id '" + canonical_id + "' does not support backend '" +
                        std::string(render_backend_type_name(recipe.backend)) + "'.";
                    if (entry.required) push_error(msg);
                    else push_warning(msg);
                    continue;
                }

                const std::optional<bool> mode_ok_hint =
                    pass_id_is_standard(entry_pass_id)
                        ? pass_registry->supports_technique_mode_hint(entry_pass_id, recipe.technique_mode)
                        : pass_registry->supports_technique_mode_hint(canonical_id, recipe.technique_mode);
                if (mode_ok_hint.has_value() && !mode_ok_hint.value())
                {
                    const std::string msg =
                        "Pass id '" + canonical_id + "' does not support technique mode '" +
                        std::string(technique_mode_name(recipe.technique_mode)) + "'.";
                    if (entry.required) push_error(msg);
                    else push_warning(msg);
                    continue;
                }

                if (backend_ok_hint.has_value() && mode_ok_hint.has_value())
                {
                    plan.pass_chain.push_back(RenderPathCompiledPass{canonical_id, entry_pass_id, entry.required});
                    continue;
                }

                const std::string msg =
                    "Pass id '" + canonical_id +
                    "' has no planner capability hints (backend/mode). "
                    "Register descriptor hints in PassFactoryRegistry for VOP-first planning.";
                if (entry.required) push_error(msg);
                else push_warning(msg);
            }

            if (plan.pass_chain.empty() && rules_.reject_empty_pass_chain)
            {
                push_error("No executable passes remain after recipe compilation.");
            }

            if (!recipe.strict_validation && !plan.errors.empty())
            {
                for (const auto& err : plan.errors)
                {
                    push_warning(std::string("Permissive mode downgrade: ") + err);
                }
                plan.errors.clear();
            }

            if (rules_.reject_empty_pass_chain && plan.pass_chain.empty())
            {
                push_error("Compiled plan has no executable passes.");
            }

            plan.valid = plan.errors.empty();
            return plan;
        }

        RenderPathExecutionPlan compile(
            const RenderPathRecipe& recipe,
            const Context& ctx,
            const PassFactoryRegistry* pass_registry = nullptr) const
        {
            const RenderPathCapabilitySet caps = make_render_path_capability_set(ctx, recipe.backend);
            return compile(recipe, caps, pass_registry);
        }

    private:
        RenderPathCompatibilityRules rules_{};
    };
}
