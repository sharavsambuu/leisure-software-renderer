#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: render_path_compiler.hpp
    МОДУЛЬ: pipeline
    ЗОРИЛГО: Рендер жорыг техникийн хувьд зөв эсэхийг шалгах (validation) болон гүйцэтгэх боломжтой дараалал болгон хөрвүүлэх (compilation).
*/


#include <expected>
#include <string>
#include <string_view>
#include <optional>
#include <unordered_set>
#include <vector>

#include "shs/core/contract_guardrails.hpp"
#include "shs/app/context.hpp"
#include "shs/renderpath/execution/pass_registry.hpp"
#include "shs/renderpath/planning/render_path_capabilities.hpp"
#include "shs/renderpath/planning/render_path_recipe.hpp"
#include "shs/renderpath/planning/substrate_resolution.hpp"
#include "shs/renderpath/planning/technique_profile.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace renderpath
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

        // RP-1: the substrate this pass was RESOLVED onto (req 4 — a resolution
        // output, not an authoring-time choice). `substrate_resolved == false`
        // means the plan carries no resolution for it; the plan is invalid in
        // that case, so this is never a silent default.
        Substrate substrate = Substrate::SoftwareRaster;
        bool substrate_resolved = false;

        // Value semantics (pod test kit requires snapshot equality).
        bool operator==(const RenderPathCompiledPass&) const = default;
    };

    // Native rejection vocabulary (R4 P4.6): each push_error site records its
    // reason at emission. Members mirror renderpath::PathSwapRejectionReason 1:1
    // (mapped in the pod gateway); strings in plan.errors are diagnostics only.
    enum class RenderPathCompileRejection : uint8_t
    {
        CompileInvalid     = 0,
        EmptyPassChain     = 1,
        BackendUnavailable = 2,
        MissingRequiredPass = 3,
        DepthUnsupported   = 4,
        OcclusionUnsupported = 5,
        // RP-1: no admissible substrate for a pass (nothing the host can drive
        // both realizes the pass and satisfies its declared intent).
        SubstrateUnresolved = 6,
        // RP-1: two adjacent passes resolved onto different substrates without a
        // declared interop boundary — the plan-visible half of the RP-2 hybrid
        // rule.
        HybridBoundaryUndeclared = 7
    };

    struct RenderPathExecutionPlan
    {
        std::string recipe_name{};
        RenderBackendType backend = RenderBackendType::Software;
        // RP-1: the policy this plan was resolved under, echoed so the plan is
        // self-describing and snapshot-comparable.
        SubstratePolicy substrate_policy = SubstratePolicy::ExactMatch;
        // RP-1: true once the resolved chain crosses substrates (or execution
        // units) at a declared interop boundary. That is the "one recipe
        // resolves to a hybrid" case, recorded as data.
        bool hybrid = false;
        TechniqueMode technique_mode = TechniqueMode::Forward;
        RenderPathRenderingTechnique render_technique = RenderPathRenderingTechnique::ForwardLit;
        RenderPathRuntimeState runtime_state{};
        std::vector<RenderPathCompiledPass> pass_chain{};
        std::vector<std::string> warnings{};
        std::vector<std::string> errors{};
        RenderPathCompileRejection rejection = RenderPathCompileRejection::CompileInvalid;
        bool valid = false;

        // Value semantics (pod test kit requires snapshot equality).
        bool operator==(const RenderPathExecutionPlan&) const = default;
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
            plan.substrate_policy = recipe.substrate_policy;
            // C2.2 (Rule 17, P1: value invariants live in the pure leaf): the
            // technique-mode transition table — a compiled plan's technique is
            // the table image of its mode. Only legal table rows compile; a
            // hand-mismatched pair is the negative test (enforced twin).
            SHS_CONTRACT_ASSERT(
                render_path_rendering_technique_for_mode(plan.technique_mode) == plan.render_technique);
            plan.runtime_state = recipe.runtime_defaults;

            auto push_warning = [&plan](const std::string& msg) {
                plan.warnings.push_back(msg);
            };
            auto push_error = [&plan](const std::string& msg,
                RenderPathCompileRejection reason = RenderPathCompileRejection::CompileInvalid)
            {
                if (plan.errors.empty()) plan.rejection = reason; // first error wins
                plan.errors.push_back(msg);
            };

            if (recipe.name.empty())
            {
                push_warning("Recipe has no name. Using 'unnamed_recipe'.");
            }
            if (recipe.pass_chain.empty() && rules_.reject_empty_pass_chain)
            {
                push_error("Recipe pass chain is empty.", RenderPathCompileRejection::EmptyPassChain);
            }

            if (!caps.has_backend)
            {
                push_error("Requested backend is not registered in context.", RenderPathCompileRejection::BackendUnavailable);
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
                push_error("Recipe requires shadows, but backend reports no depth attachment support.", RenderPathCompileRejection::DepthUnsupported);
            }

            const bool view_requires_occlusion = render_path_culling_requires_occlusion(recipe.view_culling);
            const bool shadow_requires_occlusion = render_path_culling_requires_occlusion(recipe.shadow_culling);
            const bool view_allows_occlusion = render_path_culling_allows_occlusion(recipe.view_culling);
            const bool shadow_allows_occlusion = render_path_culling_allows_occlusion(recipe.shadow_culling);
            const bool requires_occlusion = view_requires_occlusion || shadow_requires_occlusion;
            const bool allows_occlusion = view_allows_occlusion || shadow_allows_occlusion;

            const auto resolve_entry_pass_id = [](const RenderPathPassEntry& entry) -> PassId
            {
                // A meaningful typed id wins: builtin, or a consumer-minted open
                // id. Otherwise fall back to the textual builtin spelling.
                if (pass_id_in_valid_range(entry.pass_id)) return entry.pass_id;
                return parse_pass_id(entry.id);
            };

            // The plan's canonical key for an entry: the builtin spelling, the
            // name the registry minted for an open id, or the entry's own
            // textual key (string-keyed consumer pass).
            const auto resolve_canonical_pass_id =
                [pass_registry](const RenderPathPassEntry& entry, PassId entry_pass_id) -> std::string
            {
                if (pass_id_is_builtin(entry_pass_id)) return pass_id_string(entry_pass_id);
                if (pass_registry != nullptr)
                {
                    const std::optional<std::string_view> registered =
                        pass_registry->pass_id_registered_name(entry_pass_id);
                    if (registered.has_value()) return std::string(*registered);
                }
                return entry.id;
            };

            // RP-1 (req 4): substrate is a RESOLUTION OUTPUT. Per pass:
            //   intent        <- the entry's declared RenderDomain
            //   realizability <- the pass registry's substrate mask
            //   admissible    <- the capability snapshot's available substrates
            //   choice        <- the recipe's policy
            // An empty available mask means "single-substrate host / unknown",
            // which resolves as the declared substrate only — so a recipe that
            // neither widens the mask nor states a policy resolves exactly where
            // it did before this change.
            const Substrate declared_substrate = substrate_of_backend(recipe.backend);
            // ADMISSIBLE set of this compile: what the host can drive, or — on a
            // host that declares nothing — the single declared substrate. Every
            // pre-RP-1 caller is in the second case, so every predicate below
            // that used to be phrased "matches recipe.backend" is now phrased
            // "intersects the admissible set" and answers identically.
            const uint32_t admissible_mask = (caps.available_substrate_mask != substrate_mask_none())
                ? caps.available_substrate_mask
                : substrate_bit(declared_substrate);
            bool has_previous = false;
            Substrate previous_substrate = declared_substrate;
            bool previous_declares_interop = false;

            auto emit_pass = [&](const std::string& canonical_id,
                                 PassId entry_pass_id,
                                 const RenderPathPassEntry& entry,
                                 const std::optional<uint32_t>& realized_hint) -> bool
            {
                SubstrateResolutionRequest request{};
                request.intent = entry.domain;
                request.available_mask = caps.available_substrate_mask;
                request.declared = declared_substrate;
                request.policy = recipe.substrate_policy;
                request.has_predecessor = has_previous;
                request.predecessor = previous_substrate;
                if (realized_hint.has_value())
                {
                    request.realized_mask = realized_hint.value();
                    request.realized_known = true;
                }

                const SubstrateResolution resolution = resolve_substrate(request);
                if (!resolution.resolved)
                {
                    const std::string msg =
                        "Pass id '" + canonical_id +
                        "' has no admissible substrate: nothing this host can drive "
                        "both realizes the pass and satisfies its declared intent ("
                        + render_domain_name(entry.domain) + ").";
                    if (entry.required) push_error(msg, RenderPathCompileRejection::SubstrateUnresolved);
                    else push_warning(msg);
                    return false;
                }

                const bool declares_interop = (pass_registry != nullptr)
                    ? pass_registry->declares_interop_hint(canonical_id).value_or(false)
                    : false;

                // Hybrid legality (RP-2 ruling): crossing execution units —
                // which always implies crossing substrates, since
                // `execution_unit_of` is a function of the substrate — is legal
                // only where one of the crossing passes declares an interop
                // boundary. Otherwise this is a rejection, not the warning the
                // retired behaviour emitted.
                if (has_previous && substrates_cross(resolution.substrate, previous_substrate))
                {
                    if (previous_declares_interop || declares_interop)
                    {
                        plan.hybrid = true;
                    }
                    else
                    {
                        const std::string msg =
                            "Pass id '" + canonical_id + "' resolves to '" +
                            std::string(substrate_name(resolution.substrate)) +
                            "' but the previous pass resolved to '" +
                            std::string(substrate_name(previous_substrate)) +
                            "': crossing substrates requires a declared interop boundary.";
                        push_error(msg, RenderPathCompileRejection::HybridBoundaryUndeclared);
                        return false;
                    }
                }

                plan.pass_chain.push_back(RenderPathCompiledPass{
                    canonical_id, entry_pass_id, entry.required, resolution.substrate, true});
                has_previous = true;
                previous_substrate = resolution.substrate;
                previous_declares_interop = declares_interop;
                return true;
            };

            if (rules_.require_occlusion_support_for_occlusion_culling)
            {
                if (requires_occlusion && !caps.supports_occlusion_query)
                {
                    push_error("Recipe requires occlusion culling, but backend does not support occlusion queries.", RenderPathCompileRejection::OcclusionUnsupported);
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
                if (!pass_id_in_valid_range(pass_id)) return false;
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
                push_error("Recipe enables shadows but pass chain has no 'shadow_map' pass.", RenderPathCompileRejection::MissingRequiredPass);
            }

            if (rules_.require_depth_prepass_for_occlusion &&
                requires_occlusion &&
                !recipe_has_pass(PassId::DepthPrepass))
            {
                push_error("Recipe requires occlusion culling but pass chain has no 'depth_prepass' pass.", RenderPathCompileRejection::MissingRequiredPass);
            }

            std::unordered_set<std::string> seen_pass_ids{};
            for (const auto& entry : recipe.pass_chain)
            {
                if (entry.id.empty())
                {
                    // A builtin id carries its own spelling; an open id is fine
                    // *if* this registry minted a name for it. Anything else has
                    // no key at all and cannot be planned.
                    const bool named_without_text =
                        pass_id_is_builtin(entry.pass_id) ||
                        (pass_registry != nullptr &&
                         pass_registry->pass_id_registered_name(entry.pass_id).has_value());
                    if (!named_without_text)
                    {
                        if (entry.required) push_error("Pass entry has empty id and is marked required.", RenderPathCompileRejection::CompileInvalid);
                        else push_warning("Skipping optional pass entry with empty id.");
                        continue;
                    }
                }

                const PassId entry_pass_id = resolve_entry_pass_id(entry);
                if (pass_id_is_builtin(entry.pass_id) && !entry.id.empty())
                {
                    const PassId parsed_from_text = parse_pass_id(entry.id);
                    if (pass_id_is_builtin(parsed_from_text) && parsed_from_text != entry.pass_id)
                    {
                        push_warning(
                            "Pass entry textual id '" + entry.id +
                            "' does not match typed id '" + pass_id_string(entry.pass_id) +
                            "'. Typed id is used.");
                    }
                }
                else if (pass_id_is_open(entry.pass_id) && !entry.id.empty() && pass_registry != nullptr)
                {
                    // Open ids have no static spelling: their check is exact-name
                    // against what this registry minted.
                    const std::optional<std::string_view> registered =
                        pass_registry->pass_id_registered_name(entry.pass_id);
                    if (registered.has_value() && *registered != entry.id)
                    {
                        push_warning(
                            "Pass entry textual id '" + entry.id +
                            "' does not match registered name '" + std::string(*registered) +
                            "' of its typed id. The registered name is used.");
                    }
                }
                const std::string canonical_id = resolve_canonical_pass_id(entry, entry_pass_id);

                const auto insert_result = seen_pass_ids.insert(canonical_id);
                if (!insert_result.second)
                {
                    const std::string msg = "Duplicate pass id in recipe: '" + canonical_id + "'.";
                    if (rules_.reject_duplicate_pass_ids) push_error(msg, RenderPathCompileRejection::CompileInvalid);
                    else push_warning(msg);
                    continue;
                }

                if (!pass_registry)
                {
                    (void)emit_pass(canonical_id, entry_pass_id, entry, std::nullopt);
                    continue;
                }

                // canonical_id IS the registry key for every resolvable entry:
                // the builtin spelling, the minted open-id name, or the entry's
                // textual key. Typed and textual lookups therefore converge here.
                const bool has_registered_pass = pass_registry->has(canonical_id);
                if (!has_registered_pass)
                {
                    const std::string msg = "Pass id '" + canonical_id + "' is not registered in PassFactoryRegistry.";
                    if (entry.required && rules_.reject_unknown_required_passes) push_error(msg, RenderPathCompileRejection::BackendUnavailable);
                    else push_warning(msg);
                    continue;
                }

                const std::optional<uint32_t> realized_hint =
                    pass_registry->realized_substrate_mask_hint(canonical_id);
                const std::optional<bool> backend_ok_hint = realized_hint.has_value()
                    ? std::optional<bool>((realized_hint.value() & admissible_mask) != 0u)
                    : std::nullopt;
                if (backend_ok_hint.has_value() && !backend_ok_hint.value())
                {
                    const std::string msg =
                        "Pass id '" + canonical_id + "' is not realizable on any substrate " +
                        "this host can drive (admissible: " +
                        render_backend_type_name(recipe.backend) + ").";
                    if (entry.required) push_error(msg, RenderPathCompileRejection::BackendUnavailable);
                    else push_warning(msg);
                    continue;
                }

                const std::optional<bool> mode_ok_hint =
                    pass_registry->supports_technique_mode_hint(canonical_id, recipe.technique_mode);
                if (mode_ok_hint.has_value() && !mode_ok_hint.value())
                {
                    const std::string msg =
                        "Pass id '" + canonical_id + "' does not support technique mode '" +
                        std::string(technique_mode_name(recipe.technique_mode)) + "'.";
                    if (entry.required) push_error(msg, RenderPathCompileRejection::CompileInvalid);
                    else push_warning(msg);
                    continue;
                }

                if (backend_ok_hint.has_value() && mode_ok_hint.has_value())
                {
                    (void)emit_pass(canonical_id, entry_pass_id, entry, realized_hint);
                    continue;
                }

                const std::string msg =
                    "Pass id '" + canonical_id +
                    "' has no planner capability hints (backend/mode). "
                    "Register descriptor hints in PassFactoryRegistry for VOP-first planning.";
                if (entry.required) push_error(msg, RenderPathCompileRejection::CompileInvalid);
                else push_warning(msg);
            }

            if (plan.pass_chain.empty() && rules_.reject_empty_pass_chain)
            {
                push_error("No executable passes remain after recipe compilation.", RenderPathCompileRejection::EmptyPassChain);
            }

            if (!recipe.strict_validation && !plan.errors.empty())
            {
                for (const auto& err : plan.errors)
                {
                    push_warning(std::string("Permissive mode downgrade: ") + err);
                }
                plan.errors.clear();
                plan.rejection = RenderPathCompileRejection::CompileInvalid;
            }

            if (rules_.reject_empty_pass_chain && plan.pass_chain.empty())
            {
                push_error("Compiled plan has no executable passes.", RenderPathCompileRejection::EmptyPassChain);
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

        // Fallible compile (R4 P4.6): valid plan or native rejection reason.
        // Prefer this over scraping plan.errors text (see gateway history).
        std::expected<RenderPathExecutionPlan, RenderPathCompileRejection> try_compile(
            const RenderPathRecipe& recipe,
            const RenderPathCapabilitySet& caps,
            const PassFactoryRegistry* pass_registry = nullptr) const
        {
            RenderPathExecutionPlan plan = compile(recipe, caps, pass_registry);
            if (!plan.valid)
            {
                return std::unexpected(plan.rejection);
            }
            return plan;
        }

    private:
        RenderPathCompatibilityRules rules_{};
    };

    } // inline namespace renderpath
}
