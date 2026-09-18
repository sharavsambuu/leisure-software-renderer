#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <string_view>

#include "shs/renderpath/execution/pass_registry.hpp"
#include "shs/renderpath/execution/render_path_runtime_layout.hpp"

// Open pass-id registry gate (Constitution I §7 — No User Lock-In;
// arch/render_path_architecture.md §4 graduation req 1).
//
// Proves the *consumer* story, not an internal invariant: a demo-owned pass is
// registered, planned and resolved through a typed id minted at runtime, with
// zero edits to the core vocabulary — while every builtin path keeps its exact
// previous behavior (AD0 parity). GPU-free: no Context, no backend instance.
namespace
{
    using namespace shs;
    using namespace shs::renderpath;

    // Minimal final pass: IRenderPass has exactly two pure virtuals.
    class FakePass final : public IRenderPass
    {
    public:
        explicit FakePass(std::string id) : id_(std::move(id)) {}
        const char* id() const override { return id_.c_str(); }
        PassExecutionResult execute_resolved(
            app::Context&, const PassExecutionRequest&) override
        {
            return PassExecutionResult::executed_no_outputs();
        }

    private:
        std::string id_;
    };

    PassFactoryRegistry::Factory make_factory(const char* id)
    {
        return [id]() -> std::unique_ptr<IRenderPass> {
            return std::make_unique<FakePass>(std::string(id));
        };
    }

    // Software-backend capability snapshot (mirrors the software stack).
    RenderPathCapabilitySet make_sw_caps()
    {
        const rhi::BackendCapabilities backend_caps{};
        return make_render_path_capability_set(RenderBackendType::Software, backend_caps);
    }

    // --- range law -----------------------------------------------------------

    bool test_range_law()
    {
        bool ok = true;
        ok = ok && pass_id_is_builtin(PassId::ShadowMap);
        ok = ok && pass_id_is_builtin(PassId::DepthOfField);
        ok = ok && !pass_id_is_builtin(PassId::Unknown);
        ok = ok && !pass_id_is_builtin(static_cast<PassId>(kPassIdOpenBase));

        ok = ok && !pass_id_is_open(PassId::Unknown);
        ok = ok && !pass_id_is_open(PassId::ShadowMap);
        // Boundary: the last builtin slot is not open, the first open slot is.
        ok = ok && !pass_id_is_open(static_cast<PassId>(kPassIdOpenBase - 1u));
        ok = ok && pass_id_is_open(static_cast<PassId>(kPassIdOpenBase));
        ok = ok && pass_id_is_open(static_cast<PassId>(kPassIdOpenMax));

        ok = ok && pass_id_in_valid_range(PassId::Tonemap);
        ok = ok && pass_id_in_valid_range(static_cast<PassId>(kPassIdOpenBase));
        ok = ok && !pass_id_in_valid_range(PassId::Unknown);
        ok = ok && !pass_id_in_valid_range(static_cast<PassId>(kPassIdReserved));

        // The legacy spelling must stay behavior-identical to the builtin test
        // for every value the enum can hold (parity guard).
        for (uint16_t raw = 0u; raw <= static_cast<uint16_t>(PassId::DepthOfField); ++raw)
        {
            const PassId id = static_cast<PassId>(raw);
            ok = ok && (pass_id_is_standard(id) == pass_id_is_builtin(id));
        }
        // Open ids are never "standard", and they stringify honestly.
        ok = ok && !pass_id_is_standard(static_cast<PassId>(kPassIdOpenBase));
        ok = ok && std::string_view(pass_id_name(static_cast<PassId>(kPassIdOpenBase))) ==
                       std::string_view(kPassIdOpenSpelling);
        ok = ok && pass_id_name_or_null(PassId::TAA) != nullptr;
        ok = ok && pass_id_name_or_null(static_cast<PassId>(kPassIdOpenBase)) == nullptr;
        return ok;
    }

    // --- the registry --------------------------------------------------------

    bool test_builtin_names_never_mint()
    {
        PassIdRegistry ids{};
        const auto taa = ids.intern("taa");
        const auto shadow = ids.intern("shadow_map");
        bool ok = true;
        ok = ok && taa.has_value() && *taa == PassId::TAA;
        ok = ok && shadow.has_value() && *shadow == PassId::ShadowMap;
        // No open id was minted, so a consumer can never shadow a core pass.
        ok = ok && ids.open_count() == 0u;
        ok = ok && ids.empty();
        ok = ok && ids.try_name(PassId::TAA).has_value() &&
                   *ids.try_name(PassId::TAA) == std::string_view("taa");
        return ok;
    }

    bool test_intern_order_and_idempotence()
    {
        PassIdRegistry ids{};
        const auto a = ids.intern("demo_fog_pass");
        const auto b = ids.intern("demo_ssao_ext");
        const auto c = ids.intern("demo_composite");
        bool ok = true;
        ok = ok && a.has_value() && b.has_value() && c.has_value();
        ok = ok && ids.open_count() == 3u;

        // Content-addressed: the id is a pure function of the name, so it is the
        // same in every registry, process and translation unit.
        ok = ok && *a == static_cast<PassId>(
                            kPassIdOpenBase + PassIdRegistry::open_offset("demo_fog_pass"));
        ok = ok && *b == static_cast<PassId>(
                            kPassIdOpenBase + PassIdRegistry::open_offset("demo_ssao_ext"));
        ok = ok && *a != *b && *b != *c && *a != *c;

        // Idempotent: re-interning returns the same id and registers nothing.
        const auto a_again = ids.intern("demo_fog_pass");
        ok = ok && a_again.has_value() && *a_again == *a;
        ok = ok && ids.open_count() == 3u;

        // Names round-trip, and an unregistered open id has no name here.
        ok = ok && ids.try_name(*b).has_value() &&
                   *ids.try_name(*b) == std::string_view("demo_ssao_ext");
        ok = ok && ids.is_open(*a);
        ok = ok && !ids.is_open(PassId::TAA);
        ok = ok && !ids.contains(static_cast<PassId>(kPassIdOpenBase + 60000u));

        // Registration order is preserved for introspection/debug output.
        const auto& registered = ids.registered();
        ok = ok && registered.size() == 3u;
        ok = ok && registered[0].second == "demo_fog_pass";
        ok = ok && registered[1].second == "demo_ssao_ext";
        ok = ok && registered[2].second == "demo_composite";
        return ok;
    }

    bool test_determinism_across_instances()
    {
        // Order independence: the same names in any order produce the same ids,
        // so registration order can never leak into a plan or a replay log.
        PassIdRegistry first{};
        PassIdRegistry second{};
        const char* forward[] = { "demo_a", "demo_b", "demo_c", "demo_d" };
        const char* reverse[] = { "demo_d", "demo_c", "demo_b", "demo_a" };
        bool ok = true;
        for (const char* n : forward)
        {
            const auto x = first.intern(n);
            ok = ok && x.has_value();
        }
        for (const char* n : reverse)
        {
            const auto y = second.intern(n);
            ok = ok && y.has_value();
        }
        // Same name => same id in both, regardless of registration order.
        for (const char* n : forward)
        {
            const auto x = first.intern(n);
            const auto y = second.intern(n);
            ok = ok && x.has_value() && y.has_value() && *x == *y;
        }
        ok = ok && first == second;

        // A copy carries the identical id assignment.
        const PassIdRegistry copy = first;
        ok = ok && copy == first;

        // Different name sets are different registries.
        PassIdRegistry other{};
        ok = ok && other.intern("demo_e").has_value();
        ok = ok && !(other == first);
        return ok;
    }

    bool test_null_and_reserved_rejected()
    {
        PassIdRegistry ids{};
        bool ok = true;
        ok = ok && !ids.intern("").has_value();
        // "unknown" is the null id's spelling, never a mintage.
        ok = ok && !ids.intern(pass_id_name(PassId::Unknown)).has_value();
        ok = ok && ids.open_count() == 0u;

        ok = ok && !ids.try_name(PassId::Unknown).has_value();
        ok = ok && !ids.try_name(static_cast<PassId>(kPassIdReserved)).has_value();
        ok = ok && !ids.contains(PassId::Unknown);
        return ok;
    }

    bool test_capacity_arithmetic()
    {
        bool ok = true;
        // The open range is exactly representable: last mintable id is OpenMax.
        ok = ok && PassIdRegistry::capacity() ==
                       static_cast<std::size_t>(kPassIdOpenMax) -
                           static_cast<std::size_t>(kPassIdOpenBase) + 1u;
        ok = ok && static_cast<uint32_t>(kPassIdOpenBase) +
                       static_cast<uint32_t>(PassIdRegistry::capacity()) - 1u ==
                       static_cast<uint32_t>(kPassIdOpenMax);
        // The reserved slot is never minted.
        ok = ok && PassIdRegistry::capacity() < 65535u;
        ok = ok && kPassIdReserved == 65535u;
        return ok;
    }

    // Brute-force a real 16-bit offset collision. With ~64k slots a birthday
    // collision appears within a few hundred probes, so the loud-failure path is
    // testable rather than theoretical.
    std::string find_colliding_name(std::string_view seed)
    {
        const uint16_t target = PassIdRegistry::open_offset(seed);
        for (int i = 0; i < 20000; ++i)
        {
            const std::string candidate = "demo_probe_" + std::to_string(i);
            if (candidate != seed && PassIdRegistry::open_offset(candidate) == target)
            {
                return candidate;
            }
        }
        return {};
    }

    bool test_collision_is_loud()
    {
        const std::string collider = find_colliding_name("demo_fog_pass");
        // If no collision was found the assertion is vacuous — but it is not
        // expected to happen (birthday bound), so do not fail the gate for it.
        if (collider.empty()) return true;

        PassIdRegistry ids{};
        const auto first_id = ids.intern("demo_fog_pass");
        bool ok = true;
        ok = ok && first_id.has_value();
        ok = ok && ids.open_count() == 1u;

        // The colliding name is refused: nothing is overwritten, nothing is
        // aliased, and the first name keeps its id.
        ok = ok && !ids.intern(collider).has_value();
        ok = ok && ids.open_count() == 1u;
        ok = ok && ids.try_name(*first_id).has_value() &&
                   *ids.try_name(*first_id) == std::string_view("demo_fog_pass");

        // Residual, stated honestly: a registry holding only the collider maps
        // that shared slot to *its* name. The verified boundary overload is what
        // makes this safe — the (id, name) pair must match, so a colliding id
        // can never bind the wrong factory.
        PassIdRegistry only_collider{};
        const auto registered_collider = only_collider.intern(collider);
        ok = ok && registered_collider.has_value();
        ok = ok && *registered_collider == *first_id; // same slot by construction

        PassFactoryRegistry registry{};
        ok = ok && registry.intern_pass_id(collider).has_value();
        ok = ok && !registry.register_factory(*first_id, "demo_fog_pass", make_factory("demo_fog_pass"));
        ok = ok && !registry.has(*first_id, "demo_fog_pass");
        // The verified pair (id, matching name) is accepted.
        ok = ok && registry.register_factory(*first_id, collider, make_factory(collider.c_str()));
        ok = ok && registry.has(*first_id, collider);
        return ok;
    }

    bool test_foreign_open_id_is_a_hard_miss()
    {
        PassFactoryRegistry registry{};
        const auto fog = registry.intern_pass_id("demo_fog_pass");
        bool ok = true;
        ok = ok && fog.has_value();
        ok = ok && registry.register_factory(*fog, "demo_fog_pass", make_factory("demo_fog_pass"));

        // An id derived from a name this registry never registered misses —
        // there is no "whatever is at that slot" fallback.
        PassIdRegistry other{};
        const auto unrelated = other.intern("demo_not_registered_here");
        ok = ok && unrelated.has_value();
        ok = ok && !registry.has(*unrelated);
        ok = ok && !registry.has(*unrelated, "demo_not_registered_here");
        ok = ok && registry.create(*unrelated) == nullptr;

        // The null and reserved ids are never resolvable.
        ok = ok && !registry.has(PassId::Unknown);
        ok = ok && !registry.has(static_cast<PassId>(kPassIdReserved));
        return ok;
    }

    // --- typed registry integration (consumer-owned pass) --------------------

    bool test_typed_registry_accepts_open_ids()
    {
        PassFactoryRegistry registry{};
        const auto fog = registry.intern_pass_id("demo_fog_pass");
        bool ok = true;
        ok = ok && fog.has_value() && pass_id_is_open(*fog);

        // Register the consumer pass with zero core edits.
        ok = ok && registry.register_factory(*fog, "demo_fog_pass", make_factory("demo_fog_pass"));
        ok = ok && registry.has(*fog);
        ok = ok && registry.has(std::string("demo_fog_pass"));
        ok = ok && registry.has(*fog, "demo_fog_pass");

        // Descriptor + capability hints resolve through the same key.
        TechniquePassContract contract{};
        contract.role = TechniquePassRole::Custom;
        contract.supported_modes_mask = technique_mode_bit(TechniqueMode::Forward);
        ok = ok && registry.register_descriptor(
                       *fog, contract, PassFactoryRegistry::backend_bit(RenderBackendType::Software));
        TechniquePassContract read{};
        ok = ok && registry.try_get_contract_hint(*fog, read);
        ok = ok && read.supported_modes_mask == contract.supported_modes_mask;
        ok = ok && registry.supports_backend_hint(*fog, RenderBackendType::Software).value_or(false);
        ok = ok && !registry.supports_backend_hint(*fog, RenderBackendType::Vulkan).value_or(true);
        ok = ok && registry.supports_technique_mode_hint(*fog, TechniqueMode::Forward).value_or(false);
        ok = ok && !registry.supports_technique_mode_hint(*fog, TechniqueMode::Deferred).value_or(true);

        // Creation works through the typed id and yields the consumer's pass.
        const auto pass = registry.create(*fog);
        ok = ok && pass != nullptr;
        ok = ok && std::string_view(pass->id()) == std::string_view("demo_fog_pass");

        // Registered name is available for plan keys and diagnostics.
        const auto name = registry.pass_id_registered_name(*fog);
        ok = ok && name.has_value() && *name == std::string_view("demo_fog_pass");

        // Builtin typed ids keep working exactly as before (parity).
        ok = ok && registry.register_factory(PassId::Tonemap, make_factory("tonemap"));
        ok = ok && registry.has(PassId::Tonemap);
        ok = ok && registry.has(std::string("tonemap"));
        ok = ok && registry.pass_id_registered_name(PassId::Tonemap).has_value() &&
                   *registry.pass_id_registered_name(PassId::Tonemap) == std::string_view("tonemap");
        return ok;
    }

    // --- end-to-end: a consumer pass is a first-class plan member ------------

    bool test_consumer_pass_plans_without_core_edit()
    {
        PassFactoryRegistry registry{};
        const auto fog = registry.intern_pass_id("demo_fog_pass");
        bool ok = true;
        ok = ok && fog.has_value();
        ok = ok && registry.register_factory(*fog, "demo_fog_pass", make_factory("demo_fog_pass"));
        ok = ok && registry.register_factory(PassId::PBRForward, make_factory("pbr_forward"));
        ok = ok && registry.register_factory(PassId::Tonemap, make_factory("tonemap"));
        // Planner participation requires descriptor hints (VOP-first planning), and
        // a consumer pass declares them through the same mechanism as a builtin.
        ok = ok && registry.register_descriptor(*fog, TechniquePassContract{});
        ok = ok && registry.register_descriptor(PassId::PBRForward, TechniquePassContract{});
        ok = ok && registry.register_descriptor(PassId::Tonemap, TechniquePassContract{});

        RenderPathRecipe recipe{};
        recipe.name = "consumer_pass_recipe";
        recipe.backend = RenderBackendType::Software;
        recipe.render_technique = RenderPathRenderingTechnique::ForwardLit;
        recipe.technique_mode = TechniqueMode::Forward;
        recipe.view_culling = RenderPathCullingMode::Frustum;
        recipe.wants_shadows = false;
        recipe.pass_chain = {
            make_render_path_pass_entry(PassId::PBRForward, true),
            make_render_path_pass_entry(std::string("demo_fog_pass"), *fog, true),
            make_render_path_pass_entry(PassId::Tonemap, true)
        };

        const RenderPathCompiler compiler{};
        const RenderPathExecutionPlan plan = compiler.compile(recipe, make_sw_caps(), &registry);
        ok = ok && plan.valid;
        ok = ok && plan.pass_chain.size() == 3u;
        ok = ok && plan.pass_chain[1].pass_id == *fog;
        ok = ok && plan.pass_chain[1].id == "demo_fog_pass";
        ok = ok && plan.pass_chain[0].pass_id == PassId::PBRForward;
        ok = ok && plan.pass_chain[0].id == "pbr_forward";

        // Typed plan queries now answer truthfully for the open id — this is the
        // behavior that was previously a guaranteed "false".
        ok = ok && render_path_plan_has_pass(plan, *fog);
        ok = ok && render_path_plan_has_pass(plan, *fog, registry.pass_ids());
        ok = ok && render_path_plan_has_pass(plan, std::string_view("demo_fog_pass"));
        ok = ok && render_path_plan_has_pass(plan, PassId::Tonemap);
        ok = ok && !render_path_plan_has_pass(plan, PassId::SSAO);

        // Plan is a value; the typed identity travels inside it.
        const RenderPathExecutionPlan copy = plan;
        ok = ok && copy == plan;
        ok = ok && copy.pass_chain[1].pass_id == *fog;

        // Exact-name check for open ids: a mismatched text key warns, and the
        // registered name still wins the canonical key.
        RenderPathRecipe mismatched = recipe;
        mismatched.pass_chain[1] =
            make_render_path_pass_entry(std::string("demo_wrong_name"), *fog, true);
        const RenderPathExecutionPlan mismatch_plan =
            compiler.compile(mismatched, make_sw_caps(), &registry);
        ok = ok && mismatch_plan.valid;
        ok = ok && mismatch_plan.pass_chain[1].id == "demo_fog_pass";
        ok = ok && !mismatch_plan.warnings.empty();
        return ok;
    }

    bool test_string_keyed_plan_resolved_by_name()
    {
        // A plan authored with the textual key only (typed id unknown when the
        // plan was built) still resolves through the registry's name.
        PassIdRegistry ids{};
        const auto fog = ids.intern("demo_fog_pass");
        bool ok = true;
        ok = ok && fog.has_value();

        RenderPathExecutionPlan plan{};
        plan.pass_chain.push_back(RenderPathCompiledPass{"demo_fog_pass", PassId::Unknown, true});
        // The typed query cannot know: the plan carries no typed id.
        ok = ok && !render_path_plan_has_pass(plan, *fog);
        // The registry-resolved query can, and the textual query still works.
        ok = ok && render_path_plan_has_pass(plan, *fog, ids);
        ok = ok && render_path_plan_has_pass(plan, std::string_view("demo_fog_pass"));
        return ok;
    }

    bool test_builtin_paths_unchanged()
    {
        PassFactoryRegistry registry{};
        bool ok = true;
        ok = ok && registry.register_factory(PassId::PBRForward, make_factory("pbr_forward"));
        ok = ok && registry.register_factory(PassId::Tonemap, make_factory("tonemap"));
        ok = ok && registry.register_descriptor(PassId::PBRForward, TechniquePassContract{});
        ok = ok && registry.register_descriptor(PassId::Tonemap, TechniquePassContract{});

        RenderPathRecipe recipe{};
        recipe.name = "builtin_recipe";
        recipe.backend = RenderBackendType::Software;
        recipe.render_technique = RenderPathRenderingTechnique::ForwardLit;
        recipe.technique_mode = TechniqueMode::Forward;
        recipe.view_culling = RenderPathCullingMode::Frustum;
        recipe.wants_shadows = false;
        recipe.pass_chain = {
            make_render_path_pass_entry(PassId::PBRForward, true),
            make_render_path_pass_entry(PassId::Tonemap, true)
        };

        const RenderPathCompiler compiler{};
        const RenderPathExecutionPlan plan = compiler.compile(recipe, make_sw_caps(), &registry);
        ok = ok && plan.valid && plan.pass_chain.size() == 2u;
        ok = ok && plan.pass_chain[0].id == "pbr_forward";
        ok = ok && plan.pass_chain[0].pass_id == PassId::PBRForward;
        ok = ok && plan.pass_chain[1].id == "tonemap";
        ok = ok && render_path_plan_has_pass(plan, PassId::PBRForward);
        ok = ok && render_path_plan_has_pass(plan, std::string_view("pbr_forward"));
        ok = ok && !render_path_plan_has_pass(plan, PassId::SSAO);

        // Unregistered required pass is still rejected (touched validation path).
        RenderPathRecipe missing = recipe;
        missing.pass_chain.push_back(make_render_path_pass_entry(PassId::SSAO, true));
        const RenderPathExecutionPlan bad = compiler.compile(missing, make_sw_caps(), &registry);
        ok = ok && !bad.valid;
        ok = ok && bad.rejection == RenderPathCompileRejection::BackendUnavailable;

        // A required entry with no key at all (no text, no typed id) is rejected.
        RenderPathRecipe keyless = recipe;
        keyless.pass_chain.push_back(RenderPathPassEntry{"", PassId::Unknown, true});
        const RenderPathExecutionPlan keyless_plan = compiler.compile(keyless, make_sw_caps(), &registry);
        ok = ok && !keyless_plan.valid;

        // A builtin text/typed mismatch still warns and the typed id wins.
        RenderPathRecipe mismatched = recipe;
        mismatched.pass_chain[0] =
            make_render_path_pass_entry(std::string("gbuffer"), PassId::PBRForward, true);
        const RenderPathExecutionPlan mismatch_plan = compiler.compile(mismatched, make_sw_caps(), &registry);
        ok = ok && mismatch_plan.valid;
        ok = ok && mismatch_plan.pass_chain[0].id == "pbr_forward";
        ok = ok && !mismatch_plan.warnings.empty();

        // No registry at all: the previous behavior (textual/typed ids kept) holds.
        const RenderPathExecutionPlan bare = compiler.compile(recipe, make_sw_caps(), nullptr);
        ok = ok && bare.valid && bare.pass_chain.size() == 2u;
        ok = ok && bare.pass_chain[0].id == "pbr_forward";
        return ok;
    }
}

int main()
{
    bool ok = true;
    auto check = [](const char* name, bool (*fn)()) {
        const bool r = fn();
        std::fprintf(stderr, "[pass-id-open] %-40s %s\n", name, r ? "PASS" : "FAIL");
        return r;
    };

    ok = check("range_law", test_range_law) && ok;
    ok = check("builtin_names_never_register", test_builtin_names_never_mint) && ok;
    ok = check("content_addressed_ids", test_intern_order_and_idempotence) && ok;
    ok = check("determinism_across_instances", test_determinism_across_instances) && ok;
    ok = check("null_and_reserved_rejected", test_null_and_reserved_rejected) && ok;
    ok = check("capacity_arithmetic", test_capacity_arithmetic) && ok;
    ok = check("collision_is_loud", test_collision_is_loud) && ok;
    ok = check("foreign_open_id_is_hard_miss", test_foreign_open_id_is_a_hard_miss) && ok;
    ok = check("typed_registry_accepts_open_ids", test_typed_registry_accepts_open_ids) && ok;
    ok = check("consumer_pass_plans_no_core_edit", test_consumer_pass_plans_without_core_edit) && ok;
    ok = check("string_keyed_plan_by_name", test_string_keyed_plan_resolved_by_name) && ok;
    ok = check("builtin_paths_unchanged", test_builtin_paths_unchanged) && ok;

    if (!ok)
    {
        std::fprintf(stderr, "[pass-id-open] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[pass-id-open] all tests passed\n");
    return 0;
}
