#include <cstdio>
#include <memory_resource>
#include <string>
#include <vector>

#include "shs/core/contract_guardrails.hpp"
#include "shs/renderpath/renderpath.gateway.hpp"

// C2 renderpath pilot negative tests (adoption todo C2.1-C2.3). Compiled twice
// by CMake like contract_guardrails_tests.cpp:
// - shs_renderer_contract_pilot_tests        (SHS_CONTRACTS_ENFORCED: the
//   hand-broken inputs reach the violation handler with the right kind)
// - shs_renderer_contract_pilot_release_tests (assume path: house input passes
//   through with checks discarded; verified by compilation + run)
namespace
{
    // Software-backend capability snapshot as a pure value (mirrors
    // tests/renderpath_tests.cpp; no Context, no backend instances).
    shs::renderpath::RenderPathCapabilitySet make_sw_caps()
    {
        shs::rhi::BackendCapabilities backend_caps{};
        return shs::renderpath::make_render_path_capability_set(
            shs::RenderBackendType::Software, backend_caps);
    }

    shs::renderpath::RenderPathRecipe make_forward_recipe(const char* name)
    {
        shs::renderpath::RenderPathRecipe recipe{};
        recipe.name = name;
        recipe.backend = shs::RenderBackendType::Software;
        recipe.render_technique = shs::RenderPathRenderingTechnique::ForwardLit;
        recipe.technique_mode = shs::TechniqueMode::Forward;
        recipe.pass_chain = {
            shs::renderpath::make_render_path_pass_entry(shs::PassId::ShadowMap, true),
            shs::renderpath::make_render_path_pass_entry(shs::PassId::PBRForward, true),
            shs::renderpath::make_render_path_pass_entry(shs::PassId::Tonemap, true)
        };
        return recipe;
    }

#if defined(SHS_CONTRACTS_ENFORCED)
    shs::core::contract_kind g_kind{};
    std::string g_expr;
    int g_calls = 0;

    void capture_handler(shs::core::contract_kind kind, const char* expr, const char*, int) noexcept
    {
        ++g_calls;
        g_kind = kind;
        g_expr = expr;
    }
#endif

#if defined(SHS_CONTRACTS_ENFORCED)
    // C2.2 negative: a hand-mismatched technique-mode pair is an illegal
    // transition-table row — the compiler assertion fires (kind=assertion)
    // before any plan is returned.
    bool test_transition_table_rejects_mismatched_pair()
    {
        shs::renderpath::RenderPathRecipe recipe = make_forward_recipe("mismatched");
        recipe.render_technique = shs::RenderPathRenderingTechnique::Deferred; // illegal row
        const shs::renderpath::RenderPathCompiler compiler{};
        const auto caps = make_sw_caps();

        const int calls_before = g_calls;
        const auto compiled = compiler.try_compile(recipe, caps);
        (void)compiled;
        return g_calls == calls_before + 1
            && g_kind == shs::core::contract_kind::assertion
            && g_expr.find("render_path_rendering_technique_for_mode") != std::string::npos;
    }

    // C2.3 negative: the commands span and the events buffer share storage —
    // the Rule 7.1 wait-free rim precondition fires with kind=pre.
    bool test_rim_rejects_aliased_spans()
    {
        std::pmr::monotonic_buffer_resource arena{4096};
        std::pmr::vector<shs::renderpath::RenderPathEvent> events{&arena};
        events.reserve(4);
        const auto* aliased =
            reinterpret_cast<const shs::renderpath::RenderPathCommand*>(events.data());
        const std::span<const shs::renderpath::RenderPathCommand> commands{aliased, 0};

        shs::renderpath::RenderPathPodState state{};
        const shs::renderpath::RenderPathCompiler compiler{};
        const auto caps = make_sw_caps();

        const int calls_before = g_calls;
        (void)shs::renderpath::renderpath_gateway(state, commands, compiler, caps, events);
        return g_calls == calls_before + 1
            && g_kind == shs::core::contract_kind::pre
            && g_expr.find("events.data()") != std::string::npos;
    }

    // C2.1 negative: a hand-broken plan (compatibility rules with the empty-
    // chain rejection disabled commit an empty chain) trips the commit-rim
    // postcondition with kind=post, before PATH_COMPILED is emitted.
    bool test_commit_rim_rejects_hand_broken_plan()
    {
        shs::renderpath::RenderPathCompatibilityRules rules{};
        rules.reject_empty_pass_chain = false; // the hand-break
        rules.require_shadow_map_pass_when_shadows_enabled = false;
        shs::renderpath::RenderPathCompiler compiler{};
        compiler.set_rules(rules);
        shs::renderpath::RenderPathPodState state{};
        const auto caps = make_sw_caps();

        shs::renderpath::SelectPathPresetIntent intent{ make_forward_recipe("hand_broken") };
        intent.recipe.pass_chain.clear(); // the hand-break: empty committed plan
        intent.recipe.wants_shadows = false; // keep the only violation the empty chain
        std::pmr::monotonic_buffer_resource arena{4096};
        std::pmr::vector<shs::renderpath::RenderPathEvent> events{&arena};
        const std::vector<shs::renderpath::RenderPathCommand> commands{ intent };

        const int calls_before = g_calls;
        (void)shs::renderpath::renderpath_gateway(
            state, std::span<const shs::renderpath::RenderPathCommand>{commands}, compiler, caps, events);
        return g_calls == calls_before + 1
            && g_kind == shs::core::contract_kind::post
            && g_expr.find("pass_chain") != std::string::npos;
    }
#endif // SHS_CONTRACTS_ENFORCED
} // namespace

int main()
{
#if defined(SHS_CONTRACTS_ENFORCED)
    shs::core::set_contract_violation_handler(&capture_handler);
    bool ok = true;
    ok = test_transition_table_rejects_mismatched_pair() && ok;
    ok = test_rim_rejects_aliased_spans() && ok;
    ok = test_commit_rim_rejects_hand_broken_plan() && ok;
    shs::core::set_contract_violation_handler(nullptr); // restore the default
    if (!ok)
    {
        std::fprintf(stderr, "[contract-pilot-tests] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[contract-pilot-tests] all pilot negative tests passed (enforced)\n");
#else
    // Release semantics: house input compiles through the annotated seams with
    // checks discarded (assume path) and behaves identically (Rule 4.1).
    shs::renderpath::RenderPathPodState state{};
    const shs::renderpath::RenderPathCompiler compiler{};
    const auto caps = make_sw_caps();

    shs::renderpath::SelectPathPresetIntent intent{ make_forward_recipe("release_pilot") };
    const std::vector<shs::renderpath::RenderPathCommand> commands{ intent };
    std::pmr::monotonic_buffer_resource arena{4096};
    std::pmr::vector<shs::renderpath::RenderPathEvent> events{&arena};

    const auto step = shs::renderpath::renderpath_gateway(
        state, std::span<const shs::renderpath::RenderPathCommand>{commands}, compiler, caps, events);
    if (step.commands_applied != 1 || state.plan.pass_chain.empty())
    {
        std::fprintf(stderr, "[contract-pilot-tests-release] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[contract-pilot-tests-release] all pilot tests passed\n");
#endif
    return 0;
}
