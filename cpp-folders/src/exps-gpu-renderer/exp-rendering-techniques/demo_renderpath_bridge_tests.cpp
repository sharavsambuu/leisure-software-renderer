#include <cstdint>
#include <cstdio>
#include <variant>
#include <vector>

#include "demo_renderpath_bridge.hpp"

namespace
{
    using shs::renderpath::PathCompiledEvent;
    using shs::renderpath::PathSwapRejectedEvent;
    using shs::renderpath::TechniqueSwitchedEvent;

    bool test_mode_technique_round_trip()
    {
        if (shs::demo::mode_for_technique(shs::demo::technique_for_mode(shs::TechniqueMode::Forward)) !=
            shs::TechniqueMode::Forward) return false;
        if (shs::demo::mode_for_technique(shs::demo::technique_for_mode(shs::TechniqueMode::ForwardPlus)) !=
            shs::TechniqueMode::ForwardPlus) return false;
        if (shs::demo::mode_for_technique(shs::demo::technique_for_mode(shs::TechniqueMode::Deferred)) !=
            shs::TechniqueMode::Deferred) return false;
        return true;
    }

    bool test_technique_mode_cycle()
    {
        if (shs::demo::next_demo_technique_mode(shs::TechniqueMode::Forward) != shs::TechniqueMode::ForwardPlus)
            return false;
        if (shs::demo::next_demo_technique_mode(shs::TechniqueMode::ForwardPlus) != shs::TechniqueMode::Deferred)
            return false;
        if (shs::demo::next_demo_technique_mode(shs::TechniqueMode::Deferred) != shs::TechniqueMode::Forward)
            return false;
        // Extended demo modes fold back into the closed pod cycle.
        if (shs::demo::next_demo_technique_mode(shs::TechniqueMode::TiledDeferred) != shs::TechniqueMode::Forward)
            return false;
        return true;
    }

    bool test_action_to_intent()
    {
        auto intent = shs::demo::map_action_to_renderpath_command(
            shs::demo::DemoInputAction::CycleRenderingTechnique, shs::TechniqueMode::Forward);
        if (!intent) return false;
        const auto* swap = std::get_if<shs::renderpath::SetRenderingTechniqueIntent>(&*intent);
        if (!swap || swap->technique != shs::renderpath::RenderPathRenderingTechnique::ForwardPlus)
            return false;

        auto wrap = shs::demo::map_action_to_renderpath_command(
            shs::demo::DemoInputAction::CycleRenderingTechnique, shs::TechniqueMode::Deferred);
        if (!wrap) return false;
        const auto* to_forward = std::get_if<shs::renderpath::SetRenderingTechniqueIntent>(&*wrap);
        if (!to_forward ||
            to_forward->technique != shs::renderpath::RenderPathRenderingTechnique::ForwardLit)
            return false;

        if (shs::demo::map_action_to_renderpath_command(
                shs::demo::DemoInputAction::Quit, shs::TechniqueMode::Forward))
            return false;
        if (shs::demo::map_action_to_renderpath_command(
                shs::demo::DemoInputAction::LightOrbitScaleInc, shs::TechniqueMode::Forward))
            return false;
        return true;
    }

    shs::RenderPathRecipe make_demo_recipe(const char* name, shs::renderpath::RenderPathRenderingTechnique tech)
    {
        shs::RenderPathRecipe recipe{};
        recipe.name = name;
        recipe.backend = shs::RenderBackendType::Vulkan;
        recipe.render_technique = tech;
        recipe.technique_mode = shs::demo::mode_for_technique(tech);
        recipe.pass_chain = {
            shs::make_render_path_pass_entry(shs::PassId::ShadowMap, true),
            shs::make_render_path_pass_entry(shs::PassId::PBRForwardPlus, true),
            shs::make_render_path_pass_entry(shs::PassId::Tonemap, true)};
        return recipe;
    }

    // End-to-end (still GPU-free): intents -> pure reducer -> events.
    bool test_reducer_accepts_and_hot_swaps()
    {
        shs::RenderPathCompiler compiler{};
        const shs::RenderPathCapabilitySet caps = shs::make_render_path_capability_set(
            shs::RenderBackendType::Vulkan, shs::BackendCapabilities{});
        shs::renderpath::RenderPathPodState state{};
        std::pmr::vector<shs::renderpath::RenderPathEvent> events{};

        const shs::renderpath::RenderPathCommand install[] = {
            shs::renderpath::SelectPathPresetIntent{
                make_demo_recipe("demo_forward", shs::renderpath::RenderPathRenderingTechnique::ForwardPlus)}};
        shs::renderpath::reduce_render_path(state, install, compiler, caps, events);
        if (events.size() != 1 || !std::holds_alternative<PathCompiledEvent>(events[0])) return false;
        if (!state.has_plan || state.recipe.render_technique !=
                                   shs::renderpath::RenderPathRenderingTechnique::ForwardPlus) return false;

        // Same technique: no-op, no events.
        const shs::renderpath::RenderPathCommand same[] = {
            shs::renderpath::SetRenderingTechniqueIntent{
                shs::renderpath::RenderPathRenderingTechnique::ForwardPlus}};
        events.clear();
        shs::renderpath::reduce_render_path(state, same, compiler, caps, events);
        if (!events.empty()) return false;

        // Different technique: hot-swap accepted, recipe updated, events emitted.
        const shs::renderpath::RenderPathCommand swap[] = {
            shs::renderpath::SetRenderingTechniqueIntent{
                shs::renderpath::RenderPathRenderingTechnique::Deferred}};
        events.clear();
        shs::renderpath::reduce_render_path(state, swap, compiler, caps, events);
        if (events.size() != 2) return false; // PATH_COMPILED + TECHNIQUE_SWITCHED
        if (!std::holds_alternative<TechniqueSwitchedEvent>(events[1])) return false;
        if (state.recipe.render_technique != shs::renderpath::RenderPathRenderingTechnique::Deferred)
            return false;

        // Invalid candidate: rejection keeps the previous plan.
        shs::RenderPathRecipe broken =
            make_demo_recipe("broken", shs::renderpath::RenderPathRenderingTechnique::Deferred);
        broken.pass_chain.clear();
        const shs::renderpath::RenderPathCommand bad[] = {
            shs::renderpath::SelectPathPresetIntent{broken}};
        events.clear();
        shs::renderpath::reduce_render_path(state, bad, compiler, caps, events);
        if (events.size() != 1 || !std::holds_alternative<PathSwapRejectedEvent>(events[0])) return false;
        if (state.recipe.render_technique != shs::renderpath::RenderPathRenderingTechnique::Deferred)
            return false;
        return true;
    }
} // namespace

int main()
{
    struct NamedCheck
    {
        const char* name;
        bool (*fn)();
    };
    const NamedCheck checks[] = {
        {"mode_technique_round_trip", test_mode_technique_round_trip},
        {"technique_mode_cycle", test_technique_mode_cycle},
        {"action_to_intent", test_action_to_intent},
        {"reducer_accepts_and_hot_swaps", test_reducer_accepts_and_hot_swaps},
    };
    bool all_pass = true;
    for (const NamedCheck& check : checks)
    {
        const bool ok = check.fn();
        std::fprintf(stderr, "[%s] %s\n", ok ? "PASS" : "FAIL", check.name);
        all_pass = all_pass && ok;
    }
    if (!all_pass)
    {
        std::fprintf(stderr, "shs_demo_renderpath_bridge_tests: FAILED\n");
        return 1;
    }
    std::fprintf(
        stderr,
        "shs_demo_renderpath_bridge_tests: all %zu checks passed\n",
        sizeof(checks) / sizeof(checks[0]));
    return 0;
}
