#include <cstdio>
#include <memory_resource>
#include <string>
#include <variant>
#include <vector>

#include "shs/domains/renderpath/renderpath.gateway.hpp"

// Pure value tests for the renderpath pod (roadmap P1 ctest gate:
// shs_renderer_vop_renderpath_*). Zero Vulkan/SDL links: the test binary links
// only the header-only shs::core-values INTERFACE target, and the gateway
// runs with no Context, no backend instances, and no pass registry.
namespace
{
    // Software-backend capability snapshot as a pure value (no Context needed).
    // Mirrors the software stack: occlusion culling is supported via the
    // software depth-cull path.
    shs::RenderPathCapabilitySet make_sw_caps()
    {
        shs::BackendCapabilities backend_caps{};
        shs::RenderPathCapabilitySet caps =
            shs::make_render_path_capability_set(shs::RenderBackendType::Software, backend_caps);
        return caps;
    }

    // Valid forward-lit software recipe: shadows on, frustum view culling.
    shs::RenderPathRecipe make_forward_recipe(const char* name)
    {
        shs::RenderPathRecipe recipe{};
        recipe.name = name;
        recipe.backend = shs::RenderBackendType::Software;
        recipe.render_technique = shs::RenderPathRenderingTechnique::ForwardLit;
        recipe.technique_mode = shs::TechniqueMode::Forward;
        recipe.view_culling = shs::RenderPathCullingMode::Frustum;
        recipe.pass_chain = {
            shs::make_render_path_pass_entry(shs::PassId::ShadowMap, true),
            shs::make_render_path_pass_entry(shs::PassId::PBRForward, true),
            shs::make_render_path_pass_entry(shs::PassId::Tonemap, true)
        };
        return recipe;
    }

    // Occlusion-capable recipe: view culling requires occlusion, so a
    // depth_prepass pass is mandatory under the default compatibility rules.
    shs::RenderPathRecipe make_occlusion_recipe(const char* name)
    {
        shs::RenderPathRecipe recipe = make_forward_recipe(name);
        recipe.view_culling = shs::RenderPathCullingMode::FrustumAndOcclusion;
        recipe.pass_chain.insert(
            recipe.pass_chain.begin() + 1,
            shs::make_render_path_pass_entry(shs::PassId::DepthPrepass, true));
        return recipe;
    }

    // --- tests --------------------------------------------------------------

    // Path selection: preset select compiles and becomes the active plan.
    bool test_path_selection()
    {
        shs::RenderPathCompiler compiler{};
        const shs::RenderPathCapabilitySet caps = make_sw_caps();

        shs::renderpath::RenderPathPodState state{};
        std::pmr::monotonic_buffer_resource arena{ 4096 };
        std::pmr::vector<shs::renderpath::RenderPathEvent> events{ &arena };

        std::vector<shs::renderpath::RenderPathCommand> commands{
            shs::renderpath::SelectPathPresetIntent{ make_forward_recipe("forward_sw") }
        };
        shs::renderpath::renderpath_gateway(state, commands, compiler, caps, events);

        if (!state.has_plan) return false;
        if (!state.plan.valid) return false;
        if (state.plan.recipe_name != "forward_sw") return false;
        if (state.plan.technique_mode != shs::TechniqueMode::Forward) return false;
        if (state.plan.runtime_state.view_occlusion_enabled != state.recipe.runtime_defaults.view_occlusion_enabled) return false;
        if (state.plan.runtime_state.lit_mode != state.recipe.runtime_defaults.lit_mode) return false;
        if (state.plan.runtime_state.enable_shadows != state.recipe.runtime_defaults.enable_shadows) return false;
        if (events.size() != 1) return false;
        const auto* compiled = std::get_if<shs::renderpath::PathCompiledEvent>(&events[0]);
        if (!compiled) return false;
        if (compiled->pass_count != 3) return false;
        return true;
    }

    // Technique switching: hot-swap recompiles and flips technique_mode.
    bool test_technique_switching()
    {
        shs::RenderPathCompiler compiler{};
        const shs::RenderPathCapabilitySet caps = make_sw_caps();

        shs::renderpath::RenderPathPodState state{};
        state.recipe = make_forward_recipe("forward_sw");
        state.plan = compiler.compile(state.recipe, caps);
        state.has_plan = state.plan.valid;
        if (!state.has_plan) return false;

        std::pmr::monotonic_buffer_resource arena{ 4096 };
        std::pmr::vector<shs::renderpath::RenderPathEvent> events{ &arena };

        std::vector<shs::renderpath::RenderPathCommand> commands{
            shs::renderpath::SetRenderingTechniqueIntent{ shs::RenderPathRenderingTechnique::ForwardPlus }
        };
        shs::renderpath::renderpath_gateway(state, commands, compiler, caps, events);

        if (state.plan.render_technique != shs::RenderPathRenderingTechnique::ForwardPlus) return false;
        if (state.plan.technique_mode != shs::TechniqueMode::ForwardPlus) return false;
        if (!state.plan.valid) return false;

        // PathCompiled (swap accepted) + TechniqueSwitched (raw fact).
        if (events.size() != 2) return false;
        const auto* switched = std::get_if<shs::renderpath::TechniqueSwitchedEvent>(&events[1]);
        if (!switched) return false;
        if (switched->previous != shs::RenderPathRenderingTechnique::ForwardLit) return false;
        if (switched->current != shs::RenderPathRenderingTechnique::ForwardPlus) return false;
        return true;
    }

    // Culling-mode changes: accepted when capabilities allow, rejected when
    // they do not (previous plan surviving the rejection).
    bool test_culling_mode_changes()
    {
        shs::RenderPathCompiler compiler{};

        // (a) Accepted: occlusion-capable caps + depth_prepass in the chain.
        shs::BackendCapabilities occlusion_backend_caps{};
        shs::RenderPathCapabilitySet occlusion_caps = shs::make_render_path_capability_set(
            shs::RenderBackendType::Software, occlusion_backend_caps);
        occlusion_caps.supports_occlusion_query = true;

        shs::renderpath::RenderPathPodState state{};
        state.recipe = make_occlusion_recipe("occlusion_sw");
        state.plan = compiler.compile(state.recipe, occlusion_caps);
        state.has_plan = state.plan.valid;
        if (!state.has_plan) return false;

        std::pmr::monotonic_buffer_resource arena_a{ 4096 };
        std::pmr::vector<shs::renderpath::RenderPathEvent> events_a{ &arena_a };
        std::vector<shs::renderpath::RenderPathCommand> relax{
            shs::renderpath::SetViewCullingModeIntent{ shs::RenderPathCullingMode::Frustum }
        };
        shs::renderpath::renderpath_gateway(state, relax, compiler, occlusion_caps, events_a);

        if (state.recipe.view_culling != shs::RenderPathCullingMode::Frustum) return false;
        if (events_a.size() != 2) return false; // PathCompiled + CullingModeChanged
        const auto* changed = std::get_if<shs::renderpath::CullingModeChangedEvent>(&events_a[1]);
        if (!changed) return false;
        if (!changed->view_chain) return false;
        if (changed->current != shs::RenderPathCullingMode::Frustum) return false;
        return true;
    }

    // Rejection behavior: invalid compile keeps previous plan + PATH_SWAP_REJECTED.
    bool test_rejection_keeps_previous_plan()
    {
        shs::RenderPathCompiler compiler{};
        const shs::RenderPathCapabilitySet caps = make_sw_caps();

        shs::renderpath::RenderPathPodState state{};
        state.recipe = make_forward_recipe("forward_sw");
        state.plan = compiler.compile(state.recipe, caps);
        state.has_plan = state.plan.valid;
        if (!state.has_plan) return false;
        const shs::RenderPathExecutionPlan previous_plan = state.plan;

        // Invalid candidate: empty pass chain must fail compilation.
        shs::RenderPathRecipe broken = make_forward_recipe("broken_sw");
        broken.pass_chain.clear();

        std::pmr::monotonic_buffer_resource arena{ 4096 };
        std::pmr::vector<shs::renderpath::RenderPathEvent> events{ &arena };
        std::vector<shs::renderpath::RenderPathCommand> commands{
            shs::renderpath::SelectPathPresetIntent{ broken }
        };
        shs::renderpath::renderpath_gateway(state, commands, compiler, caps, events);

        // Previous plan kept; recipe unchanged.
        if (!state.has_plan) return false;
        if (state.plan.recipe_name != previous_plan.recipe_name) return false;
        if (state.plan.technique_mode != previous_plan.technique_mode) return false;
        if (state.plan.pass_chain.size() != previous_plan.pass_chain.size()) return false;
        if (state.recipe.name != "forward_sw") return false;

        if (events.size() != 1) return false;
        const auto* rejected = std::get_if<shs::renderpath::PathSwapRejectedEvent>(&events[0]);
        if (!rejected) return false;
        if (rejected->reason != shs::renderpath::PathSwapRejectionReason::EmptyPassChain) return false;
        return true;
    }

    // Runtime toggles mutate runtime state; never trigger a recompile.
    bool test_runtime_toggles()
    {
        shs::RenderPathCompiler compiler{};
        const shs::RenderPathCapabilitySet caps = make_sw_caps();

        shs::renderpath::RenderPathPodState state{};
        state.recipe = make_forward_recipe("forward_sw");
        state.plan = compiler.compile(state.recipe, caps);
        state.has_plan = state.plan.valid;
        if (!state.has_plan) return false;

        std::pmr::monotonic_buffer_resource arena{ 4096 };
        std::pmr::vector<shs::renderpath::RenderPathEvent> events{ &arena };
        std::vector<shs::renderpath::RenderPathCommand> commands{
            shs::renderpath::SetRuntimeToggleIntent{ shs::renderpath::RuntimeToggle::DebugAabb, true },
            shs::renderpath::SetRuntimeToggleIntent{ shs::renderpath::RuntimeToggle::Shadows, false }
        };
        shs::renderpath::renderpath_gateway(state, commands, compiler, caps, events);

        if (state.plan.runtime_state.debug_aabb != true) return false;
        if (state.recipe.runtime_defaults.debug_aabb != true) return false;
        if (state.plan.runtime_state.enable_shadows != false) return false;
        if (state.recipe.runtime_defaults.enable_shadows != false) return false;
        if (events.size() != 2) return false;
        const auto* toggled = std::get_if<shs::renderpath::RuntimeToggledEvent>(&events[1]);
        if (!toggled) return false;
        if (toggled->toggle != shs::renderpath::RuntimeToggle::Shadows) return false;
        if (toggled->enabled) return false;

        // Toggle before any plan exists: recipe defaults update, no crash.
        shs::renderpath::RenderPathPodState fresh{};
        std::pmr::monotonic_buffer_resource arena2{ 4096 };
        std::pmr::vector<shs::renderpath::RenderPathEvent> events2{ &arena2 };
        std::vector<shs::renderpath::RenderPathCommand> pre_plan{
            shs::renderpath::SetRuntimeToggleIntent{ shs::renderpath::RuntimeToggle::LitMode, false }
        };
        shs::renderpath::renderpath_gateway(state, pre_plan, compiler, caps, events);
        if (state.recipe.runtime_defaults.lit_mode != false) return false;
        if (events.size() != 3) return false;
        return true;
    }
} // namespace

int main()
{
    bool ok = true;

    ok = test_path_selection() && ok;
    ok = test_technique_switching() && ok;
    ok = test_culling_mode_changes() && ok;
    ok = test_rejection_keeps_previous_plan() && ok;
    ok = test_runtime_toggles() && ok;

    if (!ok)
    {
        std::fprintf(stderr, "[renderpath-tests] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[renderpath-tests] all tests passed\n");
    return 0;
}
