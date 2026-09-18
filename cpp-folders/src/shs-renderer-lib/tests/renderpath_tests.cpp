#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory_resource>
#include <string>
#include <variant>
#include <vector>

#include "shs/core/testing/pod_test_kit.hpp"
#include "shs/renderpath/execution/render_path_registry.hpp"
#include "shs/renderpath/planning/frame_graph.hpp"
#include "shs/renderpath/renderpath.gateway.hpp"

// Pure value tests for the renderpath pod (roadmap P1 ctest gate:
// shs_renderer_vop_renderpath_*). Zero Vulkan/SDL links: the test binary links
// only the header-only shs::core-values INTERFACE target, and the gateway
// runs with no Context, no backend instances, and no pass registry.
namespace
{
    // Software-backend capability snapshot as a pure value (no Context needed).
    // Mirrors the software stack: occlusion culling is supported via the
    // software depth-cull path.
    shs::renderpath::RenderPathCapabilitySet make_sw_caps()
    {
        shs::rhi::BackendCapabilities backend_caps{};
        shs::renderpath::RenderPathCapabilitySet caps =
            shs::renderpath::make_render_path_capability_set(shs::RenderBackendType::Software, backend_caps);
        return caps;
    }

    // Valid forward-lit software recipe: shadows on, frustum view culling.
    shs::renderpath::RenderPathRecipe make_forward_recipe(const char* name)
    {
        shs::renderpath::RenderPathRecipe recipe{};
        recipe.name = name;
        recipe.backend = shs::RenderBackendType::Software;
        recipe.render_technique = shs::RenderPathRenderingTechnique::ForwardLit;
        recipe.technique_mode = shs::TechniqueMode::Forward;
        recipe.view_culling = shs::RenderPathCullingMode::Frustum;
        recipe.pass_chain = {
            shs::renderpath::make_render_path_pass_entry(shs::PassId::ShadowMap, true),
            shs::renderpath::make_render_path_pass_entry(shs::PassId::PBRForward, true),
            shs::renderpath::make_render_path_pass_entry(shs::PassId::Tonemap, true)
        };
        return recipe;
    }

    // Occlusion-capable recipe: view culling requires occlusion, so a
    // depth_prepass pass is mandatory under the default compatibility rules.
    shs::renderpath::RenderPathRecipe make_occlusion_recipe(const char* name)
    {
        shs::renderpath::RenderPathRecipe recipe = make_forward_recipe(name);
        recipe.view_culling = shs::RenderPathCullingMode::FrustumAndOcclusion;
        recipe.pass_chain.insert(
            recipe.pass_chain.begin() + 1,
            shs::renderpath::make_render_path_pass_entry(shs::PassId::DepthPrepass, true));
        return recipe;
    }

    // --- tests --------------------------------------------------------------

    // Path selection: preset select compiles and becomes the active plan.
    bool test_path_selection()
    {
        shs::renderpath::RenderPathCompiler compiler{};
        const shs::renderpath::RenderPathCapabilitySet caps = make_sw_caps();

        shs::renderpath::RenderPathPodState state{};
        std::pmr::monotonic_buffer_resource arena{ 4096 };
        std::pmr::vector<shs::renderpath::RenderPathEvent> events{ &arena };

        std::vector<shs::renderpath::RenderPathCommand> commands{
            shs::renderpath::SelectPathPresetIntent{ make_forward_recipe("forward_sw") }
        };
        shs::renderpath::renderpath_gateway(state, commands, compiler, caps, events);

        if (state.plan_generation != 1) return false; // K4.1: first install bumps 0 -> 1
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
        shs::renderpath::RenderPathCompiler compiler{};
        const shs::renderpath::RenderPathCapabilitySet caps = make_sw_caps();

        shs::renderpath::RenderPathPodState state{};
        state.recipe = make_forward_recipe("forward_sw");
        state.plan = compiler.compile(state.recipe, caps);
        state.plan_generation = state.plan.valid ? 1u : 0u;
        if (state.plan_generation == 0) return false;

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
        shs::renderpath::RenderPathCompiler compiler{};

        // (a) Accepted: occlusion-capable caps + depth_prepass in the chain.
        shs::rhi::BackendCapabilities occlusion_backend_caps{};
        shs::renderpath::RenderPathCapabilitySet occlusion_caps = shs::renderpath::make_render_path_capability_set(
            shs::RenderBackendType::Software, occlusion_backend_caps);
        occlusion_caps.supports_occlusion_query = true;

        shs::renderpath::RenderPathPodState state{};
        state.recipe = make_occlusion_recipe("occlusion_sw");
        state.plan = compiler.compile(state.recipe, occlusion_caps);
        state.plan_generation = state.plan.valid ? 1u : 0u;
        if (state.plan_generation == 0) return false;

        std::pmr::monotonic_buffer_resource arena_a{ 4096 };
        std::pmr::vector<shs::renderpath::RenderPathEvent> events_a{ &arena_a };
        std::vector<shs::renderpath::RenderPathCommand> relax{
            shs::renderpath::SetViewCullingModeIntent{ shs::RenderPathCullingMode::Frustum }
        };
        shs::renderpath::renderpath_gateway(state, relax, compiler, occlusion_caps, events_a);

        if (state.recipe.view_culling != shs::RenderPathCullingMode::Frustum) return false;
        if (events_a.size() != 2) return false; // PathCompiled + ViewCullingModeChanged
        const auto* changed = std::get_if<shs::renderpath::ViewCullingModeChangedEvent>(&events_a[1]);
        if (!changed) return false;
        if (changed->previous != shs::RenderPathCullingMode::FrustumAndOcclusion) return false;
        if (changed->current != shs::RenderPathCullingMode::Frustum) return false;
        return true;
    }

    // Rejection behavior: invalid compile keeps previous plan + PATH_SWAP_REJECTED.
    bool test_rejection_keeps_previous_plan()
    {
        shs::renderpath::RenderPathCompiler compiler{};
        const shs::renderpath::RenderPathCapabilitySet caps = make_sw_caps();

        shs::renderpath::RenderPathPodState state{};
        state.recipe = make_forward_recipe("forward_sw");
        state.plan = compiler.compile(state.recipe, caps);
        state.plan_generation = state.plan.valid ? 1u : 0u;
        if (state.plan_generation == 0) return false;
        const shs::renderpath::RenderPathExecutionPlan previous_plan = state.plan;

        // Invalid candidate: empty pass chain must fail compilation.
        shs::renderpath::RenderPathRecipe broken = make_forward_recipe("broken_sw");
        broken.pass_chain.clear();

        std::pmr::monotonic_buffer_resource arena{ 4096 };
        std::pmr::vector<shs::renderpath::RenderPathEvent> events{ &arena };
        std::vector<shs::renderpath::RenderPathCommand> commands{
            shs::renderpath::SelectPathPresetIntent{ broken }
        };
        shs::renderpath::renderpath_gateway(state, commands, compiler, caps, events);

        // Previous plan kept; recipe unchanged.
        if (state.plan_generation == 0) return false;
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

    // A rejected culling change must not announce a mutation that never happened.
    bool test_rejected_view_culling_has_no_changed_fact()
    {
        shs::renderpath::RenderPathCompiler compiler{};
        auto caps = make_sw_caps();
        caps.supports_occlusion_query = false;
        shs::renderpath::RenderPathPodState state{};
        state.recipe = make_forward_recipe("forward_sw");
        state.plan = compiler.compile(state.recipe, caps);
        if (!state.plan.valid) return false;
        state.plan_generation = 1;
        const auto previous = state;

        std::pmr::monotonic_buffer_resource arena{4096};
        std::pmr::vector<shs::renderpath::RenderPathEvent> events{&arena};
        const std::vector<shs::renderpath::RenderPathCommand> commands{
            shs::renderpath::SetViewCullingModeIntent{
                shs::RenderPathCullingMode::FrustumAndOcclusion}
        };
        const auto step = shs::renderpath::renderpath_gateway(
            state, commands, compiler, caps, events);
        if (state != previous) return false;
        if (step != shs::renderpath::RenderPathStep{0, 0, 1, 1}) return false;
        if (events.size() != 1)
        {
            std::fprintf(stderr, "[renderpath-tests] rejected view culling emitted %zu facts; expected only rejection\n",
                events.size());
            return false;
        }
        const auto* rejected = std::get_if<shs::renderpath::PathSwapRejectedEvent>(&events.front());
        return rejected && rejected->reason == shs::renderpath::PathSwapRejectionReason::OcclusionUnsupported;
    }

    // Every swap handler must preserve the caller's prefix and reject without
    // a false change fact. A later accepted retry must still install normally.
    bool test_rejected_swap_facts_and_recovery()
    {
        using namespace shs::renderpath;
        shs::renderpath::RenderPathCompiler compiler{};
        const auto caps = make_sw_caps();
        auto unavailable = caps;
        unavailable.has_backend = false;
        RenderPathPodState initial{};
        initial.recipe = make_forward_recipe("forward_sw");
        initial.plan = compiler.compile(initial.recipe, caps);
        if (!initial.plan.valid) return false;
        initial.plan_generation = 1;
        const std::vector<RenderPathCommand> commands{
            SetRenderingTechniqueIntent{shs::RenderPathRenderingTechnique::ForwardPlus},
            SetViewCullingModeIntent{shs::RenderPathCullingMode::None},
            SetShadowCullingModeIntent{shs::RenderPathCullingMode::None}
        };
        for (const auto& command : commands)
        {
            auto state = initial;
            auto replay = initial;
            std::pmr::monotonic_buffer_resource arena{4096};
            std::pmr::vector<RenderPathEvent> events{&arena};
            events.push_back(PathSwapRejectedEvent{PathSwapRejectionReason::CompileInvalid});
            auto replay_events = events;
            const std::span<const RenderPathCommand> batch{&command, 1};
            const auto step = renderpath_gateway(state, batch, compiler, unavailable, events);
            if (state != initial || step != RenderPathStep{0, 0, 1, 1}) return false;
            if (events.size() != 2 || events.front() != replay_events.front()) return false;
            const auto* rejected = std::get_if<PathSwapRejectedEvent>(&events.back());
            if (!rejected || rejected->reason != PathSwapRejectionReason::BackendUnavailable) return false;
            if (renderpath_gateway(replay, batch, compiler, unavailable, replay_events) != step
                || replay != state || replay_events != events) return false;
            if (renderpath_gateway(state, {}, compiler, unavailable, events) != RenderPathStep{0, 0, 0, 1}
                || state != initial || events != replay_events) return false;
            const auto retry = renderpath_gateway(state, batch, compiler, caps, events);
            if (retry != RenderPathStep{1, 0, 0, 2} || events.size() != 4
                || !std::holds_alternative<PathCompiledEvent>(events[2])) return false;
            const bool changed = std::visit([&](const auto& cmd) {
                using T = std::decay_t<decltype(cmd)>;
                if constexpr (std::is_same_v<T, SetRenderingTechniqueIntent>)
                    return std::holds_alternative<TechniqueSwitchedEvent>(events.back());
                else if constexpr (std::is_same_v<T, SetViewCullingModeIntent>)
                    return std::holds_alternative<ViewCullingModeChangedEvent>(events.back());
                else
                    return std::holds_alternative<ShadowCullingModeChangedEvent>(events.back());
            }, command);
            if (!changed) return false;
        }
        return true;
    }

    // Runtime toggles mutate runtime state; never trigger a recompile.
    bool test_runtime_toggles()
    {
        shs::renderpath::RenderPathCompiler compiler{};
        const shs::renderpath::RenderPathCapabilitySet caps = make_sw_caps();

        shs::renderpath::RenderPathPodState state{};
        state.recipe = make_forward_recipe("forward_sw");
        state.plan = compiler.compile(state.recipe, caps);
        state.plan_generation = state.plan.valid ? 1u : 0u;
        if (state.plan_generation == 0) return false;

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

    // K3.2 (Run A): same-value commands emit *Unchanged facts, not silence.
    bool test_unchanged_facts()
    {
        shs::renderpath::RenderPathCompiler compiler{};
        const shs::renderpath::RenderPathCapabilitySet caps = make_sw_caps();

        shs::renderpath::RenderPathPodState state{};
        state.recipe = make_forward_recipe("forward_sw");
        state.plan = compiler.compile(state.recipe, caps);
        state.plan_generation = state.plan.valid ? 1u : 0u;
        if (state.plan_generation == 0) return false;

        std::pmr::monotonic_buffer_resource arena{ 4096 };
        std::pmr::vector<shs::renderpath::RenderPathEvent> events{ &arena };
        std::vector<shs::renderpath::RenderPathCommand> commands{
            shs::renderpath::SetRenderingTechniqueIntent{ shs::RenderPathRenderingTechnique::ForwardLit },
            shs::renderpath::SetViewCullingModeIntent{ shs::RenderPathCullingMode::Frustum },
            shs::renderpath::SetShadowCullingModeIntent{ shs::RenderPathCullingMode::FrustumAndOptionalOcclusion }
        };
        const shs::renderpath::RenderPathStep step =
            shs::renderpath::renderpath_gateway(state, commands, compiler, caps, events);

        if (step.noops_observed != 3) return false;
        if (step.commands_applied != 0) return false;
        if (step.swaps_rejected != 0) return false;
        if (events.size() != 3) return false;
        if (!std::holds_alternative<shs::renderpath::TechniqueUnchangedEvent>(events[0])) return false;
        if (!std::holds_alternative<shs::renderpath::ViewCullingUnchangedEvent>(events[1])) return false;
        if (!std::holds_alternative<shs::renderpath::ShadowCullingUnchangedEvent>(events[2])) return false;
        // No-op commands must not bump the plan generation.
        if (state.plan_generation != 1) return false;
        return true;
    }

    // K4.1 (Run A): plan_generation semantics — 0 = none, bumped on every
    // successful install, untouched by rejections and runtime toggles.
    bool test_plan_generation_semantics()
    {
        shs::renderpath::RenderPathCompiler compiler{};
        const shs::renderpath::RenderPathCapabilitySet caps = make_sw_caps();

        shs::renderpath::RenderPathPodState state{};
        if (state.plan_generation != 0) return false; // 0 = no plan ever installed

        std::pmr::monotonic_buffer_resource arena{ 4096 };
        std::pmr::vector<shs::renderpath::RenderPathEvent> events{ &arena };

        shs::renderpath::RenderPathRecipe broken = make_forward_recipe("broken_sw");
        broken.pass_chain.clear();

        std::vector<shs::renderpath::RenderPathCommand> commands{
            shs::renderpath::SelectPathPresetIntent{ broken },                                       // rejected
            shs::renderpath::SelectPathPresetIntent{ make_forward_recipe("first") },                 // gen 1
            shs::renderpath::SetRuntimeToggleIntent{ shs::renderpath::RuntimeToggle::Shadows, false }, // no bump
            shs::renderpath::SetRenderingTechniqueIntent{ shs::RenderPathRenderingTechnique::ForwardPlus } // gen 2
        };
        const shs::renderpath::RenderPathStep step =
            shs::renderpath::renderpath_gateway(state, commands, compiler, caps, events);

        if (step.swaps_rejected != 1) return false;
        if (step.commands_applied != 3) return false; // select + toggle + technique swap
        if (state.plan_generation != 2) return false;
        if (step.plan_generation != 2) return false;
        if (!state.plan.valid) return false;
        if (state.plan.render_technique != shs::RenderPathRenderingTechnique::ForwardPlus) return false;
        return true;
    }
    // K1.1 DoD: pod test kit — the same command log applied twice yields
    // identical state + identical event logs (replay determinism over the
    // Kleisli house shape; value semantics pinned by RenderPathPodState ==).
    bool test_kit_replay_deterministic()
    {
        struct GatewayContext
        {
            shs::renderpath::RenderPathCompiler compiler{};
            shs::renderpath::RenderPathCapabilitySet caps{};
        };

        shs::renderpath::RenderPathCompiler compiler{};
        GatewayContext context{ compiler, make_sw_caps() };

        shs::renderpath::RenderPathPodState initial{};
        initial.recipe = make_forward_recipe("forward_sw");
        initial.plan = compiler.compile(initial.recipe, context.caps);
        initial.plan_generation = initial.plan.valid ? 1u : 0u;
        if (initial.plan_generation == 0) return false;

        std::vector<shs::renderpath::RenderPathCommand> commands{
            shs::renderpath::SetRuntimeToggleIntent{ shs::renderpath::RuntimeToggle::DebugAabb, true },
            shs::renderpath::SetRenderingTechniqueIntent{ shs::RenderPathRenderingTechnique::ForwardPlus },
            shs::renderpath::SetRenderingTechniqueIntent{ shs::RenderPathRenderingTechnique::ForwardPlus }, // noop fact
            shs::renderpath::SelectPathPresetIntent{ make_forward_recipe("forward_sw") }                    // re-select
        };

        return shs::pod_test::replay_is_deterministic<
            shs::renderpath::RenderPathPodState,
            shs::renderpath::RenderPathCommand,
            GatewayContext,
            shs::renderpath::RenderPathEvent
        >(
            [](shs::renderpath::RenderPathPodState& s,
               std::span<const shs::renderpath::RenderPathCommand> cmds,
               const GatewayContext& c,
               std::pmr::vector<shs::renderpath::RenderPathEvent>& ev) {
                shs::renderpath::renderpath_gateway(s, cmds, c.compiler, c.caps, ev);
            },
            initial, commands, context);
    }

    // K1.1 DoD: pod test kit — the empty command log leaves the state
    // bit-identical and emits nothing.
    bool test_kit_empty_log_stable()
    {
        struct GatewayContext
        {
            shs::renderpath::RenderPathCompiler compiler{};
            shs::renderpath::RenderPathCapabilitySet caps{};
        };

        shs::renderpath::RenderPathCompiler compiler{};
        GatewayContext context{ compiler, make_sw_caps() };

        shs::renderpath::RenderPathPodState initial{};
        initial.recipe = make_forward_recipe("forward_sw");
        initial.plan = compiler.compile(initial.recipe, context.caps);
        initial.plan_generation = initial.plan.valid ? 1u : 0u;
        if (initial.plan_generation == 0) return false;

        return shs::pod_test::empty_log_is_stable<
            shs::renderpath::RenderPathPodState,
            shs::renderpath::RenderPathCommand,
            GatewayContext,
            shs::renderpath::RenderPathEvent
        >(
            [](shs::renderpath::RenderPathPodState& s,
               std::span<const shs::renderpath::RenderPathCommand> cmds,
               const GatewayContext& c,
               std::pmr::vector<shs::renderpath::RenderPathEvent>& ev) {
                shs::renderpath::renderpath_gateway(s, cmds, c.compiler, c.caps, ev);
            },
            initial, context);
    }
    // --- RP-2: execution-unit × substrate axes + hybrid legality -------------
    //
    // Owner ruling 2026-09-18 made these two axes independent and made an
    // undeclared cross-unit crossing a *rejection* rather than a warning. These
    // gates pin both halves so the vocabulary cannot silently re-collapse.

    // One unit carries several substrates; the axes must not be identified.
    bool test_domain_axes_are_independent()
    {
        using shs::renderpath::ExecutionUnit;
        using shs::RenderBackendType;
        using shs::renderpath::Substrate;

        if (shs::renderpath::execution_unit_of(Substrate::SoftwareRaster) != ExecutionUnit::Host) return false;
        if (shs::renderpath::execution_unit_of(Substrate::OpenGL) != ExecutionUnit::Device) return false;
        if (shs::renderpath::execution_unit_of(Substrate::Vulkan) != ExecutionUnit::Device) return false;

        // RenderBackendType <-> Substrate is the declared 1:1 identity.
        if (shs::renderpath::substrate_of_backend(RenderBackendType::Software) != Substrate::SoftwareRaster) return false;
        if (shs::renderpath::substrate_of_backend(RenderBackendType::Vulkan) != Substrate::Vulkan) return false;
        if (shs::renderpath::backend_of_substrate(Substrate::OpenGL) != RenderBackendType::OpenGL) return false;
        return true;
    }

    // Compatibility falls out of the two axes instead of being special-cased.
    bool test_domain_compatibility_relation()
    {
        using shs::renderpath::Substrate;
        const auto device = shs::renderpath::render_domain_device();
        const auto host = shs::renderpath::render_domain_host();
        const auto any = shs::renderpath::render_domain_any();
        const auto unspecified = shs::renderpath::render_domain_unspecified();
        const auto gl = shs::renderpath::render_domain_substrate(Substrate::OpenGL);
        const auto vk = shs::renderpath::render_domain_substrate(Substrate::Vulkan);

        // Unpinned device resource accepts either device substrate...
        if (!shs::renderpath::render_domains_compatible(device, gl)) return false;
        if (!shs::renderpath::render_domains_compatible(device, vk)) return false;
        // ...but two distinct device substrates are not interchangeable.
        if (shs::renderpath::render_domains_compatible(gl, vk)) return false;
        // Host and device never mix implicitly (this is the hybrid rejection).
        if (shs::renderpath::render_domains_compatible(host, device)) return false;
        if (shs::renderpath::render_domains_compatible(host, gl)) return false;
        // "unspecified" (author said nothing) and "any" (author said wildcard)
        // are separate names, both permissive.
        if (!shs::renderpath::render_domains_compatible(unspecified, device)) return false;
        if (!shs::renderpath::render_domains_compatible(any, host)) return false;
        return true;
    }

    // Declared resource domain vs. the concrete backend a pass runs on.
    bool test_domain_backend_matching()
    {
        using shs::RenderBackendType;
        using shs::renderpath::Substrate;
        if (!shs::renderpath::render_domain_matches_backend(
                shs::renderpath::render_domain_host(), RenderBackendType::Software)) return false;
        if (shs::renderpath::render_domain_matches_backend(
                shs::renderpath::render_domain_host(), RenderBackendType::Vulkan)) return false;
        if (!shs::renderpath::render_domain_matches_backend(
                shs::renderpath::render_domain_device(), RenderBackendType::OpenGL)) return false;
        if (shs::renderpath::render_domain_matches_backend(
                shs::renderpath::render_domain_substrate(Substrate::OpenGL), RenderBackendType::Vulkan)) return false;
        return true;
    }
    // Minimal pass: explicit I/O domain plus a switchable interop boundary.
    struct DomainProbePass : shs::renderpath::IRenderPass
    {
        const char* label = "probe";
        shs::renderpath::PassIODesc io{};
        bool interop = false;

        const char* id() const override { return label; }
        bool is_interop_pass() const override { return interop; }
        shs::renderpath::PassIODesc describe_io() const override { return io; }
        shs::renderpath::PassExecutionResult execute_resolved(
            shs::Context&, const shs::renderpath::PassExecutionRequest&) override
        {
            return shs::renderpath::PassExecutionResult::executed_no_outputs();
        }
    };

    // Two passes write one shared resource from different execution units;
    // `interop` decides whether that crossing is declared and therefore legal.
    void hybrid_cross_unit_plan(bool interop, bool& out_valid, size_t& out_errors)
    {
        DomainProbePass host_pass{};
        host_pass.label = "host_writer";
        host_pass.interop = interop;
        DomainProbePass device_pass{};
        device_pass.label = "device_writer";
        device_pass.interop = interop;

        const uint64_t key =
            shs::renderpath::pass_rt_resource_key(shs::renderpath::PassResourceType::ColorHDR, 7u);
        shs::renderpath::PassResourceRef shared{};
        shared.key = key;
        shared.type = shs::renderpath::PassResourceType::ColorHDR;
        shared.access = shs::renderpath::PassResourceAccess::Write;
        shared.name = "shared_color";
        shared.domain = shs::renderpath::render_domain_host();
        host_pass.io.write(shared);

        shared.domain = shs::renderpath::render_domain_device();
        device_pass.io.write(shared);

        shs::renderpath::FrameGraph graph{};
        graph.add_node(shs::renderpath::FrameGraphNode{ &host_pass, host_pass.label, host_pass.io, 0 });
        graph.add_node(shs::renderpath::FrameGraphNode{ &device_pass, device_pass.label, device_pass.io, 1 });
        graph.compile();

        out_valid = graph.report().valid;
        out_errors = graph.report().errors.size();
    }

    // D2: hybrid is REJECTED without an interop boundary, accepted with one.
    bool test_hybrid_interop_legality()
    {
        bool rejected_valid = true;
        size_t rejected_errors = 0;
        hybrid_cross_unit_plan(false, rejected_valid, rejected_errors);
        if (rejected_valid) return false;       // must be a rejection...
        if (rejected_errors == 0) return false; // ...and must say why

        bool accepted_valid = false;
        size_t accepted_errors = 1;
        hybrid_cross_unit_plan(true, accepted_valid, accepted_errors);
        if (!accepted_valid) return false;       // a declared boundary is legal
        if (accepted_errors != 0) return false;
        return true;
    }
    // --- RP-1 fixtures -------------------------------------------------------
    // A registry that declares, per pass, which SUBSTRATES can realize it and
    // whether it declares an interop boundary. These are exactly the two facts
    // the resolver reads, so the test controls them directly instead of hoping
    // the builtin table happens to exercise the case. (Worth knowing: the real
    // `make_standard_pass_factory_registry` registers every standard pass
    // software-only, which is the dual-realization gap recorded elsewhere —
    // here we deliberately author a device realization.)
    shs::renderpath::PassFactoryRegistry make_substrate_probe_registry(
        bool shadow_map_interop = false,
        bool forward_interop = false,
        uint32_t shadow_map_mask = shs::renderpath::substrate_mask_all(),
        uint32_t forward_mask = shs::renderpath::substrate_mask_all(),
        uint32_t tonemap_mask = shs::renderpath::substrate_mask_all())
    {
        using shs::PassId;
        using shs::renderpath::PassFactoryRegistry;
        using shs::renderpath::TechniquePassContract;
        using shs::render::technique_mode_mask_all;

        PassFactoryRegistry reg{};
        auto add = [&](PassId id, uint32_t mask, bool interop) {
            reg.register_factory(id, []() -> std::unique_ptr<shs::renderpath::IRenderPass> {
                return std::make_unique<DomainProbePass>();
            });
            TechniquePassContract contract{};
            contract.supported_modes_mask = technique_mode_mask_all();
            reg.register_descriptor(id, contract, mask, true, interop);
        };
        add(PassId::ShadowMap, shadow_map_mask, shadow_map_interop);
        add(PassId::PBRForward, forward_mask, forward_interop);
        add(PassId::Tonemap, tonemap_mask, false);
        return reg;
    }

    // Every pass in `make_soft_shadow_culling_recipe`, each realizable on every
    // substrate. Needed because that recipe's chain is longer than the three-pass
    // probe above — `DepthPrepass`, `LightCulling`, `PBRForwardPlus` and
    // `MotionBlur` are absent from it — and the registry gate below compiles the
    // *default registry's* recipe rather than a test-authored stand-in, so the
    // fixture has to cover exactly the passes that recipe names.
    shs::renderpath::PassFactoryRegistry make_full_probe_registry()
    {
        using shs::PassId;
        using shs::renderpath::PassFactoryRegistry;
        using shs::renderpath::TechniquePassContract;
        using shs::render::technique_mode_mask_all;

        PassFactoryRegistry reg{};
        auto add = [&](PassId id) {
            reg.register_factory(id, []() -> std::unique_ptr<shs::renderpath::IRenderPass> {
                return std::make_unique<DomainProbePass>();
            });
            TechniquePassContract contract{};
            contract.supported_modes_mask = technique_mode_mask_all();
            reg.register_descriptor(
                id, contract, shs::renderpath::substrate_mask_all(), true, false);
        };
        add(PassId::ShadowMap);
        add(PassId::DepthPrepass);
        add(PassId::LightCulling);
        add(PassId::PBRForwardPlus);
        add(PassId::Tonemap);
        add(PassId::MotionBlur);
        return reg;
    }

    // Three required passes, no per-pass intent: the policy is the only thing
    // choosing. `declared` stays Software throughout, so any pass that resolves
    // elsewhere resolved there by substitution — which is the feature.
    shs::renderpath::RenderPathRecipe make_substrate_probe_recipe(
        shs::renderpath::SubstratePolicy policy)
    {
        using namespace shs;
        using namespace shs::renderpath;
        RenderPathRecipe recipe{};
        recipe.name = "substrate_probe";
        recipe.substrate_policy = policy;
        recipe.backend = RenderBackendType::Software;
        recipe.render_technique = RenderPathRenderingTechnique::ForwardLit;
        recipe.technique_mode = TechniqueMode::Forward;
        recipe.view_culling = RenderPathCullingMode::Frustum;
        recipe.shadow_culling = RenderPathCullingMode::Frustum;
        recipe.wants_shadows = false;
        recipe.pass_chain = {
            make_render_path_pass_entry(PassId::ShadowMap, true),
            make_render_path_pass_entry(PassId::PBRForward, true),
            make_render_path_pass_entry(PassId::Tonemap, true)
        };
        return recipe;
    }

    // A host that can drive every substrate: the explicit opt-in that makes a
    // multi-substrate resolution possible at all.
    shs::renderpath::RenderPathCapabilitySet make_multi_substrate_caps()
    {
        shs::renderpath::RenderPathCapabilitySet caps = make_sw_caps();
        caps.available_substrate_mask = shs::renderpath::substrate_mask_all();
        return caps;
    }

    bool all_passes_on(const shs::renderpath::RenderPathExecutionPlan& plan,
                       shs::renderpath::Substrate substrate)
    {
        if (!plan.valid || plan.pass_chain.empty()) return false;
        for (const auto& pass : plan.pass_chain)
        {
            if (!pass.substrate_resolved) return false;
            if (pass.substrate != substrate) return false;
        }
        return true;
    }

    // --- RP-1 acceptance tests ----------------------------------------------

    // RP-1 (req 4): ONE recipe, four policies, four resolutions — no recipe
    // fork. Every pass here realizes every substrate, so the policy is the only
    // variable and none of the assertions can pass by accident.
    bool test_substrate_policy_resolution()
    {
        using namespace shs::renderpath;
        const RenderPathCompiler compiler{};
        const RenderPathCapabilitySet caps = make_multi_substrate_caps();
        const PassFactoryRegistry registry = make_substrate_probe_registry();

        const RenderPathExecutionPlan exact = compiler.compile(
            make_substrate_probe_recipe(SubstratePolicy::ExactMatch), caps, &registry);
        if (!all_passes_on(exact, Substrate::SoftwareRaster)) return false;

        const RenderPathExecutionPlan device = compiler.compile(
            make_substrate_probe_recipe(SubstratePolicy::DevicePreferred), caps, &registry);
        if (!all_passes_on(device, Substrate::Vulkan)) return false;

        const RenderPathExecutionPlan host = compiler.compile(
            make_substrate_probe_recipe(SubstratePolicy::HostPreferred), caps, &registry);
        if (!all_passes_on(host, Substrate::SoftwareRaster)) return false;

        const RenderPathExecutionPlan cheapest = compiler.compile(
            make_substrate_probe_recipe(SubstratePolicy::Cheapest), caps, &registry);
        if (!all_passes_on(cheapest, Substrate::SoftwareRaster)) return false;

        // The policy is echoed as plan data, and no chain crossed a boundary.
        if (device.substrate_policy != SubstratePolicy::DevicePreferred) return false;
        if (device.hybrid || exact.hybrid || host.hybrid || cheapest.hybrid) return false;

        // The substitution is real: the declared substrate is Software
        // throughout, and the device-preferring policy resolved off it anyway.
        if (device.backend != shs::RenderBackendType::Software) return false;
        return true;
    }

    // Declared per-pass intent BINDS the policy: a pass that says it must run on
    // the device cannot be resolved onto the host rasterizer, even under
    // HostPreferred. Intent narrows; policy only chooses inside the narrowing.
    bool test_substrate_intent_binds_policy()
    {
        using namespace shs::renderpath;
        const RenderPathCompiler compiler{};
        const RenderPathCapabilitySet caps = make_multi_substrate_caps();
        const PassFactoryRegistry registry = make_substrate_probe_registry();

        RenderPathRecipe recipe = make_substrate_probe_recipe(SubstratePolicy::HostPreferred);
        for (auto& entry : recipe.pass_chain)
        {
            entry = with_domain(entry, render_domain_device());
        }

        const RenderPathExecutionPlan plan = compiler.compile(recipe, caps, &registry);
        // HostPreferred's first admissible choice is the host rasterizer; the
        // intent removes it, so the ladder falls to the next device substrate.
        if (!all_passes_on(plan, Substrate::OpenGL)) return false;
        if (plan.hybrid) return false; // unanimous, so nothing crossed
        return true;
    }

    // RP-2's hybrid rule, now reachable at PLAN time: two adjacent passes
    // resolved onto different substrates is a rejection unless the crossing
    // declares an interop boundary. Shared staging is the resource tier's half
    // (enforced in `frame_graph.hpp`, same-key by construction); this pins the
    // plan-visible half, which is the half the resolver controls.
    bool test_substrate_hybrid_legality()
    {
        using namespace shs::renderpath;
        const RenderPathCompiler compiler{};
        const RenderPathCapabilitySet caps = make_multi_substrate_caps();

        // ShadowMap is host-realizable only, the forward/tonemap passes
        // device-realizable only: under DevicePreferred they cannot land on one
        // substrate. No intent is declared — the crossing comes from
        // realizability alone.
        const uint32_t sw_only = substrate_bit(Substrate::SoftwareRaster);
        const uint32_t device_only = substrate_mask_of(ExecutionUnit::Device);
        const RenderPathRecipe recipe = make_substrate_probe_recipe(SubstratePolicy::DevicePreferred);

        const PassFactoryRegistry undeclared =
            make_substrate_probe_registry(false, false, sw_only, device_only, device_only);
        const RenderPathExecutionPlan rejected = compiler.compile(recipe, caps, &undeclared);
        if (rejected.valid) return false;
        if (rejected.rejection != RenderPathCompileRejection::HybridBoundaryUndeclared) return false;
        if (rejected.errors.empty()) return false; // must say why

        // Identical resolution, with the crossing pass declaring the boundary.
        const PassFactoryRegistry declared =
            make_substrate_probe_registry(false, true, sw_only, device_only, device_only);
        const RenderPathExecutionPlan accepted = compiler.compile(recipe, caps, &declared);
        if (!accepted.valid) return false;
        if (!accepted.hybrid) return false; // a hybrid, recorded as data
        if (accepted.pass_chain.size() != 3) return false;
        if (accepted.pass_chain[0].substrate != Substrate::SoftwareRaster) return false;
        if (accepted.pass_chain[1].substrate != Substrate::Vulkan) return false;
        if (accepted.pass_chain[2].substrate != Substrate::Vulkan) return false;
        return true;
    }

    // The plan's snapshot-equality contract must keep holding across
    // resolutions: a resolution is DATA (req 4's stated residual), never ambient
    // state. Same inputs => equal plans; one changed input => unequal.
    bool test_substrate_resolution_snapshot_contract()
    {
        using namespace shs::renderpath;
        const RenderPathCompiler compiler{};
        const RenderPathCapabilitySet caps = make_multi_substrate_caps();
        const PassFactoryRegistry registry = make_substrate_probe_registry();

        const RenderPathRecipe recipe = make_substrate_probe_recipe(SubstratePolicy::DevicePreferred);
        const RenderPathExecutionPlan first = compiler.compile(recipe, caps, &registry);
        const RenderPathExecutionPlan again = compiler.compile(recipe, caps, &registry);
        if (!(first == again)) return false; // deterministic, no hidden state

        // The only difference is the policy, and it is visible in the plan.
        RenderPathRecipe rehosted = recipe;
        rehosted.substrate_policy = SubstratePolicy::HostPreferred;
        const RenderPathExecutionPlan host = compiler.compile(rehosted, caps, &registry);
        if (host == first) return false;
        if (host.substrate_policy == first.substrate_policy) return false;
        if (!all_passes_on(host, Substrate::SoftwareRaster)) return false;

        // A host that declares NO substrates resolves onto the declared one —
        // the pre-RP-1 identity, unchanged, for the very recipe that substituted
        // elsewhere above.
        const RenderPathExecutionPlan single = compiler.compile(recipe, make_sw_caps(), &registry);
        if (!all_passes_on(single, Substrate::SoftwareRaster)) return false;
        if (single.hybrid) return false;
        if (!(single == compiler.compile(recipe, make_sw_caps(), &registry))) return false;

        // Unchanged path, and the historical reason: a pass no admissible
        // substrate can realize is still `BackendUnavailable`.
        const uint32_t device_only = substrate_mask_of(ExecutionUnit::Device);
        const PassFactoryRegistry device_registry =
            make_substrate_probe_registry(false, false, device_only, device_only, device_only);
        const RenderPathExecutionPlan unavailable = compiler.compile(
            make_substrate_probe_recipe(SubstratePolicy::ExactMatch), make_sw_caps(), &device_registry);
        if (unavailable.valid) return false;
        if (unavailable.rejection != RenderPathCompileRejection::BackendUnavailable) return false;

        // Distinct from the above: realizability is fine, but the declared
        // intent cannot be satisfied here — that is the new, narrower reason.
        RenderPathRecipe conflicting = make_substrate_probe_recipe(SubstratePolicy::DevicePreferred);
        conflicting.pass_chain[0] = with_domain(conflicting.pass_chain[0], render_domain_device());
        const RenderPathExecutionPlan unsatisfiable = compiler.compile(conflicting, make_sw_caps(), &registry);
        if (unsatisfiable.valid) return false;
        if (unsatisfiable.rejection != RenderPathCompileRejection::SubstrateUnresolved) return false;
        return true;
    }

    // RP-1 (req 4) at the PUBLIC seam: the registry now holds ONE default recipe,
    // and that single recipe resolves to the device on a host that advertises
    // devices and to the host rasterizer on one that offers only the host. Before
    // RP-1 the same outcome required two authored recipes
    // (`soft_shadow_culling_vk_default` / `soft_shadow_culling_sw_default`) with
    // different pass chains AND different technique modes — i.e. the substrate
    // chose the *shape* of the path. This is the remedy stated where a consumer
    // can see it, which is why it asserts against the registry rather than a
    // test-authored recipe.
    bool test_registry_single_recipe_two_substrates()
    {
        using namespace shs::renderpath;

        RenderPathRegistry recipes{};
        recipes.register_default_recipes();

        // One recipe, not one per substrate.
        if (recipes.recipe_ids().size() != 1u) return false;
        if (!recipes.has_recipe("soft_shadow_culling")) return false;
        // The fork's two names are gone: a consumer that used to look either of
        // them up now finds exactly one entry, and it is neither of them.
        if (recipes.has_recipe("soft_shadow_culling_vk_default")) return false;
        if (recipes.has_recipe("soft_shadow_culling_sw_default")) return false;

        const RenderPathRecipe* recipe = recipes.find_recipe("soft_shadow_culling");
        if (recipe == nullptr) return false;
        if (recipe->substrate_policy != SubstratePolicy::DevicePreferred) return false;

        const RenderPathCompiler compiler{};
        const PassFactoryRegistry registry = make_full_probe_registry();

        // Host A: advertises every substrate. Device-preferred lands on the
        // device for every pass, and nothing crossed a boundary.
        const RenderPathExecutionPlan on_device =
            compiler.compile(*recipe, make_multi_substrate_caps(), &registry);
        if (!all_passes_on(on_device, Substrate::Vulkan)) return false;
        if (on_device.hybrid) return false;

        // Host B: offers the host rasterizer only. The SAME recipe value lands on
        // software — no clone, no second contract registry, no second plan shape.
        RenderPathCapabilitySet host_only = make_sw_caps();
        host_only.available_substrate_mask = substrate_bit(Substrate::SoftwareRaster);
        const RenderPathExecutionPlan on_host = compiler.compile(*recipe, host_only, &registry);
        if (!all_passes_on(on_host, Substrate::SoftwareRaster)) return false;
        if (on_host.hybrid) return false;

        // One authored recipe, two resolutions: same name, same pass count, same
        // policy — the plans differ only in where the passes resolved, which is
        // the feature and the reason the second recipe could be deleted.
        if (on_device.recipe_name != on_host.recipe_name) return false;
        if (on_device.pass_chain.size() != on_host.pass_chain.size()) return false;
        if (on_device.substrate_policy != on_host.substrate_policy) return false;
        if (on_device.pass_chain.size() != recipe->pass_chain.size()) return false;
        return true;
    }
} // namespace

int main()
{
    bool ok = true;
    auto check = [](const char* name, bool (*fn)()) {
        const bool r = fn();
        std::fprintf(stderr, "[renderpath-tests] %-36s %s\n", name, r ? "PASS" : "FAIL");
        return r;
    };

    ok = check("path_selection", test_path_selection) && ok;
    ok = check("technique_switching", test_technique_switching) && ok;
    ok = check("culling_mode_changes", test_culling_mode_changes) && ok;
    ok = check("rejection_keeps_previous_plan", test_rejection_keeps_previous_plan) && ok;
    ok = check("rejected_view_culling_has_no_changed_fact", test_rejected_view_culling_has_no_changed_fact) && ok;
    ok = check("rejected_swap_facts_and_recovery", test_rejected_swap_facts_and_recovery) && ok;
    ok = check("runtime_toggles", test_runtime_toggles) && ok;
    ok = check("unchanged_facts", test_unchanged_facts) && ok;
    ok = check("plan_generation_semantics", test_plan_generation_semantics) && ok;
    ok = check("kit_replay_deterministic", test_kit_replay_deterministic) && ok;
    ok = check("kit_empty_log_stable", test_kit_empty_log_stable) && ok;
    ok = check("domain_axes_are_independent", test_domain_axes_are_independent) && ok;
    ok = check("domain_compatibility_relation", test_domain_compatibility_relation) && ok;
    ok = check("domain_backend_matching", test_domain_backend_matching) && ok;
    ok = check("hybrid_interop_legality", test_hybrid_interop_legality) && ok;
    ok = check("substrate_policy_resolution", test_substrate_policy_resolution) && ok;
    ok = check("substrate_intent_binds_policy", test_substrate_intent_binds_policy) && ok;
    ok = check("substrate_hybrid_legality", test_substrate_hybrid_legality) && ok;
    ok = check("substrate_resolution_snapshot_contract", test_substrate_resolution_snapshot_contract) && ok;
    ok = check("registry_single_recipe_two_substrates", test_registry_single_recipe_two_substrates) && ok;

    if (!ok)
    {
        std::fprintf(stderr, "[renderpath-tests] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[renderpath-tests] all tests passed\n");
    return 0;
}
