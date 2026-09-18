#pragma once

/*
    SHS RENDERER SAN

    FILE: renderpath.contract.hpp
    MODULE: domains/renderpath
    PURPOSE: Domain Pod CONTRACT for the renderpath pod — the first formal
             Domain Pod in the engine lib (roadmap P1).

    The contract re-exports the value spine (recipe -> compiler -> plan ->
    capabilities -> runtime-state) from the execution planner zone. The spine
    types keep their canonical shs:: names; this pod is the sanctioned seam
    where domains face execution (checked by the boundary linter carve-out).
    pipeline/ headers remain unchanged for existing consumers — pod re-exports
    only, no breakage.
*/

#include "shs/renderpath/planning/render_path_capabilities.hpp"
#include "shs/renderpath/planning/render_path_compiler.hpp"
#include "shs/renderpath/planning/render_path_recipe.hpp"
#include "shs/renderpath/execution/render_path_runtime_state.hpp"

namespace shs::renderpath
{
    // --- Recipe vocabulary (what a render path IS) ---
    using shs::RenderPathRecipe;
    using shs::RenderPathPassEntry;
    using shs::RenderPathLightVolumeProvider;
    using shs::RenderPathCullingMode;
    using shs::RenderPathRenderingTechnique;
    using shs::make_render_path_pass_entry;
    using shs::make_soft_shadow_culling_recipe;
    // RP-1 recipe-authoring vocabulary: the per-pass substrate INTENT
    // (`RenderDomain` / `with_domain`) and the recipe-level policy that decides
    // how it is resolved. Re-exported because they are inputs a consumer writes
    // on a recipe, exactly like the fields above.
    using shs::RenderDomain;
    using shs::with_domain;
    using shs::SubstratePolicy;

    // --- Capability vocabulary (what the current stack CAN do) ---
    using shs::RenderPathCapabilitySet;
    using shs::make_render_path_capability_set;

    // --- Runtime-state vocabulary (frame-mutable toggles) ---
    using shs::RenderPathRuntimeState;

    // --- Plan vocabulary (what the compiler PRODUCED) ---
    using shs::RenderPathCompatibilityRules;
    using shs::RenderPathCompiledPass;
    using shs::RenderPathExecutionPlan;
    using shs::RenderPathCompiler;
    using shs::make_technique_profile;
} // namespace shs::renderpath
