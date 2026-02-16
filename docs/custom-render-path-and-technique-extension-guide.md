# Custom Render Path and Technique Extension Guide

Last updated: 2026-02-16

## 1) Purpose

This note explains how to add your own rendering techniques and render paths in the current SHS composition system.

It focuses on practical extension paths:

- tune an existing composition without engine edits
- add a new composition preset (`path + technique + post stack`)
- add a new rendering technique preset
- add a new render-path preset
- add a new pass (typed standard pass or custom string pass)

## 2) Mental Model (Current Architecture)

Runtime flow is data-driven:

1. `RenderPathRecipe` declares path/culling/pass-chain intent.
2. `RenderPathCompiler` validates and compiles it to `RenderPathExecutionPlan`.
3. `compile_render_path_resource_plan(...)` derives semantic resources and bindings.
4. `compile_render_path_barrier_plan(...)` derives synchronization/layout transitions.
5. `RenderPathPassDispatcher<TContext>` executes passes in compiled order with registered handlers.

Main extension files:

- recipe + enums:
  - `cpp-folders/src/shs-renderer-lib/include/shs/pipeline/render_path_recipe.hpp`
  - `cpp-folders/src/shs-renderer-lib/include/shs/frame/technique_mode.hpp`
  - `cpp-folders/src/shs-renderer-lib/include/shs/pipeline/pass_id.hpp`
- built-in presets:
  - `cpp-folders/src/shs-renderer-lib/include/shs/pipeline/render_path_presets.hpp`
  - `cpp-folders/src/shs-renderer-lib/include/shs/pipeline/render_technique_presets.hpp`
  - `cpp-folders/src/shs-renderer-lib/include/shs/pipeline/render_composition_presets.hpp`
- contract/resource validation:
  - `cpp-folders/src/shs-renderer-lib/include/shs/pipeline/pass_contract.hpp`
  - `cpp-folders/src/shs-renderer-lib/include/shs/pipeline/pass_contract_registry.hpp`
  - `cpp-folders/src/shs-renderer-lib/include/shs/pipeline/render_path_compiler.hpp`
  - `cpp-folders/src/shs-renderer-lib/include/shs/pipeline/render_path_resource_plan.hpp`
- pass dispatch:
  - `cpp-folders/src/shs-renderer-lib/include/shs/pipeline/render_path_pass_dispatch.hpp`
  - `cpp-folders/src/shs-renderer-lib/include/shs/pipeline/render_path_standard_pass_routing.hpp`

## 3) Fastest Path: Custom Composition (No Core Enum Changes)

If existing path and technique types are enough, only create a composition recipe and apply it at runtime.

```cpp
const shs::RenderCompositionRecipe custom = shs::make_builtin_render_composition_recipe(
    shs::RenderPathPreset::Deferred,
    shs::RenderTechniquePreset::PBR,
    "my_composition",
    shs::RenderCompositionPostStackPreset::Temporal);

const shs::RenderCompositionResolved resolved =
    shs::resolve_builtin_render_composition_recipe(
        custom,
        shs::RenderBackendType::Vulkan,
        "path",
        "technique");

if (!render_path_executor.apply_recipe(resolved.path_recipe, ctx, &pass_contract_registry))
{
    // inspect compiler/resource/barrier validation messages
}
```

Use this when you only need different combinations/order of already-supported features.

## 4) Add a New Rendering Technique Preset

Use this when shading behavior changes but path architecture stays similar.

Edit:

- `cpp-folders/src/shs-renderer-lib/include/shs/pipeline/render_technique_presets.hpp`

Required updates:

1. Add enum value to `RenderTechniquePreset`.
2. Update `render_technique_preset_name(...)`.
3. Update shading mapping:
   - `render_technique_shading_model(...)`
   - `render_technique_preset_from_shading_model(...)` if needed
4. Update shader variant mapping:
   - `render_technique_shader_variant(...)`
5. Update cycle order:
   - `default_render_technique_preset_order()`
6. Define defaults:
   - `make_builtin_render_technique_recipe(...)`

Important:

- If the technique needs a new shader branch, keep `render_technique_shader_variant(...)` and shader-side variant constants aligned.

## 5) Add a New Render Path Preset

Use this when pass architecture changes (for example a new deferred variant).

Edit:

- `cpp-folders/src/shs-renderer-lib/include/shs/pipeline/render_path_presets.hpp`
- possibly `cpp-folders/src/shs-renderer-lib/include/shs/frame/technique_mode.hpp` (if introducing a brand-new mode)
- `cpp-folders/src/shs-renderer-lib/include/shs/pipeline/technique_profile.hpp` (default pass chain for that mode)

Required updates:

1. Add enum value in `RenderPathPreset`.
2. Update:
   - `render_path_preset_name(...)`
   - `render_path_preset_mode(...)`
   - `render_path_preset_for_mode(...)`
   - `render_path_rendering_technique_for_mode(...)`
   - `default_light_culling_mode_for_mode(...)`
   - `default_render_path_preset_order()`
3. Ensure `make_builtin_render_path_recipe(...)` generates the right pass chain and runtime defaults.
4. If mode is new, update `TechniqueMode` and `technique_mode_mask_all()`.
5. Add/adjust default profile in `make_default_technique_profile(...)`.

## 6) Add a New Pass

There are two supported extension styles.

### 6.1 Style A: Typed standard pass (`PassId`)

Use this when the pass should be first-class and reusable across many recipes.

Edit:

- `cpp-folders/src/shs-renderer-lib/include/shs/pipeline/pass_id.hpp`
- `cpp-folders/src/shs-renderer-lib/include/shs/pipeline/pass_contract_registry.hpp`
- `cpp-folders/src/shs-renderer-lib/include/shs/pipeline/render_path_standard_pass_routing.hpp` (if it is part of standard handler routing)
- backend execution bindings (for example Vulkan host binding or shared helper usage)

Required steps:

1. Add new `PassId` value.
2. Add string mapping in `pass_id_name(...)` and `parse_pass_id(...)`.
3. Add a semantic contract in `lookup_standard_pass_contract(...)`.
4. Add it to `known_pass_ids` in standard registry builders.
5. Register dispatch handler (typed) in standard routing or host-specific dispatcher setup.

### 6.2 Style B: Custom string pass (no `PassId` change)

Use this when experimenting quickly or for demo-specific passes.

How:

1. Keep `RenderPathPassEntry.pass_id = PassId::Unknown`.
2. Set a custom string `RenderPathPassEntry.id`.
3. Register pass factory by string:
   - `PassFactoryRegistry::register_factory("my_custom_pass", ...)`
4. Register dispatcher handler by string:
   - `dispatcher.register_handler("my_custom_pass", handler)`

This is valid and already supported by the compiler/dispatcher.

## 7) Contract Semantics Rules (Most Important for Stability)

Pass contracts are not only for documentation; they drive plan/resource validation.

Edit/define semantics in:

- `cpp-folders/src/shs-renderer-lib/include/shs/pipeline/pass_contract.hpp`
- `cpp-folders/src/shs-renderer-lib/include/shs/pipeline/pass_contract_registry.hpp`

Must stay coherent for each read/write semantic:

- `space` (screen/view/light/tile)
- `encoding` (linear/srgb/depth/velocity/indices/counts)
- `lifetime` (transient/persistent/history)
- `temporal_role` (current/history_read/history_write)

If producer and consumer semantic representation mismatches, `compile_render_path_resource_plan(...)` marks plan invalid.

## 8) Runtime Integration Pattern

Typical host-side sequence:

1. Build pass registry (usually standard contract registry for backend).
2. Build/resolve recipe or composition recipe.
3. Apply recipe through `RenderPathExecutor`.
4. If valid, use:
   - `active_plan()` for pass sequence
   - `active_resource_plan()` for resource layout/bindings
   - `active_barrier_plan()` for synchronization model
5. Execute with dispatcher handlers.

Validation checks to watch:

- `RenderPathExecutionPlan.valid`
- `RenderPathResourcePlan.valid`
- `RenderPathBarrierPlan.valid`
- warning/error arrays in each plan

## 9) Minimal Checklists

### 9.1 New technique preset checklist

1. Technique enum + name mapping updated.
2. Shading model + shader variant mapping updated.
3. Technique default recipe tuned.
4. Composition presets include the new technique where desired.
5. Runtime cycle order includes it.

### 9.2 New path preset checklist

1. Path enum/mode mapping updated.
2. Default culling/runtime defaults set.
3. Pass chain/profile set for the mode.
4. Required handlers exist for all required passes.
5. Compiled plan + resource plan + barrier plan all valid.

### 9.3 New pass checklist

1. Pass ID or custom string is consistent in recipe and handler registration.
2. Contract semantics are complete and correct.
3. Pass is registered in `PassFactoryRegistry`.
4. Dispatcher handler is bound.
5. Optional vs required flag is intentional.

## 10) Practical Recommendation

Prefer this order for safe extension:

1. Start with a custom composition recipe.
2. If not enough, add a new technique preset.
3. Then add a new path preset.
4. Add new passes only when architectural needs require it.

This keeps validation signal clear and avoids large breakage while iterating quickly.
