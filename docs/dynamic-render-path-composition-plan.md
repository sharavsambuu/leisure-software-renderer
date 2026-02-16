# Dynamic Compositional and Configurable Render Paths

This document is the implementation plan for the next phase:

- Different light volume providers
- Different culling strategies
- Different rendering techniques
- Different pass chains

The goal is to compose these dimensions dynamically via recipes/configuration, without hardcoding each permutation into separate demos.

## 1) Why This Phase

Most required building blocks already exist:

- Jolt shape/volume primitives for scene and culling data
- View and shadow frustum/occlusion culling flows
- Multi-pass Vulkan flow (shadow, prepass, query, main)
- Runtime toggles for A/B testing

What is still missing is a shared composition layer:

- module selection from recipe/config
- less ad-hoc glue in each demo

## 2) Core Principles

- Data-first: behavior is selected by config/recipe.
- Small stable interfaces: each axis is a strategy module.
- Capability-aware: reject unsupported combinations early.
- Safe defaults: risky options stay OFF by default.
- Incremental migration: move existing demos in phases.

## 3) Target Architecture

### 3.1 Core Runtime Objects

- `RenderPathRecipe`
  - Declares selected modules/options.
  - Example fields:
    - `backend`
    - `light_volume_provider`
    - `view_culling`
    - `shadow_culling`
    - `render_technique`
    - `pass_chain`
    - `runtime_defaults`

- `RenderPathRegistry`
  - Maps recipe IDs to concrete implementations.

- `RenderPathCompiler`
  - Validates recipe using capability and compatibility rules.
  - Compiles into an executable graph/plan.

- `RenderPathInstance`
  - Runtime instance of a compiled recipe.
  - Runs per-frame update and pass execution/recording.

- `RenderPathRuntimeState`
  - Runtime toggles/state.
  - Example:
    - `view_occlusion_enabled`
    - `shadow_occlusion_enabled`
    - `debug_aabb`
    - `lit_mode`

### 3.2 Frame Data Contract

Use shared frame data bundles to avoid hidden coupling:

- `FrameSceneData`
- `FrameCameraData`
- `FrameLightData`
- `FrameCullData`
- `FramePassResources`
- `FrameStats`

### 3.3 Strategy Interfaces (Initial Draft)

```cpp
struct ILightVolumeProvider {
    virtual ~ILightVolumeProvider() = default;
    virtual void build(const FrameSceneData&, FrameLightData&) = 0;
};

struct ICullingStrategy {
    virtual ~ICullingStrategy() = default;
    virtual void run_view(FrameCullData&) = 0;
    virtual void run_shadow(FrameCullData&) = 0;
};

struct IRenderTechnique {
    virtual ~IRenderTechnique() = default;
    virtual void record(FramePassResources&, const FrameSceneData&, const FrameCullData&) = 0;
};

struct IPassNode {
    virtual ~IPassNode() = default;
    virtual void execute(FramePassResources&, const FrameSceneData&, const FrameCullData&) = 0;
};
```

## 4) Compatibility Model

Opening the full Cartesian product is not practical.

Use:

- `CapabilitySet`
- `CompatibilityRules`

Example capabilities:

- backend depth attachment availability
- occlusion query support
- secondary command recording support

Example rules:

- a render technique may require shadow-map pass
- a culling mode may require depth prepass
- certain resource formats may be mandatory

When a recipe is unsupported:

- strict mode: hard failure
- permissive mode: fallback + warning logs

## 5) Recipe Schema (Initial)

Start with a C++ struct; later add JSON/TOML/YAML loader.

Example:

```json
{
  "name": "soft_shadow_culling_vk_default",
  "backend": "vulkan",
  "light_volume_provider": "jolt_shape_volumes",
  "view_culling": "frustum+occlusion",
  "shadow_culling": "frustum+optional_occlusion",
  "render_technique": "forward_lit",
  "pass_chain": ["shadow_map", "depth_prepass", "occlusion_query", "main_lit", "debug_overlay"],
  "runtime_defaults": {
    "view_occlusion_enabled": true,
    "shadow_occlusion_enabled": false,
    "debug_aabb": false
  }
}
```

## 6) Implementation Phases

### Phase 0: Inventory + Baseline Freeze

- Inventory current demo behavior
- Save visual baseline screenshots and perf baseline
- Build regression reference set

### Phase 1: Composition Scaffold

- Add `RenderPathRecipe`, `RenderPathRegistry`, `RenderPathCompiler`
- Add `CapabilitySet`, `CompatibilityRules`
- Add initial `RenderPathRuntimeState`

Deliverable:

- recipe compile pipeline works (no-op or simple path)

### Phase 2: Port Vulkan Soft Shadow Culling Path

- Wrap `hello_soft_shadow_culling_vk` flow into modular path nodes
- Map existing toggles/state to runtime state
- Preserve current defaults:
  - shadow occlusion = OFF
  - view occlusion = ON

Deliverable:

- 1:1 behavior parity with current demo

### Phase 3: Port Software Soft Shadow Culling Path

- Integrate `hello_soft_shadow_culling_sw` into the same framework
- Align SW/VK stats/reporting contract

Deliverable:

- SW/VK paths selectable by recipe

### Phase 4: Expand Axes

- Add more light volume providers
- Add more culling strategy variants
- Add more render/pass variants with rule + capability checks

Deliverable:

- at least 3 production-safe preset recipes

### Phase 5: Tooling

- Recipe serialization + loader
- Debug UI/CLI selector
- Recipe validation command

Deliverable:

- path switching without rewriting demo logic

## 7) Suggested File Placement

Inside `shs-renderer-lib`:

- `include/shs/pipeline/render_path_recipe.hpp`
- `include/shs/pipeline/render_path_registry.hpp`
- `include/shs/pipeline/render_path_compiler.hpp`
- `include/shs/pipeline/render_path_runtime_state.hpp`
- `include/shs/pipeline/render_path_capabilities.hpp`
- `include/shs/pipeline/render_path_interfaces.hpp`

If separating implementation `.cpp` files:

- `src/shs-renderer-lib/src/pipeline/`

## 8) First Concrete Backlog

- Recipe/registry ID lookup skeleton
- Runtime state object and toggle mapping
- Wrap VK soft shadow pass sequence into pass nodes
- Wrap SW soft shadow pass sequence into pass nodes
- Compatibility checks for:
  - shadow map dependency
  - occlusion query dependency
  - depth attachment dependency
- Integration test harness:
  - recipe compile
  - execute N frames
  - stats sanity checks

## 9) Risks and Mitigations

- Permutation explosion
  - Mitigation: rules + curated presets
- Visual regressions
  - Mitigation: baseline screenshots + stats checks
- Runtime instability (especially aggressive occlusion)
  - Mitigation: conservative defaults + warmup + fallback
- Harder debugging
  - Mitigation: per-module stats + recipe dump logs

## 10) Definition of Done for This Phase

- Two real recipes run on the shared composition layer:
  - Vulkan soft shadow culling
  - Software soft shadow culling
- Runtime toggles/state are controlled through one contract
- Shadow occlusion default remains OFF
- Compatibility validation rejects invalid recipes early

## 11) Shader Modularization Baseline (Now In Place)

To support scalable render-path composition, a shared Vulkan shader common layer now exists:

- `cpp-folders/src/shs-renderer-lib/shaders/vulkan/common/math.glsl`
- `cpp-folders/src/shs-renderer-lib/shaders/vulkan/common/light_constants.glsl`
- `cpp-folders/src/shs-renderer-lib/shaders/vulkan/common/light_math.glsl`
- `cpp-folders/src/shs-renderer-lib/shaders/vulkan/common/culling_light_struct.glsl`

Current paths already consuming these shared blocks:

- light-types culling Vulkan demo fragment shader
- forward-plus stress scene fragment shader
- forward-plus stress light-culling compute shader
- shape cell/volume culling compute shaders

This reduces shader contract drift and should be treated as a required baseline for future recipe modules.

## 12) Immediate Follow-up (Recommended)

Before increasing render-path permutations, add one more shader refactor pass:

- extract shared BRDF helpers into common modules
- extract shared shadow sampling/filter helpers into common modules
- keep per-demo style/tuning local, but keep math/contract code shared
- README/docs explain how to add new recipes/modules

## 13) Recommended Immediate Start

Start with Phase 1 + Phase 2 for lowest risk:

- Build scaffold
- Port Vulkan soft-shadow path as the first vertical slice
- Keep current behavior 1:1 during migration

This gives a robust base to scale into more combinations safely.
