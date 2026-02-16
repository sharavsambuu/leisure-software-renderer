# Render Path Core Draft and Current State

Last updated: 2026-02-16

## 1) Purpose

This note is a handoff draft for moving from demo-specific render-path logic to reusable `shs-renderer-lib` infrastructure.

Goals:

- keep path composition dynamic across architectures (`Forward`, `Forward+`, `Deferred`, `TiledDeferred`, `ClusteredForward`)
- make pass/resource wiring data-driven instead of demo-wired
- enable reusable, header-declared path presets for demos and apps

## 2) Current State Snapshot

Compositional readiness level: `L4` for pass-chain orchestration
(shared dispatcher + plan-driven required/optional execution), with some pass bodies still demo-owned.

### 2.1 What is already done

- `HelloRenderingPaths` runtime path cycling is active and stable.
- `HelloRenderingPaths` now supports explicit combined composition cycling
  (`F3`: `{path + technique}`), while retaining manual path/technique overrides.
- `HelloRenderingPaths` deferred chain now has explicit stage handlers in pass dispatch
  (`gbuffer` and `deferred_lighting`) with telemetry-visible state in title.
  Deferred lighting is currently marked as emulated until dedicated gbuffer/deferred pipelines land.
- `HelloPassBasics` (software) now uses shared render-path presets/executor as runtime control:
  - `F2` cycles `Forward/Forward+/Deferred/TiledDeferred/ClusteredForward`
  - pass chain is configured from compiled `RenderPathExecutionPlan`
  - invalid compile/config falls back to technique-profile pipeline
- `hello_soft_shadow_culling_sw` now uses the same shared render-path executor runtime:
  - built-in software presets are applied through `RenderPathExecutor`
  - `F2` cycles render paths and updates runtime defaults (occlusion/debug/lit/shadow enable)
  - invalid apply falls back to legacy soft-shadow defaults
- render-path recipe/registry/compiler scaffolding exists.
- shared preset builder + runtime selector/executor layer exists in:
  - `cpp-folders/src/shs-renderer-lib/include/shs/pipeline/render_path_presets.hpp`
  - `cpp-folders/src/shs-renderer-lib/include/shs/pipeline/render_path_executor.hpp`
- shared rendering-technique preset layer exists in:
  - `cpp-folders/src/shs-renderer-lib/include/shs/pipeline/render_technique_presets.hpp`
- shared path+technique composition preset layer exists in:
  - `cpp-folders/src/shs-renderer-lib/include/shs/pipeline/render_composition_presets.hpp`
- shared resource-plan compiler exists in:
  - `cpp-folders/src/shs-renderer-lib/include/shs/pipeline/render_path_resource_plan.hpp`
  - it compiles pass/resource read-write contracts from a `RenderPathExecutionPlan`
    into a reusable resource/binding layout plan.
- shared standard pass-contract lookup + contract-only registry exists in:
  - `cpp-folders/src/shs-renderer-lib/include/shs/pipeline/pass_contract_registry.hpp`
  - this is now used by `HelloRenderingPaths` when compiling/validating recipe plans.
- shared runtime light-grid/tile allocation layout helper exists in:
  - `cpp-folders/src/shs-renderer-lib/include/shs/pipeline/render_path_runtime_layout.hpp`
  - `HelloRenderingPaths` now allocates tile/light-grid buffers from that shared layout,
    using active recipe + resource-plan data.
- shared Vulkan render-path global descriptor helper exists in:
  - `cpp-folders/src/shs-renderer-lib/include/shs/rhi/drivers/vulkan/vk_render_path_descriptors.hpp`
  - `HelloRenderingPaths` now reuses this for descriptor set layout/pool/update contract
    instead of embedding raw descriptor wiring in demo code.
- shared render-path pass-chain dispatcher exists in:
  - `cpp-folders/src/shs-renderer-lib/include/shs/pipeline/render_path_pass_dispatch.hpp`
  - `HelloRenderingPaths` now dispatches frame passes via active `RenderPathExecutionPlan`
    instead of hardcoded pass if/else sequencing in `draw_frame`.
- pass semantic contracts were upgraded with explicit representation metadata:
  - space
  - encoding
  - lifetime
  - temporal role
- legacy compatibility semantics (`GBufferA/B/C`) were removed from core pass semantic enums.
- software deferred adapter IO names were aligned to semantic names:
  - `technique.albedo`
  - `technique.normal`
  - `technique.material`
- typed pass identity (`PassId`) is now part of:
  - technique profiles
  - render-path recipes
  - compiled pass-chain plans
  - pass dispatch registration/execution
- `PluggablePipeline` now uses typed pass IDs in core runtime APIs:
  - `add_pass_from_registry(PassId)`
  - `find(PassId)`
  - `set_enabled(PassId, bool)`
  - typed pass timing tags (`shadow/pbr/tonemap/motion_blur`) instead of string-compare branches
- `hello_soft_shadow_culling_sw` render-plan feature gating (`depth_prepass` / `shadow_map`)
  now checks `PassId` instead of pass-name literals.
- `HelloPassPlumbing` now registers standard passes through `PassId` calls
  (only custom/non-standard passes stay string-based).
- `compile_render_path_resource_plan(...)` now performs stricter semantic mismatch validation
  (representation/lifetime mismatch is treated as plan error).
- path-owned light-culling mode switching is in place (manual culling overrides removed).
- light debug wireframe remains available.
- lighting technique selection reduced to `PBR` and `Blinn`.
- core composition control mapping is now consistent in path-focused demos:
  - `F2`: render path
  - `F3`: composition (`path + technique`)
  - `F4`: technique
- light influence range and debug volumes were aligned closer to shape/range behavior.
- `HelloRenderingPaths` now consumes recipe/preset resource knobs (`light_tile_size`, `cluster_z_slices`)
  instead of fixed compile-time constants.

Primary integration file:

- `cpp-folders/src/exp-plumbing/hello_rendering_paths.cpp`

Existing composition files:

- `cpp-folders/src/shs-renderer-lib/include/shs/pipeline/render_path_recipe.hpp`
- `cpp-folders/src/shs-renderer-lib/include/shs/pipeline/render_path_registry.hpp`
- `cpp-folders/src/shs-renderer-lib/include/shs/pipeline/render_path_compiler.hpp`
- `cpp-folders/src/shs-renderer-lib/include/shs/pipeline/technique_profile.hpp`

### 2.2 What is still demo-specific

- attachment/buffer layout ownership is still mostly in demo code
- descriptor layout/bindings are still not fully pass-contract-driven
- pass dependencies are implicit in control flow, not explicit resource graph edges
- history resources (for temporal effects) are not first-class graph resources

## 3) Must-Do Actions (to become demo-independent)

1. Add pass contracts (`inputs`, `outputs`, resolution class, history usage, backend requirements).
2. Add render-graph resource planner (lifetime, aliasing, resize classes).
3. Add pass execution registry in library (not demo-local switch trees).
4. Move frame data contracts into shared core (`camera`, `scene`, `lights`, `cull`, `stats`).
5. Route descriptor/pipeline binding through pass-owned setup and graph-provided resource views.
6. Keep demos thin: only scene setup, controls, and visualization.

## 4) Header-Level API Draft (for shs-renderer-lib)

This is the proposed draft API shape for implementation in later sessions.

```cpp
// include/shs/pipeline/render_graph_contract.hpp
namespace shs {

enum class ResourceClass : uint8_t { Full, Half, Quarter, Custom };
enum class ResourceKind : uint8_t { Color, Depth, Normal, Velocity, Material, SSAO, History };

struct ResourceDesc {
    std::string id;
    ResourceKind kind = ResourceKind::Color;
    ResourceClass klass = ResourceClass::Full;
    uint32_t custom_w = 0;
    uint32_t custom_h = 0;
    uint32_t format = 0;   // backend-specific enum bridge
    bool sampled = true;
    bool storage = false;
    bool transient = false;
    bool history = false;
};

enum class AccessMode : uint8_t { Read, Write, ReadWrite };

struct PassResourceUse {
    std::string resource_id;
    AccessMode access = AccessMode::Read;
    bool required = true;
};

struct PassContract {
    std::string pass_id;
    std::vector<PassResourceUse> inputs;
    std::vector<PassResourceUse> outputs;
    bool needs_depth_prepass = false;
    bool needs_history = false;
};

struct RenderPathContract {
    std::string name;
    TechniqueMode mode = TechniqueMode::Forward;
    std::vector<PassContract> passes;
};

} // namespace shs
```

```cpp
// include/shs/pipeline/render_path_preset.hpp
namespace shs {

enum class RenderPathPreset : uint8_t {
    Forward,
    ForwardPlus,
    Deferred,
    TiledDeferred,
    ClusteredForward,
    ForwardPlusSSAO,
    DeferredSSAOAA,
    DeferredSSAOMBDoF
};

struct RenderPathPresetDesc {
    RenderPathPreset id = RenderPathPreset::Forward;
    RenderPathRecipe recipe{};
    RenderPathContract contract{};
};

const RenderPathPresetDesc* find_render_path_preset(RenderPathPreset id);

} // namespace shs
```

```cpp
// include/shs/pipeline/render_path_executor.hpp
namespace shs {

struct RenderPathExecutionContext {
    FrameSceneData* scene = nullptr;
    FrameCameraData* camera = nullptr;
    FrameLightData* lights = nullptr;
    FrameCullData* cull = nullptr;
    FrameStats* stats = nullptr;
};

class IRenderPathPass {
public:
    virtual ~IRenderPathPass() = default;
    virtual const char* id() const = 0;
    virtual bool setup(const PassContract&) = 0;
    virtual bool execute(RenderPathExecutionContext&) = 0;
};

class RenderPathExecutor {
public:
    bool compile(const RenderPathRecipe&);
    bool build_resources(uint32_t width, uint32_t height);
    bool execute_frame(RenderPathExecutionContext&);
};

} // namespace shs
```

## 5) Suggested Preset Catalog (first stable set)

`Phase A`:

- `Forward`
- `ForwardPlus`
- `Deferred`

`Phase B`:

- `TiledDeferred`
- `ClusteredForward`
- `ForwardPlusSSAO`

`Phase C`:

- `DeferredSSAOAA`
- `DeferredSSAOMBDoF`

## 6) Migration Order (low risk)

1. Keep `HelloRenderingPaths` as validation host.
2. Introduce pass contracts for existing passes only (no new effects yet).
3. Move existing resource allocation into graph planner.
4. Flip one path to executor-driven (`ForwardPlus`) and verify parity.
5. Add `Deferred` through the same executor.
6. Add `SSAO`, then `AA`, then `MotionBlur`, then `DoF`.

## 7) Session Handoff Checklist

Before starting implementation in the next session:

1. Re-read `docs/dynamic-render-path-composition-plan.md`.
2. Re-read `docs/vulkan-backend-status-and-reader-guide.md`.
3. Re-read `docs/modern-rendering-maturity-roadmap.md`.
4. Inspect `cpp-folders/src/exp-plumbing/hello_rendering_paths.cpp` current path switch and control map.
5. Confirm `RenderPathRecipe` + `RenderPathCompiler` constraints to avoid bypassing validation.
6. Start with `Phase A` deferred baseline work under existing composition runtime.
