# SHS Renderer Constitution II: Value-Oriented Programming

This document is the second constitutional specification of SHS renderer.

- Constitution I: `docs/spec/conventions.md` (units, coordinate system, backend semantics)
- Constitution II (this document): Value-Oriented Programming (VOP)
- Roadmap: `docs/roadmap/value_oriented_programming_first_class_roadmap.md`

## 1. Purpose

VOP is adopted to make renderer behavior explicit, deterministic, and easier to optimize.

Expected outcomes:

- predictable frame planning and recipe compilation
- inspectable snapshots for replay/debug/testing
- lower hidden mutation and tighter data ownership
- cleaner multithreading boundaries

## 2. Constitutional Principle

Use pure value transforms by default. Keep side effects at boundaries.

- Value center:
  - scene/frame contracts
  - planning and dependency analysis
  - input/state reducers
  - compile-time or frame-time transformation logic
- Effect edges:
  - GPU submission
  - backend synchronization
  - resource lifetime operations
  - OS/input I/O integration points

## 3. Mandatory Rules

1. New planning APIs must return explicit value objects (structs) by value.
2. Mutating compatibility wrappers are allowed only when they delegate to value APIs.
3. Planning/reduction code must avoid hidden globals and hidden mutable state.
4. Side-effect APIs should accept already-resolved values, not recompute planning decisions internally.
5. Compile/plan/reduce stages must be deterministic for identical inputs.
6. Value objects must encode complete data contracts needed by execution edges.
7. Prefer C++20 value-centric language/library features when they improve clarity and safety without adding hidden cost.

## 4. Forbidden Patterns

1. Mixing planning and backend submission in the same method.
2. Hidden singleton state reads inside value reducers/planners.
3. Mutating global runtime state during what is documented as a planning pass.
4. Side-effect code that silently overrides resolved plan values.
5. Unstable output ordering in planner/reducer results when stable ordering is practical.

## 5. Allowed Exceptions

1. Allocation-sensitive hot paths may use output-buffer APIs (`out` params) when deterministic behavior is preserved.
2. Legacy APIs may remain during migration if they are thin wrappers over canonical value APIs.
3. Backend-specific fast paths may exist at effect edges if input/output contracts remain explicit values.

## 6. Module Directives

### Scene

- Canonical transform API: `SceneObjectSet::to_render_items()`
- Scene integration should consume `to_render_items()` explicitly at call sites (`scene.items = objects.to_render_items()`).

### Lighting

- Canonical transform API: `LightSet::to_cullable_gpu(...)`
- Allocation-sensitive alternative: `flatten_cullable_gpu(out, ...)`

### Render Path Planning

- Resolve state as values first:
  - `RenderPathExecutor::resolve_index(...)`
  - `RenderPathExecutor::resolve_recipe(...)`
- Apply at edge:
  - `RenderPathExecutor::apply_resolved(...)`

### Pipeline Orchestration

- Use `PipelineExecutionPlan` as the execution contract.
- Build plan in a pure planning stage, execute in a separate effect stage.

### Input/Runtime State

- Express runtime input as value actions and reducers.
- Keep latch/state transitions reducer-driven where practical.

## 7. C++20 Guidance (VOP-Aligned)

Recommended by default in new core APIs:

- `std::span` for non-owning contiguous data flow across reducers/planners.
- `std::string_view` for pass/recipe IDs and lookup keys where ownership is external.
- `concept`/`requires` for compile-time contracts in planning and conversion utilities.
- `std::variant` + `std::visit` for explicit action/state unions.
- `constexpr` helpers/tables for deterministic mapping logic.
- `std::ranges` algorithms/views in planning code only when they do not hide allocations or hurt readability.

Use with care:

- Coroutines for runtime scheduling only (effect edge), not for planner state mutation.
- Dynamic polymorphism for compatibility edges only; prefer value contracts in new planner paths.

Avoid in planner/reducer layers:

- mutable `static` local caches
- `dynamic_cast`-based policy decisions
- ownership-opaque raw-pointer switching

## 8. Compliance Checklist

Use this list for new renderer features and major refactors:

1. Is the feature split into value planning/reduction and side-effect execution?
2. Are planner/reducer inputs and outputs explicit structs?
3. Is deterministic behavior covered by tests for identical inputs?
4. Are compatibility mutators only wrappers over value APIs?
5. Does execution consume pre-resolved values rather than recomputing hidden state?
6. Do new/modified APIs use suitable C++20 value abstractions (`span`, `string_view`, concepts, variant) where practical?
7. Are runtime readiness transitions based on explicit execution results, not static capability claims?

## 9. Adoption Snapshot

- Added value transform for scene object conversion.
- Added value transform for light culling GPU payload generation.
- Added value-style render path resolution object (`RenderPathResolvedState`).
- Added value-style pipeline execution planning object (`PipelineExecutionPlan`) and plan builder in `PluggablePipeline`.
- Extended pipeline plan data with queue/label metadata for submission orchestration.
- Added precomputed backend execution groups in pipeline plans, reducing runtime backend-switch decisions.
- Migrated core human/bot controller helpers to value-action emission helpers.
- Added command-to-action conversion (`ICommand::to_runtime_action`) and command-processor batch reducer flow so command queues execute via value reductions.
- Removed command mutation execution path (`execute_all`) and standardized command processing on value collection/reduction (`collect_runtime_actions`, `reduce_all`).
- Hardened `ICommand` to strict action emission (`RuntimeAction to_runtime_action() const`) so command lanes cannot bypass value contracts.
- Added explicit pass execution-request value boundary (`IRenderPass::build_execution_request` + `execute_resolved`) and made pipeline runtime consume resolved pass requests.
- Removed hidden mutable caches from pass internals by moving shadow bounds cache and TAA history to explicit `Context` runtime state.
- Added request-time named transient-handle resolution in `PassExecutionRequest` and migrated key post/visibility adapters to use resolved temp RT handles at execution.
- Removed adapter-level execution fallback allocation for `depth_prepass`, `light_shafts`, and `motion_blur` so legacy `execute(...)` now delegates to request-resolved execution.
- Added pass-registry descriptor hints for planner-time contract/backend/mode checks, reducing planner pass-instantiation for standard and descriptor-enabled passes.
- Extended descriptor-hint usage to `PluggablePipeline` profile and render-path-plan assembly so known mode-incompatible passes are rejected before factory instantiation.
- Added explicit runtime execution boundary `PipelineRuntimeExecutor` and made `PluggablePipeline` delegate side-effecting pass/backend submission to that executor.
- Removed planning-time runtime-gating dependence on `Context::forward_plus` validity flags (`depth_prepass_valid`, `light_culling_valid`) so execution planning no longer depends on mutable per-frame runtime flags.
- Added explicit request-scoped runtime capability values (`depth_prepass_ready`, `light_culling_ready`) populated by runtime executor and consumed by forward+/cluster/tiled lighting-culling adapters; removed the old context validity flags.
- Added execution-result value contract (`PassExecutionResult`) so runtime readiness flips only when passes actually execute and report produced outputs (depth/light-grid/index-list), avoiding contract-claim-only readiness.
- Removed legacy pass compatibility edge by making `IRenderPass::execute_resolved(...)` the sole pass execution interface contract.
- Removed planner-side backend type branching (`dynamic_cast`) for depth-attachment policy by promoting depth-attachment knowledge into explicit backend capability values.
- Removed planner-time pass instantiation fallback in render-path compiler/resource/barrier planning; planner now requires standard contracts or registered descriptor hints.
- Added dedicated `PipelineExecutionPlanner` component and routed `PluggablePipeline::build_execution_plan(...)` through it to keep planning and runtime execution boundaries explicit.
- Added dedicated `PipelineResizeCoordinator` runtime-edge component for backend/pass resize side effects, reducing mixed responsibilities in `PluggablePipeline`.
- Replaced shared `Context::forward_plus` mutable payload with request-scoped runtime payload (`PassExecutionRequest.inputs.light_culling`) so light-culling tile data flows through explicit execution contracts.
- Unified standard pass adapters on request-first execution path (`execute_resolved(...)`) and removed dependency on legacy pass execution interfaces.
- Added guard test ensuring pipeline runtime executes passes through `execute_resolved(...)`.
- Added `shs_renderer_vop_tests` with deterministic checks for `reduce_runtime_state`, `reduce_runtime_input_latch`, and `build_execution_plan`.
- Extended VOP tests with deterministic command-processor value reduction coverage and pipeline request-gate execution checks.
- Forward classic render-path demo resolves path state as a value before applying it.
- Added value-oriented input actions/reducer (`shs/input/value_actions.hpp`) and migrated `HelloPassPlumbing` to use it.
- Migrated forward classic demo input/camera updates to value actions + reducer (`InputState -> RuntimeAction[] -> RuntimeState`).
- Added core runtime input-latch reducer (`shs/input/value_input_latch.hpp`) and migrated forward classic demo movement/mouse latching to reducer-based event application.
- Migrated remaining `exp-plumbing` demo control paths to VOP-first input/runtime flow:
  - `HelloPassBasics` free-camera motion/look now uses value actions + reducer (manual camera mutation removed).
  - `HelloRenderingPaths` now uses `RuntimeInputLatch` event reduction for movement/mouse state before runtime action emission.
  - Low-level demos (`hello_vulkan_triangle`, `hello_mesh_shader`, `hello_modern_vulkan`, `hello_ray_query`, `hello_jolt_integration`, `HelloSoftwareTriangle`) now route quit through value latch/action reduction instead of direct immediate mutation.
- Rebuilt software-lighting demos after migration (`hello_culling_sw`, `hello_occlusion_culling_sw`, `hello_light_types_culling_sw`, `hello_soft_shadow_culling_sw`, `HelloPassBasics`) to verify lighting implementations remain stable while input/control flow moved to VOP reducers.
- Retired scene mutator wrapper `sync_to_scene(...)` from `SceneObjectSet`; scene integration now uses explicit value snapshots from `to_render_items()`.
- Retired controller alias wrappers (`emit_human_commands`, `emit_orbit_bot_commands`) and kept value-action entrypoints only.
- Retired unused geometry compatibility umbrella header `shs/geometry/culling.hpp` and kept canonical direct geometry includes.
- Retired unused light-culling alias `cull_lights_tiled_depth_range(...)` to reduce duplicate API surface.
- Added automated boundary check script (`tools/check_vop_boundaries.sh`) and CMake target (`shs_renderer_vop_boundary_check`) to enforce planner-layer bans on backend-driver includes and `dynamic_cast`.
- Removed standard pass adapter helper execution wrappers (`execute(ctx, scene, fp, rtr)`) so adapter execution contracts stay request-first via `execute_resolved(...)`.
- Modernized `PassContext` resource/scene hubs to explicit typed pointers (removed raw `void*` binding pattern) and kept typed accessor APIs.
- Added explicit planner compatibility-lane diagnostics for non-standard passes that lack contract metadata; strict graph validation fails these passes.
- Enabled strict graph validation as the default `PluggablePipeline` mode; non-strict mode remains explicit opt-out for migration experiments.

## 10. Renderer Settings Benefits

VOP directly improves renderer settings behavior and maintainability:

1. Deterministic settings application
- Same `FrameParams`/technique settings produce the same plan and pass chain decisions.
- Presets become replayable and comparable without hidden state drift.

2. Cleaner technique switching
- Settings such as `technique.mode`, `active_modes_mask`, `depth_prepass`, and culling toggles map to explicit planning/execution values.
- Runtime no longer depends on implicit mutable validity flags for key depth/culling readiness.

3. Safer backend/hybrid configuration
- Backend-related settings (`strict_backend_availability`, cross-backend allowance, Vulkan-like emulation flags) are consumed through explicit plan/executor values.
- Policy decisions stay in planning; execution only consumes resolved contracts.

4. Better tuning workflow for performance settings
- Tile/cluster settings (`tile_size`, `max_lights_per_tile`) and pass enablement are easier to benchmark because decision paths are explicit.
- A/B testing of settings is simpler since action logs + settings snapshots are reproducible.

5. Improved debug and testability
- Settings regressions can be validated with deterministic tests around planner outputs and runtime request gates.
- Execution-path tests can verify contract behavior (`execute_resolved` path) independent from legacy wrappers.

6. Execution-proven readiness for advanced settings
- Technique toggles that depend on depth/light readiness now advance only from actual pass results, not static capability claims.
- This makes settings interactions around Forward+/Tiled/Clustered paths more trustworthy during performance tuning and backend bring-up.

## 11. Remaining Work to Become Fully VOP-First (As of February 24, 2026)

The constitution is mostly implemented in core, but full completion still requires:

1. Final planner/runtime extraction in pipeline facade
- Move remaining planning/validation concerns out of `PluggablePipeline` into a dedicated pure planner component.

2. Remove implicit mutable runtime payload coupling
- Completed for light-culling payload: shared mutable `Context::forward_plus` was removed and replaced by request-scoped runtime payload values.

3. Remove planner-time fallback pass instantiation
- Planner compilers already avoid pass instantiation; keep descriptor/contract registration policy enforced and run strict graph validation where planner hard guarantees are required.

4. Retire compatibility wrappers after migration
- Remove remaining mutation-era compatibility edges once all call sites are value-first.

5. Enforce boundaries by automation
- Added local automation target (`shs_renderer_vop_boundary_check`) and CTest registration for both boundary and VOP core tests (`shs_renderer_vop_tests`); next step is CI wiring plus broader static checks for hidden mutable/stateful patterns.
