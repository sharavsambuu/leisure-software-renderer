# Value-Oriented Programming First-Class Roadmap

This roadmap makes Value-Oriented Programming (VOP) a hard architectural rule in SHS renderer.

## Current Progress

- 2026-02-24: Started implementation.
- Added value-oriented input action/reducer API in `shs/input/value_actions.hpp`.
- Added core runtime input-latch reducer in `shs/input/value_input_latch.hpp`.
- Added core `PluggablePipeline` value execution planning (`PipelineExecutionPlan` + `build_execution_plan(...)`) and made execution consume the plan.
- Extended `PipelineExecutionPlan` with queue/label metadata consumed directly by Vulkan-like submission path.
- Added precomputed backend execution groups to `PipelineExecutionPlan` so backend-switch sequencing is planned, not decided during pass execution.
- Migrated core human/bot controller helpers to value-action APIs (`human_controller.hpp`, `bot_controller.hpp`).
- Added command-to-action conversion path in `ICommand` + `CommandProcessor` batch reducer flow.
- Added `shs_renderer_vop_tests` target with deterministic checks for reducers and pipeline planning.
- Migrated `HelloPassPlumbing` from command-object execution to value actions + reducer flow.
- Migrated `demo_forward_classic_renderpath` input/camera path to value actions + reducer flow.
- Migrated `demo_forward_classic_renderpath` discrete key controls to queued key actions applied in a per-frame reducer step.
- Migrated `demo_forward_classic_renderpath` movement/mouse input state to core input-latch reducer flow.
- Removed command mutation execution path (`execute_all`) and kept only value reduction (`collect_runtime_actions` + `reduce_all`).
- Hardened `ICommand` to strict value emission contract (`RuntimeAction to_runtime_action() const`).
- Simplified camera command classes to action-emission only.
- Extended `shs_renderer_vop_tests` with command-processor value reduction coverage.
- Updated migration docs with a prioritized remaining core backlog and a VOP-aligned C++20 modernization track.
- Added explicit pass execution-request value contract in `IRenderPass` (`build_execution_request(...)` + `execute_resolved(...)`) and migrated `PluggablePipeline` execution to consume it.
- Removed hidden mutable pass caches by moving them into explicit `Context` runtime state:
  - shadow mesh-bounds cache moved from `static` local to `Context::shadow.mesh_bounds_cache`
  - TAA history moved from adapter-owned members to `Context::temporal_aa`
- Added request-time transient RT resolution (`PassExecutionRequest` named handle map) and migrated key adapters (`depth_prepass`, `light_shafts`, `motion_blur`) to resolve temp targets in `build_execution_request(...)`.
- Removed execution-time transient allocation fallback from `depth_prepass`, `light_shafts`, and `motion_blur` adapter `execute(...)` paths by delegating legacy execution to the resolved request contract.
- Extended `shs_renderer_vop_tests` with pipeline request-gate coverage (invalid request skips pass execution).
- Added `PassFactoryRegistry` descriptor hints (contract + backend/mode support) and migrated render-path compiler/resource/barrier planners to use hints and standard contracts before pass instantiation fallback.
- Migrated `PluggablePipeline` profile/plan assembly to consult descriptor mode hints before pass creation, avoiding instantiation for known-incompatible technique modes.
- Introduced `PipelineRuntimeExecutor` and delegated side-effecting pass/backend submission flow from `PluggablePipeline::execute(...)` to the executor boundary.
- Removed `build_execution_plan(...)` dependency on `Context::forward_plus` runtime-validity flags for contract-gating, keeping planning independent from mutable cross-frame runtime state.
- Added explicit per-pass runtime capability state (`depth_prepass_ready`, `light_culling_ready`) in `PassExecutionRequest`, propagated by `PipelineRuntimeExecutor`, and migrated core lighting/culling adapters to consume request capabilities instead of `Context::forward_plus` validity flags.
- Removed obsolete `Context::ForwardPlusRuntimeState` validity flags (`depth_prepass_valid`, `light_culling_valid`).
- Migrated remaining standard pass adapters (`shadow_map`, `gbuffer`, `ssao`, `deferred_lighting`, `pbr_forward`, `tonemap`, `depth_of_field`, `taa`) to request-first execution (`execute(...)` delegates to `execute_resolved(...)`).
- Migrated contract-only pass implementations to request-first execution and added runtime coverage that execution uses `execute_resolved(...)`.
- Updated constitution spec with explicit VOP benefits for renderer settings/presets (determinism, technique/backend setting clarity, reproducible tuning/debug workflows).
- Added execution-proven runtime readiness (`PassExecutionResult`) so depth/light-culling readiness is advanced only by actual pass execution outputs.
- Removed legacy pass compatibility edge by making `IRenderPass::execute_resolved(...)` the sole execution interface contract.
- Removed planner-side `dynamic_cast` capability branching by lifting depth-attachment availability into explicit `BackendCapabilities` fields.
- Removed planner-time pass-instantiation fallback from render-path compiler/resource/barrier planners; planner paths now rely on standard contracts or `PassFactoryRegistry` descriptor hints.
- Extracted planning logic into dedicated `PipelineExecutionPlanner`, and made `PluggablePipeline::build_execution_plan(...)` delegate to that planner component.
- Extracted resize/on-resize side-effect orchestration into dedicated `PipelineResizeCoordinator`, further thinning `PluggablePipeline` runtime facade responsibilities.
- Removed `Context::forward_plus` mutable payload coupling by introducing request-scoped light-culling payload (`PassExecutionRequest.inputs.light_culling`) produced/consumed at runtime edge.
- Retired scene mutation wrapper `SceneObjectSet::sync_to_scene(...)`; active call sites now consume value snapshots directly (`scene.items = objects.to_render_items()`).
- Retired controller compatibility aliases (`emit_human_commands`, `emit_orbit_bot_commands`) in favor of value-action helpers (`emit_human_runtime_actions`, `emit_orbit_bot_runtime_actions`).
- Retired unused geometry compatibility umbrella header `shs/geometry/culling.hpp`; core uses direct canonical geometry headers.
- Retired unused light-culling alias `cull_lights_tiled_depth_range(...)`; canonical API remains `cull_lights_tiled_depth01_range(...)`.
- Added executable VOP boundary gate script (`tools/check_vop_boundaries.sh`) and CMake target `shs_renderer_vop_boundary_check` to block planner-side backend includes and `dynamic_cast` policy branching.
- Removed adapter-local legacy `execute(ctx, scene, fp, rtr)` helper methods from standard pass adapters; adapter execution surface is now `execute_resolved(...)` only.
- Modernized `PassContext` shared hubs from raw `void* + binding enums` to explicit typed pointer slots (`Scene*`, `ResourceRegistry*`, `RendererResources*`) with typed accessors.
- Removed `Context` single-backend compatibility pointer (`render_backend`) and standardized backend selection on explicit `backends[] + primary_backend` value state.
- Type-hardened `PassContext` camera payload pointers from `const void*` to explicit `const glm::mat4*` / `const glm::vec3*`.
- Added explicit planner compatibility-lane diagnostics for non-standard passes missing contract metadata; strict graph validation now fails those passes instead of silently accepting metadata-less planning inputs.
- Enabled strict graph validation as the default `PluggablePipeline` policy (opt-out only for migration/experimental paths).
- Started `exp-rendering-techniques` demo migration on top of VOP-first core by extracting per-frame input-latch -> `InputState` normalization into a pure helper and making Phase-I software runtime sampling pipeline explicitly strict-validated.
- Started `exp-plumbing` migration alignment: `HelloPassBasics` now enforces strict graph validation for render-path-plan configuration (no recipe-level strict opt-out), and `HelloRenderingPaths` phase-I software runtime sampling now explicitly enables strict graph validation.
- Migrated difficult `exp-plumbing` camera/input paths to reducer flow:
  - `HelloPassBasicsVulkan` now uses `PlatformInputState -> InputState -> RuntimeAction -> reduce_runtime_state(...)` for free-camera motion/look/quit.
  - `HelloRenderingPaths` now routes movement/look through the same reducer flow at `update_frame_data(...)`, while preserving existing event/toggle behavior.
- Completed VOP migration for remaining `exp-plumbing` demo input/control paths:
  - `HelloPassBasics` now uses reducer-driven free-camera updates (`InputState -> RuntimeAction -> RuntimeState`) instead of direct per-frame camera mutation.
  - `HelloRenderingPaths` now uses core input-latch reduction (`RuntimeInputLatch`) for movement/mouse state before action reduction.
  - Low-level Vulkan demos (`hello_vulkan_triangle`, `hello_mesh_shader`, `hello_modern_vulkan`, `hello_ray_query`) now consume quit intent through value input-latch + action reduction rather than direct immediate quit mutation.
  - `hello_jolt_integration` and `HelloSoftwareTriangle` now process quit via value actions/reducer flow for consistency with VOP-first runtime handling.
- Execution-proven check: updated demo targets compile successfully with `TMPDIR=/tmp cmake --build cpp-folders/build --target ...`; software lighting demos (`hello_culling_sw`, `hello_occlusion_culling_sw`, `hello_light_types_culling_sw`, `hello_soft_shadow_culling_sw`, `HelloPassBasics`) were rebuilt to verify lighting paths remain intact after input/control refactors.

## Remaining Work to Reach Full VOP-First (Core)

Priority order (core library first):

1. Finish pipeline planner/runtime extraction. (In progress)
   - Current state: `PipelineRuntimeExecutor` owns pass execution, `PipelineExecutionPlanner` owns plan assembly/validation, and `PipelineResizeCoordinator` owns resize side effects; `PluggablePipeline` still acts as orchestration facade.
   - Remaining work:
     - Keep `PluggablePipeline::execute(...)` as a thin facade over planner + runtime executor.
   - Exit condition: planner has no backend lifecycle/submission side effects.
2. Remove remaining runtime payload coupling in shared context. (Completed for light-culling payload)
   - Current state: `Context::forward_plus` payload removed from core; light-culling payload now flows as request-scoped runtime value.
   - Follow-up:
     - Keep future runtime payload additions request-scoped by default, not attached to shared context state.
3. Enforce descriptor/contract registration for planner participation. (In progress)
   - Current state: planner compilers no longer instantiate passes; they consume standard contracts or descriptor hints only, and planner runtime now emits explicit compatibility-lane diagnostics for non-standard passes with missing contract metadata.
   - Remaining work:
     - Keep strict-graph mode enabled in enforcement contexts so compatibility-lane passes fail fast.
   - Exit condition: every planner-visible pass provides descriptor/contract metadata by registration policy.
4. Retire remaining compatibility wrappers. (In progress)
   - Remaining work:
     - Deprecate and remove remaining mutation-era wrappers in non-core/legacy paths once all call sites are value-first.
   - Exit condition: core runtime path depends only on value contracts.
5. Add hard enforcement gates. (In progress)
   - Remaining work:
     - Wire CTest VOP gates (`shs_renderer_vop_boundary_check`, `shs_renderer_vop_tests`) into CI so planner-boundary and deterministic core checks fail automatically.
     - CI/static check: reject new planner-side `dynamic_cast` policy and mutable `static` caches.
     - Expand deterministic golden coverage for planner outputs and request/result runtime contracts.
   - Exit condition: VOP regressions are caught automatically.

## Full VOP-First Gate

Renderer is "fully VOP-first" only when all are true:

1. Planner code is pure/deterministic and isolated from side-effect execution.
2. Runtime is the only side-effecting layer.
3. Planner never instantiates passes for capability/contract discovery.
4. Compatibility wrappers are removed or constrained to external-only shims.
5. CI and deterministic tests enforce VOP boundaries continuously.

## C++20 Modernization Track (VOP-Aligned)

Apply C++20 features aggressively where they increase value-semantics clarity:

1. Standardize `std::span` for non-owning contiguous inputs in planner/reducer APIs.
2. Standardize `std::string_view` for pass/recipe IDs and key lookup APIs.
3. Introduce `concept`/`requires` constraints on planner conversion/build helpers.
4. Prefer `std::ranges` for deterministic pipeline/list transforms when allocation behavior stays explicit.
5. Expand `constexpr` mapping tables for pass/resource semantic conversions.
6. Keep coroutines runtime-edge only (job/task orchestration), not planner state mutation.
7. Remove planner-side `dynamic_cast` and mutable `static` caches.
   - In progress: removed backend policy `dynamic_cast` from render-path capability resolution; continue auditing for remaining planner-side dynamic type branches.

Review rule:

- For each refactor PR, include at least one C++20 uplift where suitable in touched files.

## North Star

Every frame follows:

1. `Input Actions`
2. `State Reducers`
3. `Scene/Frame Snapshots`
4. `Render Plan`
5. `Runtime Execution`

Only step 5 is side-effecting.

## Architectural Rules

1. Planning code must be pure and deterministic.
2. Runtime code must not decide policy; it only executes resolved plans.
3. Prefer value-returning APIs over mutation-based APIs.
4. Action logs must be replayable to reproduce planning outputs.

## Phase 1: Foundation Rules and Types

1. Promote VOP rules from guideline to mandatory standard.
2. Define canonical value contracts:
   - `Action` (input/render/ui commands as values)
   - `EngineState` (full app state snapshot)
   - `SceneSnapshot`
   - `FrameSnapshot`
   - `RenderPlan`
   - `PlanDiagnostics`

## Phase 2: Input and State Refactor

1. Replace polymorphic command pattern with value actions.
2. Introduce pure reducers:
   - `next_state = reduce(state, actions, dt)`
3. Remove hidden mutation from input processing paths.

## Phase 3: Render Planner Refactor

1. Make render path compilation fully value-based.
2. Make frame graph planning pure and deterministic.
3. Convert plan outputs into immutable plan bundles consumed by runtime.

## Phase 4: Runtime Boundary Hardening

1. Split pipeline APIs into:
   - pure planning entry points
   - side-effecting execution entry points
2. Restrict backend interactions (Vulkan/Software) to runtime boundary only.

## Phase 5: Demo Rewrite to VOP Flow

1. Refactor forward classic demo into action/reducer/planner/runtime stages.
2. Remove ad-hoc mutable orchestration from frame loop.
3. Ensure demo state changes are action-driven and replayable.

## Phase 6: Verification and Enforcement

1. Add deterministic golden tests for planners and reducers.
2. Add CI checks that planner layer has no backend calls.
3. Add code review checklist enforcing VOP boundaries.

## Phase 7: C++20 Uplift and Cleanup

1. Add C++20-first coding checklist to VOP reviews (`span/string_view/concepts/ranges/constexpr`).
2. Replace planner-side legacy patterns (`dynamic_cast`, hidden caches, ownership-opaque pointer switching).
3. Complete compatibility-wrapper retirement once core + demos consume value APIs end-to-end.

## Deliverables

1. Updated architecture docs and coding standards.
2. VOP-first engine APIs with runtime boundaries.
3. VOP-first forward classic demo.
4. Deterministic test coverage for reducers and planners.

## Exit Criteria

1. Planning layers are pure and reproducible.
2. Runtime is the only side-effecting layer.
3. Demos use value snapshots and action reducers end to end.
4. Legacy mutation-heavy paths are removed.
5. Planner/reducer layers avoid hidden mutable state (`static` cache, dynamic type branching).
6. Modified core modules consistently adopt suitable C++20 value abstractions.
