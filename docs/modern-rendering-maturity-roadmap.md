# Modern Rendering Maturity Roadmap

Last updated: 2026-02-16

## 1) Purpose

This roadmap defines how to move from current dynamic render-path composition into production-style modern rendering maturity.

Companion implementation guide for custom extensions:

- `docs/custom-render-path-and-technique-extension-guide.md`

Target outcome:

- reusable and configurable render architecture presets (not demo-specific code paths)
- stable runtime switching between major rendering paths
- clean integration path for modern effects (`SSAO`, `TAA`, `DoF`, `Motion Blur`, `AA`)

## 2) Current Baseline

The project is already strong on composition foundation:

- shared render-path presets + executor
- shared path+technique composition presets
- runtime path/composition controls in core demos
- standardized core controls in path demos:
  - `F2`: render path
  - `F3`: composition (`path + technique`)
  - `F4`: technique

Current maturity interpretation:

- `L4` for pass-chain orchestration and recipe-driven dispatch
- temporal core has shared Vulkan history-color ownership; remaining maturity work is mainly
  post-stack parity and backend robustness/perf baselining

Phase-A status note (current implementation):

- `HelloRenderingPaths` now executes explicit deferred-stage dispatch handlers (`gbuffer`, `deferred_lighting`) with runtime telemetry in title.
- deferred lighting now runs via native gbuffer + deferred-lighting pipelines in `HelloRenderingPaths`.
- semantic contract model was upgraded (breaking change) to include:
  - representation space (`screen/view/light/tile`)
  - encoding (`linear/srgb/depth/velocity/index/count`)
  - lifetime (`transient/persistent/history`)
  - temporal role (`current/history_read/history_write`)
- resource-plan compilation now validates semantic representation/lifetime mismatches early.
- `HelloRenderingPaths` now has semantic-debug target cycling (`F8`) driven by active resource-plan semantics.
- legacy compatibility semantics (`GBufferA/B/C`) were removed from core contract enums;
  deferred contracts now use explicit semantics (`Albedo`, `Normal`, `Material`).
- standard pass orchestration now has typed `PassId` in core structures
  (`recipe/profile/compiled-plan/dispatcher`), while preserving readable names.
- Phase-B kickoff: standard pass routing (`PassId -> handler`) is now extracted to
  shared helper `render_path_standard_pass_routing.hpp`, and `HelloRenderingPaths`
  now binds standard passes through that reusable routing layer.
- Vulkan pass-dispatch context contract is now shared in renderer-lib via
  `vk_render_path_pass_context.hpp`; `HelloRenderingPaths` consumes that shared
  context type instead of owning a local pass-context struct.
- Vulkan standard pass execution flow is now shared via
  `vk_standard_pass_execution.hpp` for these passes:
  `shadow_map`, `depth_prepass`, `light_culling`, `scene`, `gbuffer`, `deferred_lighting`.
- Phase-C scaffold started:
  - shared temporal helpers (`render_path_temporal.hpp`) now provide Halton jitter and projection-jitter utilities
  - `HelloRenderingPaths` integrates optional projection jitter runtime toggle (`F9`) without changing default output

### 2.1 Semantics Contract Baseline (Canonical Rules)

Current canonical semantic model for render-path composition:

- Pass identity:
  - canonical runtime identity for standard passes is `PassId`
  - text IDs are treated as human-readable labels and compatibility input, not source-of-truth
- Resource semantics:
  - canonical meaning is `PassSemantic` + representation metadata
  - metadata fields in contracts are mandatory for compatibility:
    - `space`
    - `encoding`
    - `lifetime`
    - `temporal_role`
- Validation rules:
  - producer/consumer mismatch on representation (`space/encoding`) is a compile-time plan error
  - producer/consumer mismatch on `lifetime` is a compile-time plan error
  - missing required pass factory remains a hard error in strict validation mode
- Naming rules:
  - legacy aliases (`GBufferA/B/C`) are banned from new contracts
  - deferred semantic channels use explicit names (`Albedo`, `Normal`, `Material`)
  - software deferred IO keys follow semantic naming (`technique.albedo`, `technique.normal`, `technique.material`)
- Migration intent:
  - continue removing remaining string-literal pass checks from older demo paths
  - keep fallback state naming neutral (`safe defaults`) instead of `legacy` terminology

## 3) What "Modern Maturity" Means Here

A mature state for this renderer means:

1. Render architecture is declared by data recipes, not hardcoded demo logic.
2. Resource lifetime/aliasing/history are first-class graph concepts.
3. Major paths (`Forward`, `Forward+`, `Deferred`, `Tiled`, `Clustered`) share common module contracts.
4. Temporal and post effects are composable optional modules.
5. Backends execute a shared plan contract with deterministic validation and profiling.

## 4) Workstreams

### W1: Graph + Resource System

- extend pass/resource contract to include:
  - history resources
  - transient aliasing classes
  - optional/required quality tiers
- make resource planner own lifetime and resize policy

### W2: Path Module Ownership

- move remaining demo-owned pass bodies into reusable modules
- isolate backend-specific command recording behind shared pass interface

### W3: Temporal Infrastructure

- frame history registry (color/depth/velocity/history IDs)
- camera jitter contract
- motion-vector validity contract

### W4: Shader + Material Scalability

- shared BRDF/shadow/post common shader modules
- bounded permutation strategy (features as recipe toggles)
- material contract that does not depend on one demo path

### W5: Backend Execution + Perf

- explicit pass dependency/barrier rules from plan
- multi-frame ownership progress (2+ frames in flight where safe)
- GPU timing and pass marker instrumentation

### W6: Validation + Regression

- recipe validation CLI/check mode
- image/perf baselines per preset
- fallback rules with explicit warnings

## 5) Implementation Phases (Modern Maturity Plan)

### Phase A: Deferred Baseline Under Current Composition

Scope:

- create stable `Deferred` baseline path using current scene/light/culling contracts
- include:
  - `GBuffer`
  - deferred light accumulation
  - tonemap output

Exit criteria:

- runtime switch parity with existing path controls
- no demo-specific branch needed for deferred in the host loop
- clear telemetry in title/logs identifies active path and pass chain

### Phase B: Forward+ and Deferred Module Parity

Scope:

- normalize shared modules used by both paths:
  - depth prepass
  - shadow map
  - light culling
  - shading pass
  - debug overlay

Exit criteria:

- both paths run through shared module registration/execution contracts
- host demo code only controls scene/input/debug, not pass sequencing

Current progress:

- shared standard pass routing is now centralized and reused by `HelloRenderingPaths`
  (host no longer duplicates per-`PassId` registration wiring).
- shared Vulkan pass-context contract is centralized (host no longer owns context schema).
- shared Vulkan pass execution flow helpers now own control-flow for core passes;
  `HelloRenderingPaths` primarily provides backend callbacks/resources.
- next work is to push command-resource ownership into reusable module interfaces
  (not host members), then reuse the same execution helpers in another Vulkan host
  for parity proof.

### Phase C: Temporal Core

Scope:

- add history resources and jitter pipeline contract
- add first temporal consumer (`TAA` baseline)

Exit criteria:

- history is graph-managed, not ad-hoc global buffers
- `TAA` can be enabled/disabled by composition recipe

Current progress:

- temporal jitter math/utilities are centralized in shared pipeline helpers
- Vulkan temporal history-color resource ownership is now centralized in shared helper:
  - `cpp-folders/src/shs-renderer-lib/include/shs/rhi/drivers/vulkan/vk_render_path_temporal_resources.hpp`
  - `HelloRenderingPaths` now consumes that shared owner instead of host-local history image state.
- deferred shading now samples history color and applies configurable temporal blend.
- frame-end history update is now recorded via explicit swapchain->history copy when
  transfer-src is supported (runtime fallback state is surfaced in title/controls).
- typed `PassId::TAA` is now integrated into:
  - pass IDs
  - technique profiles (`Deferred`, `TiledDeferred`)
  - standard pass contracts/registry
  - standard dispatch routing (VK host + software pass factory)
- temporal accumulation enablement is now recipe/pass-chain driven (TAA pass presence),
  with `F9` acting as runtime enable/disable for the active TAA-capable path.
- remaining Phase-C ownership gap (host-local history resource state) is closed for Vulkan path hosts.

### Phase D: Post-Process Stack Composition

Scope:

- add modular post passes:
  - `SSAO`
  - `Motion Blur`
  - `Depth of Field`
  - `AA` variants (`FXAA`/`TAA`)

Exit criteria:

- effects can be composed by recipe without path-specific rewrites
- pass dependency/order verified by plan compiler

Current progress:

- `PassId::SSAO` is now wired into `HelloRenderingPaths` Vulkan execution path:
  - dedicated AO render target + SSAO render pass/pipeline
  - typed pass dispatch handler execution between `GBuffer` and deferred lighting
  - deferred lighting now consumes SSAO texture (combined with material AO)
- frame telemetry now reports deferred-stage pass state including SSAO (`def:[g/a/l/t]`).
- `Motion Blur` and `Depth of Field` are now implemented as real typed post passes in
  `HelloRenderingPaths` Vulkan path:
  - deferred output can route into post ping-pong targets
  - motion blur pass reads post input and writes either post target or swapchain
  - depth-of-field pass reads post input + depth and writes final swapchain output
  - title telemetry now exposes full deferred/post chain state as `def:[g/a/l/t/m/d]`.
- composition presets now include explicit Phase-D post-stack variants via shared preset layer:
  - `RenderCompositionPostStackPreset::{Default, Minimal, Temporal, Full}`
  - shared helpers resolve per-path support and runtime enablement for `SSAO/TAA/MotionBlur/DoF`.
- `HelloRenderingPaths` now consumes curated Phase-D composition catalog
  (`make_phase_d_render_composition_recipes`) and applies post-stack state from composition recipe.
- pass execution now honors composition-level post-stack enablement:
  - disabled `SSAO` clears AO target to neutral (`1.0`) to keep deferred lighting stable
  - disabled `TAA/MotionBlur/DoF` are skipped without host-side path rewrites
  - debug availability (`F5`) and title telemetry use effective post-stack state, not only raw path capability.
- post-stack composition variants are now compiled as real recipe pass-chain variants
  (not runtime-only boolean masking):
  - `resolve_builtin_render_composition_recipe(...)` now filters path pass chains by
    `RenderCompositionPostStackPreset` (`SSAO/TAA/MotionBlur/DoF`)
  - `HelloRenderingPaths` applies resolved composition recipes through
    `RenderPathExecutor::apply_recipe(...)`, so active plan/resource validation is variant-specific.
- composition catalog snapshots now include per-variant compile/resource validation status and
  explicit post-stack pass coverage (`s/t/m/d`) at startup and `F10`.

### Phase E: Backend Robustness and Performance Maturity

Scope:

- improve backend execution robustness for larger pass graphs
- advance multi-frame ownership and profiling infrastructure

Exit criteria:

- stable behavior across resize/rebuild and long sessions
- reliable pass-level timing and regression comparison

Current progress:

- shared render-path pass dispatcher now records per-pass CPU timing samples.
- dispatcher result now exposes total dispatch CPU time and slowest pass metadata.
- `HelloRenderingPaths` now records per-pass GPU timestamp queries (when supported:
  Vulkan `1.2+` + queue timestamp support) using per-frame-slot Vulkan query pools.
- title telemetry now reports both CPU dispatch timing and GPU pass timing summaries
  (`cpu:... slow:...`, `gpu:... slow:...`).
- composition catalog is printed at startup and via `F10` so path+technique
  coverage remains visible during parity/perf sessions.
- GPU pass timing fallback behavior is now hardened:
  - preserves pending query frames when `vkGetQueryPoolResults` returns `VK_NOT_READY`
  - explicitly tracks and reports timing state (`disabled/recording/submitted/query-pending/ready/zero-sample/...`)
  - excludes failed/invalid pass samples from totals; marks `gpu` timing valid only when at least one sample is valid.
- long-session/rebuild telemetry is now surfaced live in title:
  - swapchain-generation change count
  - render-target rebuild count + last rebuild reason
  - pipeline rebuild count + last rebuild reason

Phase-E remaining completion gap:

- automate reproducible perf/visual regression capture (not manual observation)
- add soak/rebuild stress harness to validate long-session stability under composition cycling

## 6) Initial Preset Targets (Practical Set)

Use this as the first curated set after Phase B/C:

- `forward_pbr_baseline`
- `forward_blinn_baseline`
- `forward_plus_pbr_baseline`
- `forward_plus_blinn_baseline`
- `deferred_pbr_baseline`
- `deferred_blinn_baseline`
- `tiled_deferred_pbr_baseline`
- `tiled_deferred_blinn_baseline`
- `clustered_forward_pbr_baseline`
- `clustered_forward_blinn_baseline`
- `deferred_ssao_taa`
- `forward_plus_pbr_motion_blur`

Optional later presets:

- `deferred_ssao_dof_taa`
- `clustered_forward_pbr_taa`

### 6.1 Core Composition Coverage (Must-Not-Forget Matrix)

Primary rendering-path architectures that should always stay covered:

- `Forward`
- `Forward+`
- `Deferred`
- `Tiled Deferred`
- `Clustered Forward`

Lighting-technique baselines per architecture:

- `PBR`
- `Blinn-Phong`

Minimum matrix for regression/perf parity:

- `5` paths x `2` techniques = `10` baseline compositions.

## 7) Must-Do Design Constraints

1. Keep composition declarative (recipe-driven) and avoid new demo-local toggles as architecture controls.
2. Keep fallback behavior explicit and logged when a recipe is partially unsupported.
3. Avoid unbounded shader permutations; gate features by curated recipe sets.
4. Keep SW and VK parity at contract level, even if pass internals differ.
5. Do not merge temporal features before motion/depth validity contracts are verified.

## 8) Risks and Mitigation

- Risk: feature explosion and unstable combinations
  - Mitigation: curated presets + capability rules
- Risk: temporal artifacts due to weak history contracts
  - Mitigation: strict validity checks and fallback to non-temporal path
- Risk: backend divergence
  - Mitigation: shared contract tests and per-path parity checks

## 9) Immediate Next Session Plan

1. Start cross-platform baseline capture for completed `Phase F/G/H/I`:
   generate Linux/WSL + native host reference artifacts.
2. Add CI-level gating for baseline/perf drift:
   fail PRs when configured frame/gpu/rebuild thresholds regress.
3. Begin next maturity phase:
   framegraph-level visual debug views (gbuffer/velocity/depth/post intermediates) for standardized diagnosis.

## 10) Definition of "Modern Maturity Reached"

This phase is considered complete when:

- at least two major architectures (`Forward+` and `Deferred`) are fully recipe-driven
- at least one temporal effect (`TAA`) and one post effect (`SSAO` or `DoF`) are recipe-composed
- demo hosts no longer own architecture-specific pass sequencing
- path swaps are reproducible, validated, and benchmarkable with shared telemetry

## 11) Phase Status + Next Phases

Current status snapshot:

- `Phase A`: done
- `Phase B`: done (for `HelloRenderingPaths` Vulkan path host)
- `Phase C`: done (Vulkan temporal core ownership + TAA contract integration)
- `Phase D`: done (recipe-compiled post-stack composition variants)
- `Phase E`: done (runtime robustness + telemetry integration)
- `Phase F`: done (baseline matrix harness + threshold-based regression comparator)
- `Phase G`: done (timed soak harness + acceptance verdict and thresholds integrated)
- `Phase H`: done (graph-owned barriers + layout-transition-heavy swapchain transfer chains moved to shared Vulkan helpers)
- `Phase I`: done (contract parity + machine-readable report + software runtime sampling parity artifact)

Phase details and next phases:

### Phase F: Baseline Matrix and Regression Artifacts

Scope:

- capture reproducible CPU/GPU timings per pass and per composition
- capture visual snapshots for baseline compositions and selected post-stack variants
- store results as machine-readable artifacts for diffing between commits

Exit criteria:

- one command generates timing + image artifacts for the curated matrix
- clear pass/fail thresholds for perf and visual drift

Current progress:

- `HelloRenderingPaths` now supports an automated `Phase-F` benchmark mode via env toggles
  (`SHS_PHASE_F=1`) that:
  - runs a curated composition matrix with warmup/sample windows
  - emits machine-readable JSONL metrics per composition
  - can capture per-composition Vulkan swapchain snapshots as `.ppm`
  - auto-advances compositions and exits when the matrix run completes
- default output target:
  - `artifacts/phase_f_baseline_metrics.jsonl`
- default snapshot target:
  - `artifacts/phase_f_snapshots/*.ppm`

Phase F completion note:

- Exit criterion #1 is satisfied by built-in benchmark mode.
- Exit criterion #2 should be gated by project-level regression thresholds in CI/analysis tooling
  (not shell-script-specific workflow).

Phase F operational notes:

- Build:
  - `TMPDIR=/tmp cmake --build cpp-folders/build`
- Baseline generation:
  - `SHS_PHASE_F=1 ./cpp-folders/build/src/exp-plumbing/HelloRenderingPaths`
  - outputs:
    - `artifacts/phase_f_baseline_metrics.jsonl`
    - `artifacts/phase_f_snapshots/*.ppm`
- Reference vs candidate generation:
  - `SHS_PHASE_F=1 SHS_PHASE_F_OUTPUT=artifacts/phase_f_reference.jsonl SHS_PHASE_F_SNAPSHOT_DIR=artifacts/phase_f_reference_snaps ./cpp-folders/build/src/exp-plumbing/HelloRenderingPaths`
  - `SHS_PHASE_F=1 SHS_PHASE_F_OUTPUT=artifacts/phase_f_candidate.jsonl SHS_PHASE_F_SNAPSHOT_DIR=artifacts/phase_f_candidate_snaps ./cpp-folders/build/src/exp-plumbing/HelloRenderingPaths`
- Key environment controls:
  - `SHS_PHASE_F_WARMUP_FRAMES` (default: `90`)
  - `SHS_PHASE_F_SAMPLE_FRAMES` (default: `180`)
  - `SHS_PHASE_F_CAPTURE_SNAPSHOTS` (default: `1`)
  - `SHS_PHASE_F_INCLUDE_POST_VARIANTS` (default: `1`)
  - `SHS_PHASE_F_FULL_CYCLE` (default: `0`)
  - `SHS_PHASE_F_MAX_ENTRIES` (default: `0`, unlimited)

### Phase G: Soak and Rebuild Reliability Harness

Scope:

- automated long-session run that cycles compositions/paths/techniques
- periodic resize/rebuild perturbation and temporal/debug toggle perturbation
- monitor for Vulkan validation errors, device loss, invalid-plan churn, and unexpected rebuild spikes

Exit criteria:

- soak run completes target duration without critical failures
- rebuild counters/reasons remain explainable and stable

Current progress:

- `HelloRenderingPaths` now supports env-driven Phase-G timed soak mode (`SHS_PHASE_G=1`) that:
  - auto-cycles compositions on frame intervals
  - perturbs temporal/debug toggles periodically
  - logs heartbeat/churn/rebuild events into JSONL and auto-exits at configured duration
- default output:
  - `artifacts/phase_g_soak_metrics.jsonl`
- soak end event now includes explicit acceptance verdict (`accept:true/false`) using configurable thresholds:
  - average frame time
  - rebuild deltas (render-target / pipeline / swapchain-generation)
  - composition cycle-apply failure count

Phase G operational notes:

- Build:
  - `TMPDIR=/tmp cmake --build cpp-folders/build`
- Soak run:
  - `SHS_PHASE_G=1 ./cpp-folders/build/src/exp-plumbing/HelloRenderingPaths`
  - output:
    - `artifacts/phase_g_soak_metrics.jsonl`
- Output events:
  - `phase_g_begin`
  - `phase_g_heartbeat`
  - `phase_g_cycle`
  - `phase_g_end`
- Key environment controls:
  - `SHS_PHASE_G_DURATION_SEC` (default: `180`)
  - `SHS_PHASE_G_CYCLE_FRAMES` (default: `240`)
  - `SHS_PHASE_G_LOG_INTERVAL_FRAMES` (default: `120`)
  - `SHS_PHASE_G_TOGGLE_INTERVAL_CYCLES` (default: `2`)
  - `SHS_PHASE_G_OUTPUT` (default: `artifacts/phase_g_soak_metrics.jsonl`)
  - `SHS_PHASE_G_ACCEPT_MAX_AVG_FRAME_MS` (default: `50`)
  - `SHS_PHASE_G_ACCEPT_MAX_RT_REBUILDS` (default: `24`)
  - `SHS_PHASE_G_ACCEPT_MAX_PIPELINE_REBUILDS` (default: `24`)
  - `SHS_PHASE_G_ACCEPT_MAX_SWAPCHAIN_GENERATION` (default: `24`)
  - `SHS_PHASE_G_ACCEPT_MAX_CYCLE_FAILURES` (default: `0`)

### Phase H: Graph-Owned Barriers and Resource Aliasing

Scope:

- move remaining ad-hoc barrier/layout decisions from host to graph-derived plan
- add transient aliasing classes and memory reuse policy in resource planner
- tighten pass dependency ownership in compiler/resource planner

Exit criteria:

- barrier/layout schedule is produced from plan metadata for core paths
- transient memory reuse is visible/measurable without breaking correctness

Current progress:

- shared barrier/alias planner added:
  - `cpp-folders/src/shs-renderer-lib/include/shs/pipeline/render_path_barrier_plan.hpp`
  - compiles access timelines, barrier edges, lifetime windows, and transient alias slots
- planner is now integrated into `RenderPathExecutor` as `active_barrier_plan`
  (compiled alongside `active_plan` and `active_resource_plan`)
- `HelloRenderingPaths` now surfaces barrier-plan telemetry:
  - composition catalog (`F10`) prints per-composition barrier/alias counts
  - title includes live barrier/alias counters
  - apply-time logging prints barrier-plan warnings/errors
- Vulkan barrier mapping now exists for graph edges:
  - `cpp-folders/src/shs-renderer-lib/include/shs/rhi/drivers/vulkan/vk_render_path_barrier_mapping.hpp`
- `HelloRenderingPaths` now uses graph-driven barriers across core chains:
  - `Depth -> LightCulling`
  - `LightCulling -> first light-grid consumer`
  - `GBuffer Depth/Albedo/Normal/Material -> first downstream consumer`
  - `SSAO AmbientOcclusion -> deferred consumer`
  - `DeferredLighting|DeferredLightingTiled -> first post/lighting consumer`
  - `MotionBlur -> DepthOfField` (`ColorLDR`)
  - safe fallback keeps manual barriers when edge mapping is unavailable
- Vulkan stage/access mapping was tightened:
  - read+write access now maps to combined graphics stages
  - depth semantics now map to depth/stencil test stages in mixed graphics passes
- layout-transition-heavy swapchain transfer chains are now centralized in shared helpers
  (`vk_render_path_temporal_resources.hpp`) and consumed by `HelloRenderingPaths` for:
  - swapchain -> history color copy
  - swapchain -> post source color copy
  - swapchain -> readback buffer snapshot copy

Phase H completion note:

- graph-owned barrier ownership is integrated for core pass dependencies
- transfer-chain layout ownership is no longer host-local ad-hoc logic

### Phase I: Cross-Backend Composition Parity (Vulkan + Software)

Scope:

- reuse composition semantics/contracts in software renderer host where feasible
- align capability fallback behavior and debug semantics across backends
- ensure composition naming and toggle behavior stays consistent

Exit criteria:

- same curated composition list runs on both backends (with explicit capability downgrades)
- parity report exists for feature support, fallback reasons, and debug targets

Current progress:

- standard contract-only pass registry now supports backend-constrained registration:
  - `make_standard_pass_contract_registry_for_backend(RenderBackendType)`
  - used to construct Vulkan and Software scoped pass registries for parity checks
- compiler now validates backend/mode support from pass factories during recipe compilation:
  - required pass that does not support the target backend is now a compile error
  - optional pass that does not support the target backend is downgraded to warning/skip
- `HelloRenderingPaths` composition catalog (`F10`) now prints backend parity status:
  - per composition: `vk` validity + `sw` validity + pass counts
  - first software-side plan error is printed when software parity fails
- `HelloRenderingPaths` now supports built-in Phase-I parity artifact export:
  - enable with `SHS_PHASE_I=1`
  - output path override: `SHS_PHASE_I_OUTPUT`
  - default output: `artifacts/phase_i_backend_parity.jsonl`
  - includes per-composition backend validity, pass counts, post-stack flags, and first fallback/error reasons
- built-in software runtime sampling is now included in Phase-I export (default enabled):
  - per-composition software execution/configuration validity
  - sampled software frame-time average
  - deterministic LDR hash for quick visual parity checks
  - runtime controls:
    - `SHS_PHASE_I_RUNTIME_SW` (`1` default)
    - `SHS_PHASE_I_RUNTIME_WARMUP_FRAMES` (default: `2`)
    - `SHS_PHASE_I_RUNTIME_SAMPLE_FRAMES` (default: `6`)
    - `SHS_PHASE_I_RUNTIME_WIDTH` / `SHS_PHASE_I_RUNTIME_HEIGHT` (defaults: `320` / `180`)

Phase I completion note:

- contract-level parity + runtime software execution sampling are both available in one artifact flow
- parity gaps now surface as explicit per-composition JSON fields (`sw_runtime_*`, `vk_*`, `sw_*`)
