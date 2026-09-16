# SHS Domain Glossary — The Renderer Author's Catalog

> Status: **Living vocabulary index (2026-09-15)**. The single entry point for
> "which struct/enum do I select when building a renderer?" per Constitution II §6.5.
> Each entry: concept → type → header → owning module.

## 1. Decision Axes (closed menus — pick one value each)

| You want to choose… | Select | Defined in |
| :--- | :--- | :--- |
| Rendering technique | `RenderPathRenderingTechnique` (`ForwardLit`, `ForwardPlus`, `Deferred`) | `shs/execution/pipeline/render_path_recipe.hpp` |
| Light volume provider | `RenderPathLightVolumeProvider` (`Default`, `JoltShapeVolumes`, `ClusteredGrid`) | `shs/execution/pipeline/render_path_recipe.hpp` |
| View / shadow culling | `RenderPathCullingMode` (`None`, `Frustum`, `FrustumAndOcclusion`, `FrustumAndOptionalOcclusion`) | `shs/execution/pipeline/render_path_recipe.hpp` |
| Target backend | `RenderBackendType` (`Software`, `Vulkan`, …) | `shs/execution/rhi/core/backend.hpp` |
| Post-processing stack | `RenderCompositionPostStackPreset` (`Default`, `Minimal`, `Temporal`, `Full`) | `shs/execution/pipeline/render_composition_presets.hpp` |
| Path preset (coarse) | `RenderPathPreset` (`Deferred`, `TiledDeferred`, …) | `shs/execution/pipeline/render_path_presets.hpp` |
| Technique mode (compat) | `TechniqueMode` (`Forward`, `ForwardPlus`, …) | `shs/execution/pipeline/render_path_recipe.hpp` |
| Per-pass participation | `RenderPathPassEntry` + `PassId` (`ShadowMap`, `DepthPrepass`, `LightCulling`, `PBRForward`, `PBRForwardPlus`, `Tonemap`, `MotionBlur`) | `shs/execution/pipeline/render_path_recipe.hpp` |

## 2. The Composite Values (structs you assemble or pick)

> **Pod home (P1):** the recipe → compiler → plan spine is owned by the
> `renderpath` Domain Pod — `include/shs/domains/renderpath/` (Core 4:
> `renderpath.contract.hpp` re-exports everything below, `renderpath.action.hpp`
> carries the closed `RenderPathCommand` variant, `renderpath.event.hpp` the
> closed `RenderPathEvent` variant, `renderpath.reducer.hpp` the pure
> `reduce_render_path`). The `shs/execution/pipeline/` paths below remain the
> canonical definition sites; the pod is the sanctioned seam (P1).

| Concept | Type | Notes |
| :--- | :--- | :--- |
| **A complete renderer definition** | `RenderPathRecipe` | technique + light volumes + culling + pass chain + knobs; pure value |
| Per-frame toggles (debug, shadows, lit mode) | `RenderPathRuntimeState` (`recipe.runtime_defaults`) | runtime-mutable via pod commands, not recipe edits |
| Post-stack feature flags | `RenderCompositionPostStackState` (ssao/taa/motion blur/dof) | composed from the post-stack preset |
| Compiled, validated path | `RenderPathExecutionPlan` | produced by `RenderPathCompiler` — never hand-written |
| What the reducer may ask | `RenderPathCompatibilityRules` | value rules threaded through `reduce_render_path` |
| What the device allows | `RenderPathCapabilitySet` | capability gating happens *before* any backend touch |
| Scene draw list | `RenderItemSpan` from `SceneObjectSet::to_render_items(view, proj, &arena)` | backend-neutral |
| Cullable light payload | flat GPU buffers from `LightSet::to_cullable_gpu(...)` | backend-neutral |
| Physics-derived culling/volume geometry | Jolt shape adapters (`shs/domains/geometry/jolt_shapes.hpp`, `jolt_culling.hpp`) | value-typed SHS data stays authoritative |

## 3. Named Recipes (the "pick one identifier" catalog)

| Identifier | Meaning |
| :--- | :--- |
| `make_default_soft_shadow_culling_recipe(backend)` | soft-shadow + occlusion culling default; auto-selects ForwardPlus chain on Vulkan, Forward chain on software |
| `RenderPathPreset::Deferred` / `::TiledDeferred` | coarse classic path selections |
| `RenderCompositionPostStackPreset::Full` | ssao + taa + motion blur + dof |

> Target state (post-P1): the catalog becomes `inline constexpr` named recipes in a
> presets header (`preset::DeferredPlusVulkan`, `preset::MobileForwardLite`, …), and a
> novel renderer = a preset + `with_*` deltas (Constitution II §6.5 rule 3–4).

## 4. Capability Predicates (ask, don't assume)

- `render_path_preset_supports_ssao(preset)` / `render_path_preset_supports_taa(preset)`
- `render_path_culling_requires_occlusion(mode)` / `render_path_culling_allows_occlusion(mode)`
- Reflection: `render_path_*_name(...)` for every menu (logging, HUD, replay).

## 5. Vocabulary Law Pointer

New vocabulary follows Constitution §6.5 (Vocabulary Law) — this catalog never
restates or re-numbers it (precedence: §2.2). When a pod's vocabulary changes,
update the relevant table above and the Constitution's §6.5 remains the single
source of truth for naming/factory/transform conventions.

## 6. Contiguous Backing-Store Utilities (P1.5)

> **Shared lib home (P1.5, §7.2 rule 6):** pod hot-state backing stores are
> owned by the lib primitive zones — `include/shs/memory/frame_memory_resource.hpp`
> (transient frame arena, §3 Rule 5.1) and `include/shs/containers/soa_table.hpp`
> + `flat_map.hpp` (generational column table / node-free keyed lookup).
> Demos and pods must not define private copies; keyed hot lookups use
> `FlatMap`, and dense column walks target `SoaTable::column<I>()` spans.

## 7. Pod homes (Core 4 complete 11/11 — hardening campaign R1–R5b)

Each pod: `<pod>.contract/action/event/reducer.hpp` (+ `plan.hpp` where the
litmus demands; edge code under `<pod>/edge/`). Event catalog:
`docs/pods/EVENT_FLOW.md` (drift-gated by `check_vop_boundaries.sh`).

| Pod | Home | Notes |
| :--- | :--- | :--- |
| `renderpath` | `shs/domains/renderpath/` | First formal pod; reducer wraps the compiler; invalid ⇒ keep + reject. |
| `input` | `shs/domains/input/` (+ `edge/` queue) | `reduce_input` canonical; legacy `reduce_runtime_state` delegates. |
| `frame` | `shs/domains/frame/` | Empty vocabs (monostate); identity reducer, pinned. |
| `geometry` | `shs/domains/geometry/` | TBN operator ingested (rung 08); culling runtimes migrate later. |
| `lighting` | `shs/domains/lighting/` | Lambert terms ingested (rung 08); culling runtimes migrate later. |
| `camera` | `shs/domains/camera/` | Pure builders; bridge in `execution/platform/`. |
| `resources` | `shs/domains/resources/` (+ `edge/` stores) | Asset data spine; registries are edge candidates. |
| `sky` | `shs/domains/sky/` | Value models; `ISkyModel` virtual flagged for replacement. |
| `logic` | `shs/domains/logic/` | Table-driven value FSM beside the legacy callback class. |
| `scene` | `shs/domains/scene/` (+ `edge/` stores) | Item values + projection spine; stores are edge candidates. |
| `gfx` | `shs/domains/gfx/` (+ `edge/` registry) | Handles + pixel buffers; `RTRegistry` is an edge candidate. |
## 8. Bounded contexts & orchestrators (amendment, 2026-09-16)

A bounded context (Constitution II Rule 11) is a suite of cohesive pipelines
over shared PODs with one error/event language. Current contexts:

| Context | PODs | Error / event language |
| :--- | :--- | :--- |
| Rendering | `renderpath`, `frame`, `geometry`, `lighting`, `camera`, `resources`, `sky`, `scene`, `gfx` | `PathSwapRejectionReason` + per-pod `*Event` (catalog: `EVENT_FLOW.md`) |
| Session flow | `input`, `logic` | `InputEvent`, `FsmEvent` (+ rejections as facts) |

Stages compose synchronously as Kleisli chains (KDBA gateway, Rule 11) — including across contexts inside a saga orchestrator pod; direct POD writes across boundaries stay forbidden. A multi-context
workflow adds a saga orchestrator that is itself a pod (Core 4+1, §6.1);
its `.or_else()` compensator consumes the receipt/fact per Rule 12 (transient SagaContext evaporates on failure).
