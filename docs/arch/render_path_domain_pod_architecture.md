# Render Path & Vulkan Renderer as Domain Pods

> Status: **Architecture plan (approved direction, 2026-09-15)**. Implements the Core 4
> Domain Pod canon (Constitution §2.1, canon tables §6.1–6.2) across the dynamic render path system and
> the Vulkan backend. Companion roadmap: `docs/roadmap/domain_pod_engine_rollout_roadmap.md`.
> Multi-context workflows add the saga orchestrator as a pod (Core 4+1, Constitution II §6.1, Rules 11–12).

## 1. Why

The render path system is already a pure value pipeline — `RenderPathRecipe` →
`RenderPathCompiler` → execution/resource/barrier plans → `RenderPathExecutor` — but it
has no explicit command vocabulary, no event log, and its Vulkan reality is a 9,373-line
monolith (`exp-rendering-techniques/demo_forward_classic_renderpath.cpp`) with hard-coded
path logic. Adopting the Core 4 Domain Pod canon gives the renderer the same properties
the game demos already have: auditable state transitions, replayable configuration
history, GPU-free testing of path logic, and hot-swappable paths as pure reducer
transitions.

## 2. Ground Truth (as of 2026-09-15)

| Fact | Consequence |
| :--- | :--- |
| `rhi/drivers/vulkan/` does not exist; `rhi/backend/backend_factory.hpp` references `vk_backend.hpp` aspirationally | The Vulkan driver must be *created*, not refactored — build it pod-first |
| Value-desc RHI vocabulary exists: `rhi/resource/resource_desc.hpp`, `rhi/command/command_desc.hpp`, `rhi/pipeline/pipeline_desc.hpp`, `rhi/sync/sync_desc.hpp` | The "everything is a value description" layer is ready; only the driver and orchestration are missing |
| Vulkan work lives in `exps-gpu-renderer/exp-rendering-techniques/demo_forward_classic_renderpath.cpp` (9,373 lines) and `exp-plumbing/hello_*_vulkan.cpp` probes | Monolith must be decomposed into pods, not grown further |
| `shs/pipeline/` holds 25 headers (recipe, compiler, executor, barrier/resource plans, presets, registries) | Already pod-shaped; needs reorganization under Core 4 suffixes, not rewriting |
| Demos (tetris/snake) have the working reducer/event pattern; tetris pods are the reference | Reuse the demo pattern vocabulary in the lib, not the reverse |

## 3. The `renderpath` Domain Pod (Core 4 mapping)

```text
shs-renderer-lib/include/shs/domains/renderpath/
├── renderpath.contract.hpp   # CORE 1. TYPES
│       RenderPathRecipe, RenderPathExecutionPlan, RenderPathCompatibilityRules,
│       RenderPathCapabilitySet, RenderPathRuntimeState, RenderPathPassEntry,
│       all recipe enums (technique, culling mode, light-volume provider)
│       → moved (re-exported) from pipeline/render_path_recipe.hpp,
│         render_path_capabilities.hpp, render_path_runtime_state.hpp
├── renderpath.action.hpp     # CORE 2. COMMAND  (NEW)
│       closed variant RenderPathCommand:
│         SelectPathPresetIntent{PresetId}
│         SetRenderingTechniqueIntent{RenderPathRenderingTechnique}
│         SetLightVolumeProviderIntent{RenderPathLightVolumeProvider}
│         SetViewCullingIntent{RenderPathCullingMode}
│         SetShadowCullingIntent{RenderPathCullingMode}
│         SetRuntimeFlagIntent{flag_id, bool}   // debug_aabb, enable_shadows, lit_mode…
│         SetResourceKnobIntent{light_tile_size|cluster_z_slices, uint32}
│         ReplaceRecipeIntent{RenderPathRecipe}
│         RequestRecompileIntent{}
├── renderpath.event.hpp      # CORE 3. EVENT  (NEW)
│       closed variant RenderPathEvent:
│         PATH_COMPILED{plan_hash, pass_count}
│         PATH_SWAP_REJECTED{errors}            // reducer kept previous plan
│         PATH_WARNING_RAISED{warnings}
│         RUNTIME_STATE_CHANGED{flag_id, value}
├── renderpath.reducer.hpp    # CORE 4. REDUCER  (NEW)
│       pure: reduce_render_path(RenderPathDomainState, span<const RenderPathCommand>,
│                               const RenderPathCapabilitySet&, arena)
│                            → RenderPathStep{next, events}
│       Transition rule: apply intents → recompile via (value-ified) compiler rules
│       → on valid plan: swap + emit PATH_COMPILED; on invalid: KEEP previous plan
│         + emit PATH_SWAP_REJECTED. Hot-swap safety becomes a reducer invariant.
├── renderpath.plan.hpp       # EXT. Batch compilers
│       existing render_path_resource_plan / barrier_plan / runtime_layout /
│       pass_dispatch builders + build_execution_request(...)
└── (edge — NOT a pod file)
        RenderPathExecutor, pass adapters, RHI drivers stay in pipeline/ + rhi/.
```

**Reducer contract details.** `RenderPathCompiler` is currently a mutable class holding
`RenderPathCompatibilityRules`. The reducer wraps it as pure value-in/value-out: rules
become a parameter (threaded from the pod state), and `compile()` is called without
side effects. The compiler itself remains in `pipeline/` as a shared utility — the pod
owns the *decision*, not the validation helper.

## 4. Vulkan Driver: pod-first construction

Build `rhi/drivers/vulkan/` against the existing desc vocabulary so the driver has no
logic of its own beyond translation:

```text
rhi/drivers/vulkan/
├── vk_backend.hpp      # VkBackend : IRenderBackend — implements backend.hpp interface
├── vk_device.hpp       # value-desc → VkDevice/queues/instance (create once, edge)
├── vk_resources.hpp    # ResourceDesc → VkImage/VkBuffer/allocator tracking (edge)
├── vk_pipelines.hpp    # PipelineDesc → VkPipeline/PipelineLayout cache (edge)
├── vk_commands.hpp     # CommandDesc stream → VkCommandBuffer recording (edge)
└── vk_sync.hpp         # uses rhi/sync/vk_runtime.hpp + sync_desc.hpp
```

Rules:
1. **No `Vk*` handle crosses upward** out of the driver. Everything above sees
   stable IDs and value descs (already the RHI design intent).
2. **Per-frame command recording consumes a value stream**: passes emit
   `CommandDesc` spans built on the frame arena; the driver translates. This keeps
   the entire submission decision layer pure and replayable.
3. **GPU object creation is event-driven**: `PATH_COMPILED` (and resource-plan output)

## 5. Monolith Decomposition (`demo_forward_classic_renderpath.cpp`)

The 9,373-line demo becomes a thin composition of pods. Extraction order (dependency-safe):

1. **Camera / input edges** → existing `shs/input` command tokenizer (action tokens).
2. **Scene state** → `scene` module contracts (`SceneObjectSet`, `LightSet`).
3. **Path configuration** → replace hard-coded path setup with `renderpath` pod commands
   fed from a menu/UI edge (`SelectPathPresetIntent`, `SetRenderingTechniqueIntent`…).
   This is what makes paths *hot-swappable in the demo*, proving the L4 claim.
4. **Per-frame planner calls** → `spatial_fx`-style plan function emitting
   `CommandDesc` spans for the driver.
5. **Main loop** reduced to: input edge → reducers → plan → executor edge → present,
   matching the tetris `hello_3d_tetris.cpp` shape.

`hello_modern_vulkan.cpp` and the other probes are migrated incrementally or retired
once the decomposed demo covers them.

## 6. Non-Goals

- No changes to the software rasterizer's hot paths (Constitution III's SoA/ECS
  tenets and this Constitution's §7.1/§7.2 memory laws are already satisfied there).
- No GPU payloads through pods: cull/light GPU structs remain flat SHS-space value data.
- No dynamic allocation per frame in reducers: command/event vectors use the frame arena.
- No loss of the Jolt culling bridge: `RenderPathLightVolumeProvider::JoltShapeVolumes`
  remains a recipe value; shapes stay behind the geometry adapter (see Jolt assessment).

## 7. Key Files (target state)

| Concern | File |
| :--- | :--- |
| Pod canon | `docs/spec/value_oriented_programming.md` §6 |
| This architecture | `docs/arch/render_path_domain_pod_architecture.md` |
| Phased rollout | `docs/roadmap/domain_pod_engine_rollout_roadmap.md` |
| Path pipeline (current) | `docs/arch/render_path_architecture.md` |
| Pod reference implementation | `tetris/domains/matrix/*` (fully canonical) |

   are the only triggers for pipeline/render-pass/image creation. No lazy hidden
   caches — caches are explicit tables keyed by desc hashes (DOD: open addressing,
   generation counters, no per-node allocation).
4. **Backend-agnostic pods**: the `renderpath` pod must compile and test with zero
   Vulkan includes. Capability gating happens in the reducer via
   `RenderPathCapabilitySet`; the driver only ever receives already-validated plans.

