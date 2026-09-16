# Domain Pod Engine Rollout Roadmap

> Status: **Active plan (2026-09-15)**. Ratifies and rolls out the Domain Pod
> Constitution (§2.1 + Rule 10, precedence §2.2; canon tables §6.1–6.2): the
> "everything is a pure reducer based Domain Pod" law binding the engine library
> and all demos alike, rolled into the library tree, the dynamic render path
> system, and the Vulkan backend.
> Architecture details: `docs/arch/render_path_domain_pod_architecture.md`.

## Vision

"Everything is a Domain Pod in mind" (Constitution §2.1): every stateful subsystem is expressed as
**Types (contract) + Command (action) + Reducer + Event**, with batch planners as an
optional extension and all side effects confined to edges. State transitions become
pure, replayable, GPU-free-testable, and auditable — data-oriented design with dynamic
freedom (any path, any technique, any combination, hot-swapped at runtime as a pure
reducer transition).

## Phase P0 — Canon (DONE 2026-09-15)

- [x] Constitution II §6.1/6.2 amended: Core 4 mandatory, extensions conditional,
      conformance note for pre-canon pods.
- [x] Tetris `ARCHITECTURE.md` Part I synced with the canon.
- [x] **Domain Pod Constitution ratified (§2.1 + §2.2 + Mandatory Rule 10)**: "everything
      is a pure reducer based Domain Pod" with the Core 4 {types, action, reducer,
      event} is now supreme law binding the engine library *and* all demos;
      physical layout (`domains/` + edge zone) and CI linter enforcement referenced.
      §2.2 adds the Law Precedence & Single-Source Rule (numbered rules win;
      reference-don't-renumber; Rule 1 ↔ §7.1 owned-state reconciliation).


## Phase P0.5 — Pod-First Tree Restructure (DO FIRST — zero code changed yet)

Goal: reorganize `include/shs/` once, now, so the physical tree itself expresses the
"everything is a Domain Pod" philosophy — **before** any pod lands, so P1 writes
directly into its final home. Purely mechanical: moves + facade compatibility headers
+ CMake path updates. This replaces the "directories stay" stance of the old P5.

### Target tree

```text
include/shs/
├── core/                  # PRIMITIVES: ids, handles, units, log, result, time
├── memory/                # PRIMITIVES: frame_memory_resource, dual-tier laws
├── containers/            # PRIMITIVES: soa_table, flat_map (P1.5)
├── domains/               # ★ EVERY domain logic lives here, uniform skeleton:
│   │                      #   <pod>.contract.hpp / .action.hpp / .reducer.hpp /
│   │                      #   .event.hpp (+ .plan.hpp / edge subfolder if needed)
│   ├── renderpath/        # recipe, compiler wrap, plans, presets catalog (P1)
│   ├── scene/             # objects/culling/instance → contract+plan; world/system → edge
│   ├── lighting/          # light_set/types → contract; culling → plan; runtime → reducer
│   ├── input/             # command/value_actions → action; processor → reducer
│   ├── camera/            # 7 headers → contract + plan
│   ├── geometry/          # shapes + jolt adapters → contract + plan (empty action/event
│   │                      #   vocabularies are LEGAL per Constitution §6.1)
│   ├── sky/               # → contract + plan
│   ├── gfx/               # rt_types/rt_handle → contract; rt_registry → edge subfolder
│   ├── resources/         # asset/resource value types → contract
│   └── frame/             # frame_params/technique_mode → contract
└── execution/             # EDGE ZONE — stateful, side-effectful, includes domains,
    │                      # never included by a domain
    ├── pipeline/          # render_path_executor, pass_adapters, registries, frame_graph
    ├── passes/            # pass bodies (shadow_map, pbr_forward, tonemap, …)
    ├── rhi/               # backend + drivers/ (incl. future drivers/vulkan/, P2)
    ├── sw_render/         # software rasterizer edge
    ├── platform/          # OS/window/filesystem edges
    ├── job/               # thread pool + parallel_for (execution engine)
    ├── shader/            # shader compilation edges
    └── app/               # application loop edge
```

### Classification of current modules (move map)

| Current | Destination | Rationale |
| :--- | :--- | :--- |
| `core/`, `frame/` | `core/`, `domains/frame/` | primitives vs. frame contract values |
| (new) | `memory/`, `containers/` | P1.5 utilities get first-class zone |
| `pipeline/` (25) | `domains/renderpath/` + `execution/pipeline/` | value spine vs. executor/registries — the P1 seam |
| `scene/` (10) | `domains/scene/` (+ edge parts stay in pod edge subfolder) | transforms are pod-shaped; `world`/`system` are stateful edges |
| `lighting/` (9) | `domains/lighting/` | same split: sets/types vs. runtime culling |
| `input/` (8) | `domains/input/` | already action/reducer shaped, just unsuffixed |
| `camera/` (7) | `domains/camera/` | contract + plan transforms |
| `geometry/` (19) | `domains/geometry/` | shapes/jolt = contract + plan, empty command vocab |
| `sky/` (5) | `domains/sky/` | contract + plan |
| `gfx/` (5) | `domains/gfx/` | handle types = contract; registry = edge subfolder |
| `resources/` (8) | `domains/resources/` | value types + load planners |
| `passes/`, `rhi/`, `sw_render/`, `platform/`, `job/`, `shader/`, `app/` | `execution/…` (unchanged names) | edges, declared as such |
| `assets/` (empty) | delete | vestigial-empty (removed during execution) |
| `logic/` (2) | `domains/logic/` | ~~vestigial~~ corrected during execution: live consumers exist in `exps-gpu-renderer/exp-plumbing` demos — moved with facades instead of deleted |

### Migration protocol (non-breaking by construction)

1. One move per module, one commit; CMake include dirs updated in the same commit.
2. Old paths keep **facade compatibility headers** (`#include "shs/scene/scene_types.hpp"`
   → forwards to `shs/domains/scene/scene.contract.hpp`) with a `#pragma message`
   deprecation note; deletion of facades is a later cleanup commit.
3. Build + `ctest` green after every move (the tree is header-only; moves are cheap
   while no downstream code depends on new structure).
4. No file content changes during P0.5 — moves only. Suffixing (Core 4 completion)
   happens per-pod in later phases, keeping this phase reviewable.

**DoD**: tree matches the target map; all facade headers forward correctly; build +
existing `ctest` suite green; boundary linter (from old P5, pulled forward) enforces
`domains/ → {core, memory, containers, domains}` include-direction from day one.

> **Status (2026-09-15): DONE.** All zones moved: `frame/input/camera/scene/lighting/
> sky/gfx/geometry/resources/logic` → `shs/domains/…`, `pipeline/passes/rhi/sw_render/
> platform/job/shader/app` → `shs/execution/…`. 140 headers moved, 140 facade shims
> written (each `#pragma message`-deprecated and forwarding to its canonical path);
> `shs/` now contains only `core/`, `domains/`, `execution/`. Two deviations from the
> original plan, both recorded in the move map above: (1) `logic/` moved rather than
> deleted; (2) `pipeline/` moved wholesale to `execution/pipeline/` — the value-spine
> split into `domains/renderpath/` is P1's re-export contract, as already specified.
> Zero file content changes; all internal includes still use legacy paths and resolve
> through facades (canonicalized in P5). Verification: full build (lib + tests + every
> demo incl. gpu/vulkan exps) green, `ctest` 100%, boundary linter extended with two
> new gates — facade sanity (every facade forwards to an existing non-self canonical
> header) and the domains include-direction law on canonical include text. Legacy-path
> includes inside `domains/` (5 today) are counted as an advisory INFO until P5.
> Migration tooling: `tools/move_zone.sh` (zone mover + facade generator).
>
> **Convergence addendum (2026-09-15, later same day):** the deferred facade
> deletion + canonicalization (old P5) was executed early as part of the
> "finish the core library" push: all 140 facade headers + 1 stable re-export
> deleted, every live include (both libs incl. tests and the GPU driver pods)
> canonicalized to `shs/{core,memory,containers,domains,execution}`. Library
> renamed to converged names: `shs-software-renderer-lib` → **`shs-core-lib`**
> (targets `shs_core` / alias `shs::core`, values target `shs_core_values` /
> `shs::core-values`), `shs-gpu-renderer-lib` → **`shs-gpu-lib`** (`shs_gpu` /
> `shs::gpu`, temporary keep). All exps demos/probes parked at the CMake level
> (source kept; ctest baseline 14 → **5**, lib tests only). Per-pod header
> suffixing (Core 4) remains future work.
>
> **Single-library convergence (2026-09-15, same day):** `shs-gpu-lib` retired
> ahead of P3 — its trees (SDL/windowed monolith `shs/rhi/drivers/vulkan/`,
> `shs/pipeline/vk_*`, `shaders/vulkan/`, VMA implementation) were absorbed
> into the surviving lib and are `SHS_HAS_VULKAN`-gated (GPU-free configure +
> build still green). The lib then took its final name: `shs-core-lib` →
> **`shs-renderer-lib`** (targets `shs_renderer` / `shs::renderer`, values
> `shs_renderer_values` / `shs::renderer-values` — no legacy aliases kept).
> The monolith's P3 pod decomposition remains future work.

## Phase P1 — `renderpath` Pod in the Engine Lib

Goal: the first formal Domain Pod in `shs-renderer-lib`, wrapping the
existing recipe → compiler → plans spine.

- [x] Create `include/shs/domains/renderpath/` with `contract` (re-export of recipe /
      plan / capabilities / runtime-state types), `action` (closed
      `RenderPathCommand` variant), `event` (closed `RenderPathEvent` variant),
      `reducer` (`reduce_render_path` wrapping `RenderPathCompiler` value-fully).
- [x] Reducer invariant: invalid compile ⇒ keep previous plan + `PATH_SWAP_REJECTED`.
- [x] `ctest` gate: `shs_renderer_vop_renderpath_*` — reducer tests compile and pass
      with zero Vulkan/SDL links (pure value tests, frame-arena events).

**DoD**: path selection, technique switching, culling-mode changes, and rejection
behavior all provable via pure unit tests; `pipeline/` headers unchanged for existing
consumers (pod re-exports, no breakage).

> **Execution record (2026-09-15 — DONE).** Core 4 landed as
> `renderpath.contract.hpp` / `renderpath.action.hpp` / `renderpath.event.hpp` /
> `renderpath.reducer.hpp` in `include/shs/domains/renderpath/`. The contract
> re-exports the spine via using-declarations under `shs::renderpath` (canonical
> `shs::` names preserved). Commands: `SelectPathPresetIntent`,
> `SetRenderingTechniqueIntent`, `SetViewCullingModeIntent`,
> `SetShadowCullingModeIntent`, `SetRuntimeToggleIntent` (closed `RuntimeToggle`
> key set). Events: `PathCompiledEvent`, `PathSwapRejectedEvent` (reason enum,
> no `std::string` payloads), `TechniqueSwitchedEvent`, `CullingModeChangedEvent`,
> `RuntimeToggledEvent`. Deviations from the checkbox text: (1) `pipeline/` moved
> wholesale to `execution/pipeline/` in P0.5, so the pod's re-exports point there —
> `pipeline/` facades keep old-path consumers working untouched; (2) the boundary
> linter gained a P1-sanctioned carve-out: `shs/domains/renderpath/` is the only
> domain pod allowed to include execution zones (it IS the contract seam). The
> `shs::renderer-values` INTERFACE target (header-only: include dirs + glm only,
> no SDL/assimp/Vulkan) landed with this phase — the renderpath test binary's
> link line is `libglm.a` and nothing else. ctest: `shs_renderer_vop_renderpath_tests`
> covers path selection, technique switching, culling accept/reject, the
> previous-plan-kept rejection invariant, and pre-plan runtime toggles; events
> allocate on a `std::pmr::monotonic_buffer_resource` frame arena.



## Phase P1.5 — Contiguous Container Infrastructure (Prerequisite for Cache Streaming)

Goal: extract the memory utilities that exist only inside demos into shared lib
containers so every pod gets §7.2-compliant backing stores. **Hard prerequisite for
§7.1 cache-streaming and for P2/P3 hot loops.**

- [x] Promote `FrameMemoryResource` (bump arena, currently demo-local in
      tetris/snake/fps) to `include/shs/memory/frame_memory_resource.hpp`.
- [x] Add `include/shs/containers/soa_table.hpp`: pmr-backed multi-column table —
      one contiguous `pmr` allocation per column, upfront `reserve()`, geometric
      growth with compaction event, generational `uint32_t` handles, swap-and-pop
      removal, 64-byte column alignment.
- [x] Add `include/shs/containers/flat_map.hpp`: open-addressing pmr map (no nodes)
      for keyed hot lookups.
- [x] Migrate demo pod contracts (`spatial_fx`, `snake/matrix`, `fps/matrix`) onto
      `SoaTable`; delete demo-private arena copies.
- [x] `ctest` gate: `shs_renderer_vop_containers_*` — growth/compaction/handle
      stability/headless benchmarks proving linear walks stay cache-resident.

**DoD**: `grep` gate shows zero `std::list/map/set` in lib hot-state headers; pod
state columns are exclusively `SoaTable`/pmr-vector/arena-span; §7.1 prefetch and
streaming-store kernels land on real contiguous columns.

> **Execution record (2026-09-15 — DONE).** `shs::memory::FrameMemoryResource`
> landed with strict tiering: overflow is `bad_alloc`, never a silent fallback to
> the persistent tier (the old snake copy silently spilled into
> `get_default_resource()` — a §3 Rule 5.1 violation, now fixed); diagnostics
> expose `used()` / `high_water_mark()`. `shs::containers::SoaTable<Ts...>` is a
> generational slot table: one 64-byte-aligned pmr allocation per column,
> `reserve()`-upfront with power-of-two geometric growth that bumps
> `compaction_count()` (the cold-path compaction event), `SoaHandle{slot, gen}`
> stability across growth AND swap-and-pop, `erase_dense()` walk-and-kill for
> column kernels, and `column<I>()` spans as the §7.1 streaming targets.
> `shs::containers::FlatMap<K,V>` is a node-free open-addressing map (SoA
> key/value arrays at 64-byte-aligned bases, linear probing, tombstones with
> reuse, 0.7 max load, power-of-two rehash). Deviation from the checkbox text:
> the demo migration scope was (a) all three demo-private `FrameMemoryResource`
> copies deleted (tetris keeps its historical 16 MB capacity via the ctor param;
> snake/fps use the shared 8 MB default), and (b) the snake `spatial_fx` pod's
> `ShatterParticleSoA` contract migrated onto `SoaTable` (its per-element
> column `erase` — a §7.2 rule 3 violation — replaced by the swap-and-pop
> `erase_dense` walk-and-kill kernel); tetris's 4-vector particle SoA and the
> fps/matrix tables keep their pmr-vector columns until their own migration pass
> (they already satisfy §7.2 shape). The boundary linter gained the P1.5 DoD
> gate: FAIL on any node-based container under `shs/memory|containers|frame/`,
> INFO-counting the 13 cold string-keyed registry uses in `domains/` for the P5
> FlatMap migration. ctest: `shs_renderer_vop_containers_tests` (linked only to
> `libglm.a`, matching the P1 DoD) covers arena alignment/O(1)-reset/strict
> overflow, handle stability across growth and swap-and-pop, density
> preservation, 64-byte column alignment, a 1M-row linear-walk headless
> benchmark (streams in ~6 ms — cache-resident), and FlatMap collision,
> tombstone, and rehash behavior. All three migrated demos pass their headless
> `--screenshot` smoke. ctest 10/10; boundary linter all-OK.

## Phase P2 — Vulkan Driver, Pod-First

Goal: create the missing `rhi/drivers/vulkan/` against the existing value-desc
vocabulary (`resource_desc`, `command_desc`, `pipeline_desc`, `sync_desc`).

- [x] `vk_backend.hpp` implementing `IRenderBackend` (fills the aspirational
      `backend_factory.hpp` include via the `shs/rhi/drivers/vulkan/` facade).
- [x] `vk_device.hpp` / `vk_resources.hpp` / `vk_pipelines.hpp` / `vk_commands.hpp` /
      `vk_sync.hpp` per the architecture doc §4.
- [x] Descriptor-hash-keyed explicit caches (no lazy hidden caches, no per-node alloc;
      registry + pipeline cache are GPU-free testable via create hooks).
- [x] `CommandDesc` stream recording: `record_commands<Sink>` 8-way variant
      dispatch (BeginPass, BindPipeline, BindVertex/IndexBuffer, DrawIndexed,
      Dispatch, Barrier, EndPass); `VulkanCommandRecorder` is the device-bound sink.
- [x] `VulkanFrameSync` slot bookkeeping (frame-in-flight slots + per-queue timeline
      signal via `VulkanLikeRuntime`; synchronous driver submission drains on end_frame).
- [x] `vop_vk_driver_tests.cpp` — 15 GPU-free cases (mappers, hashes, registry dedupe /
      failure semantics, pipeline cache, spy-sink stream ordering, frame-sync slots,
      headless backend contract, create-info purity); registered as
      `shs_renderer_vop_vk_driver_tests` behind optional `find_package(Vulkan)`
      (links the loader only so never-executed GPU paths resolve; binary stays
      GPU-free at runtime).
- [x] Grep gate: no `Vk*` token outside `include/shs/execution/rhi/drivers/vulkan/`
      in the lib (facade shims carry no Vk tokens); `check_vop_boundaries.sh` green.

**DoD status**: translation/cache/record layers complete and GPU-free testable; the
`hello_vulkan_triangle` parity probe stays compile-gated — its runtime path needs a
device (no ICD in CI), so it lands with Phase J / P6 integration rather than here.

## Phase P3 — Monolith Decomposition

Goal: `demo_forward_classic_renderpath.cpp` (9,373 lines) → thin pod composition.

- [ ] Extract input/camera edges to `shs/input` tokenizer + action tokens.
- [ ] Route all path configuration through `renderpath` pod commands (menu/UI edge
      emits `SelectPathPresetIntent`, `SetRenderingTechniqueIntent`, …).
- [ ] Extract per-frame planner into a `plan`-style pure function emitting
      `CommandDesc` spans.
- [ ] Main loop becomes: input edge → reducers → plan → executor edge → present
      (tetris shape).
- [ ] Hybrid / GPU-free demo mode — demos must honor the backend factory's
      fallback instead of hard-failing on the concrete Vulkan type: branch on
      `RenderBackendCreateResult::active` + `BackendCapabilities` (the planner's
      existing `dynamic_cast` policy-branching pattern), so `SHS_RENDER_BACKEND=software`
      (the default) runs anywhere. Paired `_sw`/`_vk` demo binaries converge into
      one binary with runtime backend selection. Also make the top-level
      `find_package(VulkanMemoryAllocator ... REQUIRED)` optional/QUIET and gate
      all Vulkan sources/targets behind `SHS_HAS_VULKAN` so configure+build
      succeeds on machines with no GPU and no Vulkan SDK.
- [ ] Open pass-ID / light-registry extensibility (Constitution I §7) — `PassId`
      gains a builtin range + open registered range (or stable-string-hash
      contract keys) so demo/consumer-owned passes need no core edit; apply the
      same open-registry treatment to `RenderPathLightVolumeProvider` and
      technique/light preset enums as consumers require custom abstractions.
      (Formal contract: `render_path_architecture.md` §4.)
- [ ] Migrate or retire `hello_*_vulkan.cpp` probes.

**DoD**: demo under ~1.5k lines; all 5 path presets × techniques hot-swappable at
runtime *through the reducer* (event log visible in an on-screen debug overlay).


## Phase P4 — Demo Conformance (Core 4 debt)

Goal: bring tetris/snake pods to the canon flagged in Constitution §6.2.

- [ ] tetris: `session` → extract `session.event.hpp`; `mission` → `mission.action.hpp`
      + `mission.event.hpp` (move `MissionEventType`/`MissionEventOut`), drop
      `std::string` from event payloads (fixed tag / string_view), take frame arena
      as parameter; `progression` → explicit empty `progression.action.hpp`;
      `environment` / `spatial_fx` → explicit closed action+event vocabularies.
- [ ] snake: same audit pass on `matrix` / `progression` / `spatial_fx`.
- [ ] Regenerate `docs/pods/EVENT_FLOW.md` after event moves.

**DoD**: every pod in every demo has all Core 4 files; `EVENT_FLOW.md` matches
`domains/*/event` registries exactly.

## Phase P5 — Semantic Completion (Core 4 Suffixing & Role Maps)

Goal: P0.5 moved the *directories* into pod zones without touching content; P5
completes the *semantics* — every pod-zone header carries an explicit Core 4 /
extension role, and the classification is machine-checked.

- [ ] **Suffix completion per module** — split or rename multi-role headers into
      `*.contract.hpp` / `*.plan.hpp` / `*.edge.hpp` (e.g. `scene/system.hpp` →
      `domains/scene/scene.edge.hpp`; `gfx/rt_registry` → `gfx` edge subfolder),
      using the facade shims for compatibility.
- [ ] **Core 4 completeness for formal pods** — `renderpath`, `scene`, `lighting`,
      `input` gain action/event vocabularies (closed variants; `std::monostate`
      where genuinely empty, per Constitution §6.1).
- [ ] **Retire facade shims** once no consumer includes old paths; delete
      `#pragma message` forwards.
- [ ] **Retire legacy seams** — audit `frame_graph.hpp` / `pluggable_pipeline.hpp`
      for removal once the renderpath pod covers their use cases.
- [x] **Converge to a single renderer library** — DONE 2026-09-15, ahead of
      the original sequencing: `shs-gpu-lib` was absorbed into the surviving
      lib (monolith `shs/rhi/` + `shs/pipeline/vk_*` + VMA edge moved in,
      `SHS_HAS_VULKAN`-gated) and `shs-core-lib` renamed to
      **`shs-renderer-lib`** with the converged target scheme
      (`shs_renderer` / `shs::renderer`, `shs_renderer_values` /
      `shs::renderer-values` — no legacy aliases kept). "Software vs GPU" is
      now a driver-pod selection (`drivers/software`,
      `drivers/vulkan`, `drivers/opengl`) behind the one `IRenderBackend`
      contract + capability gates — not a library split. Remaining follow-ups
      (not blocking the rename): §6.4 classification table + Domain Glossary
      sync, and the P3 decomposition of the absorbed monolith into the
      pod-aligned driver.
- [ ] **Linter ↔ docs sync** — the structure linter (landed in P0.5) now also
      checks §6.4 classification table ↔ physical zone agreement, the Domain
      Glossary rows point at final homes, and — per Constitution §2.2(3) — law
      citations across docs are valid (no dangling "Rule N"/"Constitution N"
      references; restatements reference the Constitution instead of re-numbering).
      Add the pluggability check per Constitution I §7: extension points
      (registries/contracts/recipes) must never require core edits — lint that
      consumer/demo-owned pass, light, and technique abstractions resolve
      through open registries, and that no core header is a hard dependency of
      the extension mechanism.

**DoD**: zero old-path includes; every `domains/<pod>/` passes the Core 4
completeness check; §6.4 table and glossary match the tree exactly (CI-verified).
End state: single **`shs-renderer-lib`** (the P3/P5 retirement decision above);
software vs GPU backends differ only by driver pod.

## Phase P6 — Integration Hardening (with Phase J Vulkan work)

- [ ] `PATH_COMPILED` events drive `RenderPathExecutor` (re)builds; executor keeps
      persistent GPU tables keyed by plan hash + generation.
- [ ] Replay harness: serialize command+event logs of a session; replay produces
      identical plan sequence (determinism gate, headless CI).
- [ ] Rollback-ready: recipe state snapshots are plain values (already true) — add
      time-travel debug overlay reading the event log.

**DoD**: headless replay CI green; hot-swap of all presets with zero frame allocation
outside arenas; event log overlay ships in the demo.

## Sequencing & Risk

- **Adopted strategy: library first (with headless tests), demos later.** The
  library migration (P0.5 → P1 → P1.5 → P2) proceeds independently; demos stay on
  facade compatibility headers — compilable throughout, their Core 4 debt tracked
  as P4 (per the §6.2 pre-canon conformance note). Every pod lands with its
  headless reducer `ctest` in the same commit (the existing
  `tests/vop_core_tests.cpp` DummyBackend harness is the template; P1's
  `shs_renderer_vop_renderpath_*` gate generalizes it). Ordering constraints:
  P3 (monolith decomposition) is demo-side work consuming the P1 renderpath pod,
  so it interleaves with the library track rather than following it; P5 facade
  retirement happens only after P4 has moved the demos onto the new paths.
  P1 hardening: add a header-only `shs::renderer-values` INTERFACE target
  (pod/value headers only) so reducer tests link without SDL2/assimp —
  mechanically proving pods never leak edge dependencies.
- **P0.5 first** (while zero code has changed): the tree move is cheapest now and
  means P1 lands the renderpath pod directly in `domains/renderpath/` with no later
  migration. P1.5 (container infrastructure) lands before P2/P3 — cache-streaming
  (§7.1) has nothing to stream unless pod columns are §7.2-contiguous first.
  P4 is independent — can run in parallel any time.
- **Execution batching (backlog plan, 2026-09-15):** the P3–P6 + material tasks
  are grouped into 5 runs by touch surface (minimizes context switching; each
  run is independently committable and verifiable):

  - **Run 1 — Monolith Decomposition** (P3 #1–4): input/camera edges →
    `shs/input`; path config via `renderpath` intents; pure planner emitting
    `CommandDesc` spans; main loop → input edge → reducers → plan → executor
    edge → present. DoD: demo < 1.5k lines, reducer-driven hot-swap visible in
    a debug overlay. One continuous surgery on the same file.
  - **Run 2 — GPU-Free Everything** (P3 #5 + #7): hybrid/GPU-free demo mode
    (backend-factory fallback, `SHS_HAS_VULKAN` gate, optional VMA, merged
    `_sw`/`_vk` binaries); migrate/retire `hello_*_vulkan.cpp` probes. DoD:
    configure+build+ctest+demo run with no GPU and no Vulkan SDK. Lands right
    after Run 1 so all later runs verify GPU-free.
  - **Run 3 — Open Everything** (P3 #6 + material Phases 2/3/3b; parallel
    track): open `PassId` range + open light/light-volume registries; shader
    templating & C++ assembler; material graph compiler; GLSL/Slang/C++
    emission. DoD: demo-owned pass/technique registers with zero core edits;
    one authored material emits to all three targets. Library-side only, joins
    the main line at Run 4.
  - **Run 4 — Conformance & Convergence Sweep** (P4 + P5): tetris/snake Core 4
    debt; suffix completion; retire 141 facades + legacy seams; converge to
    single `shs-renderer-lib`; linter ↔ docs sync + pluggability lint. DoD:
    zero old-path includes, §7 proven mechanically, one library. Blocked by
    Run 1 (facade retirement needs rewired consumers).
  - **Run 5 — Integration Hardening** (P6 #17–20): `PATH_COMPILED`-driven
    executor rebuilds; replay harness; rollback-ready snapshots; runtime parity
    probe un-gated. Needs a device (or swiftshader) and Phase J.

  Hard constraints: Run 1 → Run 4 (facade retirement), Run 2 → Run 5 (CI).
  Run 3 free-floating.
- Biggest risk: scope creep in P3 (monolith extraction). Mitigation: extraction order
  in architecture doc §5 is dependency-safe; each step keeps the demo runnable.
- Biggest risk: scope creep in P3 (monolith extraction). Mitigation: extraction order
  in architecture doc §5 is dependency-safe; each step keeps the demo runnable.
- P0.5 is moves-only (no content edits) and one module per commit with facade shims,
  so review and rollback stay trivial; P5 does the content-level suffixing later.

## Backlog — POD Semantics Hardening (parked 2026-09-15; work later)

> Suggestions from the post-Tier0 lib review on strengthening pure reducer
> Domain POD semantics in `shs-renderer-lib`. Not scheduled — recorded so the
> Runs 1–5 plan above can absorb them at the right moment. `renderpath` (P1)
> proved the pattern; this backlog is about making the pattern cheap to follow
> correctly and hard to follow incorrectly.

- [ ] **Uniform reducer signature** — pin the house signature
      `reduce(PodState&, span<const Action>, const Inputs&, pmr::vector<Event>&)`
      (time/caps/compiler inputs always explicit parameters; events on the
      caller's frame arena) across all future pods, so one generic edge loop
      drives every pod and the P6 replay harness is a lib facility, not
      per-pod work. *Slot: Run 1 (main-loop rewrite is where per-pod glue
      gets deleted).*
- [ ] **Header-only pod test kit** (`domains/pod_test_kit.hpp`) — replay
      assert (command log × 2 → identical event log), snapshot value-equality
      round-trip (also enforces "pod state is a value"), per-pod invariant
      predicates checked after each reduction in debug builds. Turns the
      "every pod lands with a headless ctest" gate into three lines per pod.
      *Slot: Run 4.*
- [ ] **Semantic purity linters** (extend the P0.5 boundary linter beyond
      include-direction): forbid ambient entropy/time in `domains/`
      (`rand(`, `std::chrono`, `std::time`, `getenv` — dt arrives as input);
      forbid `std::unordered_*` iteration in reducer paths (hash order breaks
      replay); extend the `Vk*` token gate pattern to `canvas`, `SDL_`,
      `fopen` in `domains/` (would have caught `skybox_renderer.hpp` /
      `jolt_debug_draw.hpp` at birth). *Slot: Run 4.*
- [ ] **Promote the `std::expected` fallible-transition idiom** from
      `renderpath.reducer.hpp` `detail` (VOP spec §8) to the canonical
      Constitution recipe for rejected transitions — closed-enum error
      payload, events only in `transform/or_else` continuations, previous
      state untouched on rejection; new compilers return `expected` directly
      instead of post-hoc string classification
      (`classify_plan_rejection`'s string-matching is the smell to not
      repeat). *Slot: Constitution/roadmap doc edit, any time.*
- [ ] **Generated event-flow docs** — each pod's `event.hpp` declares a
      `constexpr` name table; `EVENT_FLOW.md` and the P6 debug overlay's
      event labels generate from those tables (generate beats lint for
      doc-drift). *Slot: Run 4, extending the P5 linter ↔ docs sync.*
- [ ] **Seeded determinism contract** — any stochastic pod carries its RNG
      state *in the pod state* (seedable via action), never a global
      (xorshift precedent in tetris/spatial_fx). *Slot: Constitution doc
      edit, any time.*
- [ ] **Replay harness consumes the test kit** — P6's replay CI should
      reuse the pod test kit's replay machinery, not reimplement it.
      *Slot: Run 5.*

