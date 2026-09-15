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

## Phase P1 — `renderpath` Pod in the Engine Lib

Goal: the first formal Domain Pod in `shs-software-renderer-lib`, wrapping the
existing recipe → compiler → plans spine.

- [ ] Create `include/shs/domains/renderpath/` with `contract` (re-export of recipe /
      plan / capabilities / runtime-state types), `action` (closed
      `RenderPathCommand` variant), `event` (closed `RenderPathEvent` variant),
      `reducer` (`reduce_render_path` wrapping `RenderPathCompiler` value-fully).
- [ ] Reducer invariant: invalid compile ⇒ keep previous plan + `PATH_SWAP_REJECTED`.
- [ ] `ctest` gate: `shs_renderer_vop_renderpath_*` — reducer tests compile and pass
      with zero Vulkan/SDL links (pure value tests, frame-arena events).

**DoD**: path selection, technique switching, culling-mode changes, and rejection
behavior all provable via pure unit tests; `pipeline/` headers unchanged for existing
consumers (pod re-exports, no breakage).


## Phase P1.5 — Contiguous Container Infrastructure (Prerequisite for Cache Streaming)

Goal: extract the memory utilities that exist only inside demos into shared lib
containers so every pod gets §7.2-compliant backing stores. **Hard prerequisite for
§7.1 cache-streaming and for P2/P3 hot loops.**

- [ ] Promote `FrameMemoryResource` (bump arena, currently demo-local in
      tetris/snake/fps) to `include/shs/memory/frame_memory_resource.hpp`.
- [ ] Add `include/shs/containers/soa_table.hpp`: pmr-backed multi-column table —
      one contiguous `pmr` allocation per column, upfront `reserve()`, geometric
      growth with compaction event, generational `uint32_t` handles, swap-and-pop
      removal, 64-byte column alignment.
- [ ] Add `include/shs/containers/flat_map.hpp`: open-addressing pmr map (no nodes)
      for keyed hot lookups.
- [ ] Migrate demo pod contracts (`spatial_fx`, `snake/matrix`, `fps/matrix`) onto
      `SoaTable`; delete demo-private arena copies.
- [ ] `ctest` gate: `shs_renderer_vop_containers_*` — growth/compaction/handle
      stability/headless benchmarks proving linear walks stay cache-resident.

**DoD**: `grep` gate shows zero `std::list/map/set` in lib hot-state headers; pod
state columns are exclusively `SoaTable`/pmr-vector/arena-span; §7.1 prefetch and
streaming-store kernels land on real contiguous columns.

## Phase P2 — Vulkan Driver, Pod-First

Goal: create the missing `rhi/drivers/vulkan/` against the existing value-desc
vocabulary (`resource_desc`, `command_desc`, `pipeline_desc`, `sync_desc`).

- [ ] `vk_backend.hpp` implementing `IRenderBackend` (fills the aspirational
      `backend_factory.hpp` include).
- [ ] `vk_device.hpp` / `vk_resources.hpp` / `vk_pipelines.hpp` / `vk_commands.hpp` /
      `vk_sync.hpp` per the architecture doc §4.
- [ ] Descriptor-hash-keyed explicit caches (no lazy hidden caches, no per-node alloc).
- [ ] `CommandDesc` stream recording: passes emit value command spans on the frame
      arena; driver translates.

**DoD**: `hello_vulkan_triangle` parity rebuilt on the driver; no `Vk*` handle escapes
the driver boundary (checked by grep gate in CI); GPU object creation happens only on
`PATH_COMPILED` / resource-plan events.

## Phase P3 — Monolith Decomposition

Goal: `demo_forward_classic_renderpath.cpp` (9,373 lines) → thin pod composition.

- [ ] Extract input/camera edges to `shs/input` tokenizer + action tokens.
- [ ] Route all path configuration through `renderpath` pod commands (menu/UI edge
      emits `SelectPathPresetIntent`, `SetRenderingTechniqueIntent`, …).
- [ ] Extract per-frame planner into a `plan`-style pure function emitting
      `CommandDesc` spans.
- [ ] Main loop becomes: input edge → reducers → plan → executor edge → present
      (tetris shape).
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
- [ ] **Linter ↔ docs sync** — the structure linter (landed in P0.5) now also
      checks §6.4 classification table ↔ physical zone agreement, the Domain
      Glossary rows point at final homes, and — per Constitution §2.2(3) — law
      citations across docs are valid (no dangling "Rule N"/"Constitution N"
      references; restatements reference the Constitution instead of re-numbering).

**DoD**: zero old-path includes; every `domains/<pod>/` passes the Core 4
completeness check; §6.4 table and glossary match the tree exactly (CI-verified).

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
- Biggest risk: scope creep in P3 (monolith extraction). Mitigation: extraction order
  in architecture doc §5 is dependency-safe; each step keeps the demo runnable.
- P0.5 is moves-only (no content edits) and one module per commit with facade shims,
  so review and rollback stay trivial; P5 does the content-level suffixing later.
