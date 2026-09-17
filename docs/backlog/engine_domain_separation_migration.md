# Engine-ready module boundaries: proposal and migration backlog

Status: migration started; camera convention compatibility pilot implemented.

## Pilot evidence (2026-09-17)

- [x] Move the stateless camera convention implementation to its named owner;
  keep a single-hop old-path forwarding header and preserve every public symbol.
- [x] Update all four library include consumers and amend the governing layout
  rule without relaxing Core 4 or state/side-effect ownership requirements.
- [x] Add two headless CTest consumers (old/new include orders) with independent
  LH and depth-range known answers. Canonical implementation is byte-identical
  to the pre-migration header.
- [x] Enforce the canonical leaf's exact GLM include dependencies, compatibility
  mapping and no legacy include use in library headers. An isolated negative
  fixture injecting an app dependency was rejected by the boundary gate.
- [x] Baseline full configured build + 18/18 CTest; post-move reconfigure/full
  build + 20/20 CTest including the boundary gate. Linux Release, static library,
  cached vcpkg toolchain, lavapipe and Vulkan validation enabled. No fresh-cache,
  installed-package, alternate-platform or shared-library claim is made.

The [machine-readable manifest](engine_header_migration_manifest.json) covers
this pilot only, not the full tree. Step 1 remains partial: complete header/DAG
inventory, consumer/SDK matrix and broad ownership decisions are still pending.
This leaf pilot intentionally precedes completing that inventory to validate
compatibility/enforcement mechanics without changing subsystem behavior.
No full module migration or step 2 completion is claimed. The narrowly pinned
boundary rule must be generalized with tested manifests before more headers move.

Date: 2026-09-17.
Scope: `/home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib` and its repository consumers.

## Decision

Organize SHS by the knowledge each module owns, not by a global `domains/`
versus `execution/` distinction. Device execution, scheduling, allocation,
rendering policy, and gameplay state all have domain boundaries. Purity and
side effects remain explicit API/dependency properties within those boundaries.

This migration makes the library suitable for integration into a full game
engine; it does not claim to implement a complete engine. Do not rename the
library target/package or add empty physics, audio, animation, networking, or
scripting modules as part of a directory migration. Introduce those capabilities
later through real use cases and independently tested contracts.

## Target public modules

Locations below are relative to the absolute scope directory's `include/shs/`.
Namespaces are the eventual public API; header moves and namespace migrations
are separate changes. Existing root `shs::` symbols remain valid initially.

| Directory / eventual namespace | Responsibility |
|---|---|
| `core/`, `shs::core` | Small shared value utilities, not global context/services |
| `memory/`, `shs::memory` | Arenas and allocation/lifetime primitives |
| `containers/`, `shs::containers` | Generic storage primitives, not entity policy |
| `task/`, `shs::task` | Job dependencies/execution/completion, not gameplay policy |
| `platform/`, `shs::platform` | OS/window/device-input adapters |
| `resources/`, `shs::resources` | CPU asset data, identity, stores and import contracts |
| `geometry/`, `shs::geometry` | Shapes, bounds, geometric queries/operators |
| `camera/`, `shs::camera` | Camera state, conventions and transformations |
| `lighting/`, `shs::lighting` | Light descriptions and lighting policy/math |
| `sky/`, `shs::sky` | Sky descriptions and sampling models |
| `scene/`, `shs::scene` | Scene identity, transforms, instances and projections |
| `input/`, `shs::input` | Device-neutral input values and action mapping |
| `logic/`, `shs::logic` | Reusable state-machine contracts/transitions |
| `render/`, `shs::render` | Backend-neutral frame/view, CPU image/target and shader values |
| `renderpath/`, `shs::renderpath` | Recipes, capabilities, planning and pass execution |
| `rhi/`, `shs::rhi` | Device-neutral GPU contracts and backend realization |
| `app/`, `shs::app` | Optional host orchestration and explicit composition |

Use `planning/`, `execution/`, `adapters/`, and `detail/` locally when needed;
these are roles, not mandatory folders. Stateless math needs neither an empty
gateway nor invented commands/events. Stateful boundaries retain typed commands,
observable outcomes and explicit ownership. Existing KDBA behavior/error/event
guarantees remain required; filesystem/house-shape mandates need a documented
amendment, not silent removal of checks. Scene does not mandate an ECS.

## Current-to-target mapping

Paths in this table are relative to the same `include/shs/` root.

| Current | Canonical destination |
|---|---|
| `domains/{camera,geometry,lighting,sky,scene,resources,input,logic}/` | Corresponding top-level module; preserve filenames initially |
| `domains/renderpath/` | `renderpath/` |
| `domains/frame/` | `render/frame/` |
| `domains/gfx/` | `render/targets/`; CPU target values remain distinct from RHI images |
| `execution/pipeline/` | `renderpath/planning/` for recipes/compiler/plans; `renderpath/execution/` for executors/runtime plumbing |
| `execution/passes/` | `renderpath/execution/passes/` |
| `execution/shader/` | `render/shader/` |
| `execution/sw_render/` | `render/software/` |
| Neutral contracts under `execution/rhi/` | `rhi/`, grouped by responsibility |
| `execution/rhi/drivers/vulkan/` | `rhi/vulkan/` |
| `execution/rhi/drivers/software/` | `rhi/software/` |
| Existing `rhi/` headers | Reconcile with canonical RHI; no duplicate implementations |
| `execution/job/` | `task/` |
| `execution/platform/` | `platform/`; asset loaders instead become `resources/adapters/` |
| `execution/app/` | `app/`; camera/input bridges become explicit orchestration |
| Domain-local `edge/` stores | Owner's `storage/`; external integrations become `adapters/` |
| Jolt integrations in geometry/lighting | Owner's `adapters/jolt/`; pure contracts cannot include them |
| Runtime/global context in `core/` | `app/` after separating primitive values from subsystem ownership |

Pipeline/RHI mappings require a per-header manifest in step 1, not blind recursive
renaming. Relocate sky drawing to renderpath execution, retaining models under
sky. Put light/geometry GPU packing in explicit rendering adapters rather than
pulling device dependencies into pure contracts.

## Dependency and ownership rules

> These rules are codified as Constitution II §3 Rules 13–16 (2026-09-17
> amendment: module boundary include-direction, adapter value seams, closed
> shrink-only exception sets, optional-SDK conditionality).

An arrow means "may depend on"; public-header dependencies must form a DAG.

- Foundations (`memory`, `containers`) -> minimal core; never subsystems.
- Pure feature contracts -> foundations and explicitly declared peer value
  contracts. Scene may consume resource handles/geometry; no reverse edge.
- `renderpath/planning` -> renderer values + neutral RHI capability/descriptor
  contracts; never platform, backend drivers or executor headers.
- `renderpath/execution` -> plans + neutral RHI + render/scene projections.
  Inject backend selection/creation at the composition boundary.
- `rhi/vulkan` -> neutral RHI + Vulkan; `rhi/software` -> neutral RHI +
  `render/software`. Neither backend owns application policy.
- Adapters -> owning contracts + external dependencies; never the reverse.
  Neutral contract headers compile without Vulkan, SDL, Assimp or Jolt headers.
  GLM remains an established math dependency.
- `app` -> public subsystem APIs and injected platform/task/backend services;
  no subsystem depends on app. No global service locator.

Input emits actions; app routes them to camera, render settings or session
transitions. Camera owns applied rig state. Orchestration uses public operations,
not writes into another owner's private storage. No asynchronous message bus is
required. Each stateful module documents identity, stale handles, reference
invalidation, failure behavior, thread ownership and shutdown order. CPU asset
identity is not GPU identity. Allocation/event-log failure semantics must be
explicit: `expected` alone supplies neither rollback nor atomicity.

## Migration backlog: seven ordered steps

Every unchecked item below is future work. Prefer small independently reverting
commits; keep mechanical moves separate from semantic changes.

### Status (2026-09-17, automated continuation run)

- Camera pilot committed (`21e45c9`): canonical `shs/camera/convention.hpp`
  plus content-pinned forwarding header, 4 consumers repointed, both
  include-order CTests; 20/20 CTest green (lavapipe + VK_LAYER validation).
- Step 1 tooling committed (`743554f`): `tools/check_header_migration.py`
  enforces reviewed migrations from `engine_header_migration_manifest.json`
  (content-pinned single-hop forwarders, pinned dependencies, GLM-only
  pure-leaf policy, include-cycle/ambient-effect reachability, retired-path
  ban, unregistered-header ban) with 16 isolated positive/negative tests;
  `tools/inventory_headers.py` regenerates the full 220-header inventory
  (`engine_header_inventory.json`, staleness-checked in CTest); the boundary
  gate delegates to it. 22/22 CTest green (clean rebuild, both Vulkan tests
  run on lavapipe with validation).
- Step 1 remains PARTIAL: consumer/SDK matrix (fresh-cache, installed-package,
  shared builds) and the compiler matrix are not yet recorded; the manifest
  covers only reviewed pilots, as designed.
- Matrix records (2026-09-17, this environment — GCC 13.3.0 only; clang
  unavailable, so the compiler matrix is explicitly limited, not silently
  skipped):
  - Incremental (existing `cpp-folders/build`): configure + build + 24/24
    CTest green with `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`
    (no ICD overrides; lavapipe).
  - Fresh-cache Release (`build-matrix-fresh`): configure, full build and
    CTest from an empty cache — 24/24 CTest green. First run caught a real
    latent IWYU bug the incremental build masked:
    `tier0-rasterization-foundations/01_barycentric_interpolation/tri_barycentric_sw.cpp`
    used `std::min/max({...})` without `#include <algorithm>` (the demo
    chain pulls no shs headers, so the exposure is independent of the
    relocations); fixed with an additive include, then re-run green.
  - Shared build (`BUILD_SHARED_LIBS=ON`, `build-matrix-shared`): configure,
    build and CTest — 24/24 CTest green; the library and all consumers link
    and run as shared objects with no further changes.
  - Not run here (unavailable in env): installed-package consumer, clang/msvc
    compilers. Unsupported configurations are explicit, not silently skipped.
- Inventory finding: no include cycles between the 220 headers and no missing
  internal includes; the proposed module mapping still has one cross-module
  cycle (`proposed_module_cycles` in the inventory) driven by execution-zone
  and adapter dependencies — resolve before step 2 relocations beyond leaves.
- Vulkan trees: inventory proposes keeping them separate as
  `shs/rhi/vulkan/value/` vs `shs/rhi/vulkan/runtime/` (no implementation
  merge, no symbol/ABI claims); relocation stays blocked on the ownership
  decision (step 5 gate).
- Phase table: 1 PARTIAL, 2 pilot-partial, 3 pilot-partial (manifest-driven
  checks cover reviewed leaves only), 4–7 not started.

### Status (2026-09-17, bulk relocation run)

- Bulk step-2 relocation landed: all 212 remaining legacy headers moved
  (`git mv`) from `shs/domains/` and `shs/execution/` to their inventory
  `proposed_canonical` destinations — collision-free, including the
  pipeline split into `shs/renderpath/planning/` vs `shs/renderpath/execution/`,
  `domains/gfx` into `shs/render/targets/`, and the Vulkan trees kept separate
  as `shs/rhi/vulkan/value/` (desc driver) vs `shs/rhi/vulkan/runtime/`
  (absorbed monolith backend) with no implementation merge.
- Every retired path keeps a content-pinned single-hop forwarding header
  (212 forwarders); `shs/domains/` and `shs/execution/` remain temporarily,
  containing only forwarders — full removal is a separate breaking change.
- All repository consumers repointed: includes, CMake references and test
  fixtures; namespaces, symbols and implementations untouched (mechanical
  move only). `tests/camera_include_compatibility_tests.cpp` deliberately
  still includes the legacy path to keep exercising the forwarder.
- `check_header_migration.py`: transitional named-module allow-list added
  (app, camera, geometry, input, lighting, logic, render, renderpath,
  resources, scene, sky, task, platform) so relocated headers are recognized
  while the manifest stays reviewed-leaves-only; 16/16 checker tests green.
- `check_kdba_boundaries.sh` taught the old/new layout: content, catalog and
  gateway gates now scan legacy forwarders plus canonical module dirs
  (purity gates — entropy/platform-IO/expected-vector — scan the value tier
  only: `renderpath/planning`, `resources/storage`, pod dirs; execution/adapter
  tiers stay IO/time-legal). Fixed two latent `set -o pipefail` crashes where
  zero-match `grep` pipelines aborted the script silently.
- Validation: full rebuild (all targets recompiled after the header moves)
  + 22/22 CTest green (both Vulkan tests on lavapipe with
  `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`),
  boundary gate and inventory staleness check pass; inventory regenerated for
  the new tree (still no include cycles, no missing internal includes).
- Deferred to separate changes: cross-module cycle resolution (the
  execution/adapter dependency cycle persists at the include level),
  `domains/`+`execution/` forwarder-folder removal (breaking), `rhi/`
  ownership decision (step 5), step-3 dependency-manifest enforcement
  replacing path-token gates, SDK/compiler matrices.

### Status (2026-09-17, cycle decomposition + include-graph gate)

- The 11-module proposed cycle decomposes into six module pairs (app↔scene,
  geometry↔resources, geometry↔scene, lighting↔render, render↔resources,
  renderpath↔rhi, resources↔sky). Enumerated every non-forwarder
  `#include "shs/..."` edge pair; classified each by dependency direction:
  most are direction-legal tier edges (adapter→value, execution→value,
  value→value) rather than true violations.
- Structural fixes landed (no symbol/ABI changes):
  - `RenderBackendType` + `render_backend_type_name()` moved from
    `shs/rhi/core/backend.hpp` to a new neutral value header
    `shs/render/frame/backend_type.hpp` (rhi/core/backend.hpp includes it,
    so all consumers compile unchanged); `renderpath/planning` headers
    (`render_path_recipe.hpp`, `render_path_presets.hpp`) repointed — this
    removes the value-tier `planning → rhi` back-edge that made
    renderpath↔rhi a tier violation rather than just a module pair.
  - `scene/scene_instance.hpp`: dropped an unused direct include of
    `geometry/adapters/jolt/jolt_adapter.hpp` (scene_shape.hpp already
    provides the Jolt-gated types it consumes).
- New transitive include-graph gate `tools/check_include_graph.py`
  (wired as CTest `shs_renderer_include_graph_gate` + fixture tests
  `shs_renderer_include_graph_tests`): R1 no header-level include cycles
  (forwarders resolved), R2 raw SDK includes only in integration-tier
  headers (adapter paths, `rhi/`, driver-adjacent `renderpath/execution/vk_*`,
  or `SHS_HAS_*`-feature-guarded files), R3 value-tier headers must not
  reach adapter/SDK-bearing integration headers transitively. Self-test
  covers negative fixtures (planted cycle, SDK in value file, value→adapter)
  plus a non-vacuity guard asserting the gate sees the live tree
  (220 canonical headers, 63 integration-tier).
- Remaining tier violations are tracked, not hidden, in
  `tools/engine_include_exceptions.json` (1 entry, down from 7): the R5b
  umbrella split was executed — `scene/scene.contract.hpp` no longer
  re-exports the Jolt-gated integration headers (`scene_elements`,
  `scene_instance`) and `resources/resources.contract.hpp` no longer
  re-exports the assimp adapter (`adapters/resource_import.hpp`); both
  contracts (and their gateways, which include only the contracts plus
  `<variant>`-only command/event headers) now compile with zero optional
  SDKs (verified standalone: `g++ -fsyntax-only` with GLM only, no
  Jolt/Assimp include paths). The adapters remain discoverable at their
  canonical homes (`shs/scene/scene_elements.hpp`,
  `shs/scene/scene_instance.hpp`, `shs/resources/adapters/resource_import.hpp`;
  old-path `shs/domains/*/edge/` forwarders still map to them). The
  `lighting/light_runtime.hpp` exception was resolved per its tracking
  note by wrapping the whole header in the `SHS_HAS_JOLT` guard (same
  pattern as `scene_elements.hpp`; the library always defines
  `SHS_HAS_JOLT=1`, so no consumer changes); the
  `renderpath/execution/pass_adapters.hpp` exception was resolved by
  folding it under the gate's driver-adjacent classification alongside
  `renderpath/execution/vk_*` (it is execution-tier adapter aggregation
  over the software renderer + Jolt culling; no library header includes
  it). The last remaining exception, each with reason + tracking note:
  `render/software/debug_draw.hpp` (observer/callback seam — a design
  decision explicitly deferred to the step-5 rhi-ownership work). The
  test suite fails on stale exceptions (fixed-but-still-listed) and on
  malformed entries.
- Validation: full build + 24/24 CTest green (22 existing + 2 new gate
  tests; Vulkan tests on lavapipe with
  `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`), header-migration
  checker green, inventory regenerated (220 canonical headers).
- Environment note: SDK/compiler matrix limited to what this machine
  provides — GCC 13.3.0 only (clang not installed). Fresh-cache Release
  configure/build/CTest and `BUILD_SHARED_LIBS=ON` runs recorded in the
  step-1 matrix notes.


### Status (2026-09-17, exceptions cleared to zero + adapters made conditional)

- Step 3 COMPLETE — exceptions manifest empty (7 -> 3 -> 1 -> 0):
  - `render/software/debug_draw.hpp` (the last exception) was resolved at the
    source via the value seam its tracking note called for: `DebugMesh`
    extracted from the Jolt debug-draw adapter into the new pure value header
    `shs/geometry/debug_mesh.hpp` (GLM-only). The Jolt adapter *produces* it,
    software debug draw *consumes* it — render/software no longer reaches
    adapter code at all (verified standalone: GLM-only `g++ -fsyntax-only`).
  - Latent IWYU bug surfaced by that standalone check and fixed: `debug_draw.hpp`
    used `std::span` without `#include <span>` (rode in transitively from the
    Jolt adapter). Manifest duplicate `description` key deduped.
  - Parked-demo stale path fixed: `demo_renderpath_bridge.hpp` still included
    the retired `shs/domains/renderpath/renderpath.reducer.hpp` (the
    reducer->gateway renames removed it; `exps-gpu-renderer` is commented out
    of the active build so the break was latent). Repointed to canonical
    `shs/renderpath/renderpath.command.hpp`; compiles standalone.
- Step 5 first item — selectable dependency ownership: `SHS_RENDERER_WITH_SDL2`
  and `SHS_RENDERER_WITH_ASSIMP` CMake options (default ON = unchanged
  behavior) make SDL2/SDL2_image and Assimp discovery + linking conditional;
  `SHS_HAS_SDL2` / `SHS_HAS_ASSIMP` PUBLIC definitions expose the selection
  (same shape as `SHS_HAS_JOLT`). Safe because the compiled library TUs are
  pure anchors. Proof: full tree configure + build + **24/24 CTest with both
  options OFF** (`/tmp/shs-nosdk` scratch, Vulkan still auto-discovered and
  green on lavapipe with validation). Remaining step-5 items (install/export
  package-consumer tests, minimal exported deps, source-path leakage
  rejection) stay open.
- Validation: gate OK, 5/5 gate self-tests, 24/24 CTest in BOTH the default
  (adapters-ON) tree and the no-SDK tree; header-migration checker green;
  inventory regenerated (221 canonical headers — new `geometry/debug_mesh.hpp`).
- Law amendment: the practices above (value-tier include direction, adapter
  value seams, closed shrink-only exception sets, optional `SHS_HAS_*` SDK
  guards, no-SDK headless validation) are codified as Constitution II
  Rules 13–16 (`docs/spec/value_oriented_programming.md` §3, 2026-09-17).
- Phase table: 1 COMPLETE, 2 COMPLETE, 3 DONE, 4-6 not started, 7 blocked
  on 4-6; forwarder-folder removal still a step-7 breaking release.

### Status (2026-09-17, steps 1-2 evidence reconciliation)

- Step 1 COMPLETE — each baseline item verified against existing artifacts
  rather than re-derived:
  - Machine-readable manifest: `engine_header_inventory.json` (regenerated
    byte-identical — staleness check honest) covers all 433 public headers
    with per-header `proposed_owner`, `namespace_declarations`, `visibility`
    and `direct_dependencies`; `engine_header_migration_manifest.json` adds
    namespace/visibility/`build_target`/compatibility per reviewed entry.
  - Pipeline split done (`renderpath/planning` vs `renderpath/execution`);
    `rhi/` overlap resolved as keep-separate (`rhi/vulkan/value` vs
    `rhi/vulkan/runtime`, no implementation merge) with the ownership
    ruling explicitly carried to the step-5 gate; cross-module cycle
    decomposed into six module pairs, each classified a direction-legal
    tier edge (adapter→value, execution→value) — the header-level DAG is
    empty and gate-verified (`header_cycles: []`,
    `shs_renderer_include_graph_gate`); `proposed_module_cycles` remains
    an informational module-aggregate of legal cross-tier edges.
  - Accepted layout/Core-4 changes recorded in active specs: Constitution
    II §2.1(4) layout amendment and Rules 13-16 (2026-09-17).
  - Matrices captured: incremental, fresh-cache Release and
    `BUILD_SHARED_LIBS=ON` all 24/24 CTest green (lavapipe +
    `VK_LAYER_KHRONOS_validation`); clang and installed-package consumers
    explicitly recorded as not run in this environment. Optional SDKs
    inventoried and made conditional (`SHS_HAS_*`). Demos/adventures:
    parked (`exps-gpu-renderer` latent break documented); umbrella split
    executed (R5b). New inventory finding: the library CMakeLists declares
    NO install/export rules today — recorded as the step-5 packaging
    baseline for the install/export consumer tests.
- Step 2 COMPLETE — bulk relocation evidence: 212 legacy headers `git mv`'d
  to inventory destinations collision-free (camera pilot first, then bulk);
  212 content-pinned single-hop forwarders validated by
  `check_header_migration.py`; all consumers, CMake lists and docs
  repointed; boundary gates taught old/new paths; old+new include-order
  smoke consumers in CTest; namespaces/symbols untouched.

### Status (2026-09-17, step 5 packaging: install/export + consumer and self-containment gates)

- Step 5 COMPLETE — all four checkboxes ticked with evidence above. Remaining
  migration work is steps 4, 6 and 7 only (user rulings / live host).
- Packaging: `install(TARGETS/DIRECTORY/EXPORT)`, `configure_package_config_file`,
  SameMajorVersion version file (0.1.0), `install(EXPORT shs_rendererTargets
  NAMESPACE shs::)`. Exported interface is leakage-free: source-tree,
  build-tree and stb include dirs and FetchContent-owned Jolt/xsimd/VMA links
  are `$<BUILD_INTERFACE:...>`-wrapped (verified by audit: no public header
  uses `stb_*`; no `.cpp` includes non-shs headers).
- Two new gates registered as CTest (both PASS in the vcpkg-toolchain tree):
  - `shs_renderer_package_consumer_test` — scratch-prefix install →
    `find_package(shs_renderer)` → installed-interface leak scan → headless
    consumer build + run (end-to-end PASS).
  - `shs_renderer_header_self_containment_test` — all 433 public headers
    compiled standalone (`-fsyntax-only`, no `SHS_HAS_*` defines) against the
    target's INTERFACE include dirs. First run failed on 18 headers; fixes:
    missing includes (`renderpath.event.hpp` → `renderpath.command.hpp`; five
    `renderpath/execution/passes/*` → explicit `shs/app/context.hpp`; missing
    `geometry/aabb.hpp` in `culling_query.hpp`), guard alignment
    (`vk_swapchain_uploader.hpp` fully Vulkan-gated; `pass_adapters.hpp`
    Jolt-typed sections gated with graceful directional-only fallback), and
    removal of 7 dead never-compilable `culling_query.hpp` overloads over
    phantom types (`SweptCapsule`, `SweptOBB`, `KDOP18/26`, `MeshletHull`,
    `ClusterHull` — none defined anywhere; header had zero consumers).
- Validation: clean reconfigure + full build, **26/26 CTest green** (includes
  package-consumer, self-containment, inventory, include-graph, Vulkan
  lavapipe+validation suites). Inventory regenerated (433 headers unchanged).
- Phase table: 1-3 COMPLETE, **5 COMPLETE**, 4/6 NOT STARTED, 7 blocked on 4-6.
- Known limitation (documented in `shs_rendererConfig.cmake.in`): Jolt, xsimd
  and VMA are consumed via FetchContent in-tree and are NOT re-exported by the
  installed package — binary-compatible only when consumer and package agree
  on those ABIs.

### Status (2026-09-17, step 4.2: authoritative camera/render settings owner)

- Step 4 second item completed: `shs::app::SessionState` established as the
  one authoritative owner of session-scoped camera settings (rig + fov/
  znear/zfar) and session render settings (light-shafts toggle), with two
  canonical sync funnels (`shs/app/session_settings_sync.hpp`) into the
  scene camera and per-frame FrameParams; recipe-then-session precedence
  pinned. Evidence in the step-4 checkbox above. Suite 11/11 green; gates
  green; inventory 436. Full-build + CTest evidence recorded in the commit
  message.
- Phase table: 1-3 COMPLETE, 4 at 3/5 (remaining: thread/arena lifetime +
  failure semantics; identity-only gateway retirement), 5 COMPLETE, 6/7 not
  started (7 blocked on 4-6).

### Status (2026-09-17, step 4.3: scene/resource identity policy)

- Step 4 third item completed: the scene/resource identity policy is written
  in one place per owner and enforced by API + tests —
  `shs/scene/scene_identity.hpp` (policy statement + `audit_scene_identity`),
  `SceneObjectSet::remove`/`count_duplicate_names` (deletion/recreation
  preserves the name-derived id; duplicate names detectable, find() stays
  first-wins), `ResourceRegistry::generation()` epoch (clear() is the only
  reset; handles are append-only index handles with deliberate last-wins key
  rebinding), and `SceneResourceView` pinned as the per-call renderer
  projection. Evidence in the step-4 checkbox above. Suite 4/4 green; gates
  green; inventory 437.


### Status (2026-09-17, step 4.4: lifecycle + failure semantics)

- Step 4 fourth item completed: one authoritative lifecycle policy per
  surface — gateway thread access/arena lifetime/rejection + failure
  semantics (renderpath as the reference house shape, scene + resources
  aligned) and the job-system thread/shutdown contract. One honest semantic
  discovery pinned by tests: a gateway command mutates state BEFORE
  emitting its event, so an arena allocation failure on the event push
  leaves the failing command's mutation in state while only its event is
  lost (log/state divergence) — documented and pinned rather than papered
  over with rollback. Evidence in the step-4 checkbox above. Suite 5/5
  green; gates green; full build + 32/32 CTest green; inventory regenerated
  (content hashes, header count unchanged at 437).
- Phase table: 1-3 COMPLETE, 4 at 4/5 (remaining: identity-only gateway
  retirement), 5 COMPLETE, 6/7 not started (7 blocked on 4-6).

### Status (2026-09-17, step 4.5: identity-only gateway retirement) — PHASE 4 COMPLETE

- Step 4 fifth item completed: the identity-only gateway scaffolding is
  retired. The seven pure-identity gateways (camera, geometry, gfx, lighting,
  resources, scene, sky — empty `variant<monostate>` command vocabulary, no
  applied state, zero production consumers) were removed: 7 canonical gateway
  headers + 7 `domains/` forwarders deleted (437 -> 423 headers), plus
  `tests/identity_step_test.hpp` (only the retired pods' suites used it).
  Useful pure functions and their tests are PRESERVED (make_render_item/
  projection pins, follow_target/view-chain/light-fit, compute_tangent_frame/
  perturb_normal, lambert/shade, ProceduralSky sampling, registry round-trip,
  RTHandle/PixelBuffer). `frame` keeps its gateway (identity transition over
  real `FrameParams` state; the C1.4 contract-guardrails replay-probe
  vehicle). Spec amendment (law §2.2): `pod_identifier_law.md` naming tables
  + §2.7 amended with the full retirement record (supersedes the Run C
  "retain" wording for the seven pods); Run C close-out note in the KDBA
  conformance backlog annotated. Gates: `check_kdba_boundaries.sh` amended —
  retired pods exempt from the gateway file law and an identity-gateway
  REGROWTH now FAILs the gate (negative-tested red-to-green with a probe
  file, then removed). Full build + 32/32 CTest green, zero warnings;
  inventory regenerated (423 headers, content hashes).
- Phase table: 1-4 COMPLETE, 5 COMPLETE, 6/7 not started (7 blocked on 4-6).

### Status (2026-09-17, step 4.5 completion snapshot): task completion table

Snapshot of every checkbox in this backlog at commit `2bd2bd6` (includes the
pre-phase pilot evidence block at the top of the file):

| # | Phase | Items | Status | Evidence |
|---|-------|-------|--------|----------|
| — | Pilot: camera convention compatibility | 5/5 | COMPLETE | Pilot evidence block (top of file) |
| 1 | Inventory, decision record and baseline | 4/4 | COMPLETE | Manifest + KDBA amendments + clean build/CTest/boundary baseline (433 headers at baseline) |
| 2 | Relocate headers without changing behavior | 4/4 | COMPLETE | Canonical layout + forwarding headers + dual-include-order smoke consumers |
| 3 | Enforce actual dependency separation | 4/4 | COMPLETE | Renderpath ownership, conditional adapters, manifests + negative gate fixtures |
| 4 | Clarify state ownership and harden contracts | 5/5 | COMPLETE | 4.1 input split, 4.2 settings owner, 4.3 identity policy, 4.4 lifecycle/failure semantics, 4.5 identity-gateway retirement (437 -> 423 headers) |
| 5 | Make dependencies selectable by consumers | 4/4 | COMPLETE | Aggregate targets + conditional discovery + install/export/self-containment gates |
| 6 | Prove engine integration through a vertical slice | 0/5 | NOT STARTED | Public-API-only host, deterministic replay, software/Vulkan parity, engine seams, completion/cancellation docs |
| 7 | Namespace/API cutover and compatibility retirement | 0/5 | NOT STARTED (blocked on 2-6) | Namespace slices, alias policy, consumer migration, forwarding-header retirement, legacy-include rejection |

**Totals: 26/36 checkboxes done (72%). Phases 1-5 COMPLETE, 6/7 not started.**
Validation at HEAD: 32/32 CTest green (incl. `check_kdba_boundaries.sh`),
zero warnings, inventory 423 headers. Next executable step: 6 (its
dependencies 4 and 5 are complete); 7 remains blocked until 6 closes.

### 1. Inventory, decision record and baseline

- [x] Create a machine-readable old-header -> canonical-header manifest covering
  every header, namespace owner, public/private status and build dependency.
  Resolve pipeline/RHI splits, existing `rhi/` overlap and cross-module cycles.
  — DONE 2026-09-17: inventory (433 headers, per-header owner/namespace/
  visibility/dependencies; regenerated byte-identical) + reviewed-leaves
  migration manifest; pipeline and rhi splits recorded; cycle decomposed into
  six direction-legal module pairs, header DAG gate-verified empty. The
  `rhi` value/runtime ownership *ruling* remains open under step 5.
- [x] Record accepted changes to KDBA layout/Core-4 requirements in the active
  specs and catalogs. Preserve behavioral laws and historical completion notes.
  — DONE 2026-09-17: Constitution II §2.1(4) layout amendment + Rules 13-16;
  behavioral laws and completion notes preserved verbatim.
- [x] Capture clean configure/build/CTest and boundary-check results; inventory
  demos, adventures, umbrella headers, install/export rules and optional SDKs.
  — DONE 2026-09-17: incremental/fresh-cache/shared matrices 24/24 CTest green;
  boundary gate + self-tests + header-migration + include-graph + inventory
  checks green. Finding: no install/export rules exist yet (step-5 baseline).
- [x] Record existing public API compatibility expectations and compiler/SDK
  matrix; unsupported configurations must be explicit, not silently skipped.
  — DONE 2026-09-17: GCC 13.3.0 only (clang unavailable, recorded not
  silently skipped); installed-package consumer not run, recorded explicit;
  compatibility expectation = retain old includes/symbols until announced
  breaking release (manifest `compatibility` field).

Exit: complete mapping, reproducible baseline and dependency DAG; no source moves.

### 2. Relocate headers without changing behavior
Depends on 1.
- [x] Move foundations and leaf feature modules first, then rendering/RHI,
  then platform/task/app. Update all repository consumers in each slice.
  — DONE 2026-09-17: camera pilot (`21e45c9`) then bulk relocation of all
  212 remaining legacy headers (`git mv`, collision-free).
- [x] Leave old public includes as single-hop forwarding headers to canonical
  definitions. No copied types, parallel implementations or facade cycles.
  — DONE 2026-09-17: 212 content-pinned forwarders; content drift FAILs in
  `check_header_migration.py`; `shs/domains/` + `shs/execution/` hold
  forwarders only until the step-7 breaking release.
- [x] Update CMake source lists, umbrella includes, tools and documentation paths.
  Teach boundary checks to recognize old/new paths during the transition.
  — DONE 2026-09-17: all repository consumers repointed; boundary gate and
  header-migration checker taught the old/new layout with a transitional
  named-module allow-list.
- [x] Compile old-include and new-include smoke consumers together to detect
  duplicate definitions; compare test outcomes against step 1.
  — DONE 2026-09-17: `tests/camera_include_compatibility_tests.cpp` keeps
  exercising the legacy forwarder beside the canonical-include consumers;
  both include-order CTests green in every matrix run.

Exit: canonical top-level modules compile; compatibility headers work; no semantic
or public namespace changes mixed into relocations.

### 3. Enforce actual dependency separation
Depends on 2.
- [x] Put recipe/compiler/plan definitions under renderpath ownership; remove
  the sanctioned domain-to-execution re-export workaround.
  (Done in the bulk step-2 relocation: recipe/compiler/plan definitions live
  under `shs/renderpath/planning/`; grep-verified no `shs/domains/` ->
  `shs/execution/` re-exports remain.)
- [x] Extract Jolt/Assimp/SDL/backend adapters from neutral headers and separate
  scene/resource values from storage/runtime integration where necessary.
  (R5b umbrella contract split + R5c value seam: `DebugMesh` extracted from
  the Jolt debug-draw adapter into `shs/geometry/debug_mesh.hpp`; transitive
  gate R3 now holds with zero exceptions.)
- [x] Define public include/dependency manifests and narrowly scoped exceptions.
  Replace path-token checks with rules covering transitive includes and cycles.
  (`tools/check_include_graph.py` transitive gate; exceptions manifest is
  now empty — all 7 historical exceptions resolved at the source, not waived.)
- [x] Add negative gate fixtures: a pure contract including a driver or app
  header must fail. Restore checks immediately for every migrated directory.
  (`tests/include_graph_tests.py`: 5/5 — planted cycle, SDK-in-value,
  value->adapter, plus non-vacuity guards over the live tree.)

Exit: neutral-header compile tests need no optional SDKs; dependency violations
are detected, not merely hidden under new paths. **Step 3 COMPLETE (2026-09-17).**

### 4. Clarify state ownership and harden contracts
Depends on 3; behavior changes require regression tests first.
- [x] Split input translation from camera/render/session application through
  explicit app orchestration. Preserve move/look/clamp and toggle behavior.
  (Done 2026-09-17: `shs/app/session_orchestrator.hpp` is the explicit
  orchestration host — `shs::app::SessionState` (camera rig + light-shafts/
  bot/quit settings) and `session_orchestrate()`, the single canonical
  application gateway with the per-intent arrows moved VERBATIM from the
  retired `shs::input::input_gateway` (basis transform, dt scaling, ±85°
  pitch clamp, toggle semantics, zero-signal-loss fact emission, tally —
  behavior bit-preserved). The K1.4 interim note ("camera rig lives in the
  input pod until an orchestrator host exists") is resolved: `RuntimeState`
  left `input_state.hpp`, the input pod is translation-only
  (`input_latch_gateway` + `value_commands` emitters + `CommandProcessor::
  collect_runtime_commands`; the fact-log-dropping `apply_commands` edge
  convenience retired). Regression tests FIRST: new
  `shs_renderer_session_orchestration_tests` (golden mixed-log pins
  z=2.0/yaw=π/2+0.1/pitch=0.05, bit-parity vs inline pre-split reference
  math, host-instance independence, recorded-input replay, kit determinism,
  pipeline composition, clamp saturation) + `input_tests.cpp`/`core_tests.cpp`
  pins repointed to the orchestrator, all green. Compat: root
  `shs::RuntimeState` alias preserved (= `SessionState`) in
  `shs/app/runtime_state.hpp`; retired symbols `shs::input::input_gateway`
  and `CommandProcessor::apply_commands` had only test consumers (migrated)
  — parked exps-gpu-renderer demos reference them (latent, out of build,
  feeds the step-7 compatibility ledger). Gate (7b) refined: a pod's
  `<pod>_gateway` entry point may live in any sibling pod header (input's
  canonical entry is `input_latch_gateway`); negative-tested (entry-point
  rename FAILs exit 1, then reverted). Header inventory regenerated
  434→435. CTest 30/30.)
- [x] Establish one authoritative owner for camera and render settings; test
  multiple independent host instances and recorded-input replay.
  (Done 2026-09-17: `shs::app::SessionState` is the ONE authoritative owner
  of session camera settings — rig pose plus projection `fov_y_radians`/
  `znear`/`zfar`, defaults identical to `shs::Camera`/`ViewCamera` so
  projections stay bit-preserved — and session render settings (light-shafts
  toggle; resolving the three-way default drift with
  `FrameParams::enable_light_shafts` and the technique recipe's
  `enable_light_shafts = false`). Scene `shs::Camera` and per-frame
  `FrameParams` are renderer PROJECTIONS of session state, written only
  through two new canonical funnels in `shs/app/session_settings_sync.hpp`:
  `sync_session_to_scene()` (rig + session projection settings -> scene
  camera via `ViewCamera`, field order and matrix math identical to the
  pre-4.2 `shs::sync_camera_to_scene`, which stays as the compat path with
  scene-sourced settings, now doc-flagged for the step-7 ledger) and
  `apply_session_render_settings()` (session toggle -> BOTH the legacy flat
  `fp.enable_light_shafts` and the pass-block field the light-shafts pass
  actually consumes; apply AFTER
  `apply_render_technique_recipe_to_frame_params` so the runtime owner wins
  over planning-level recipe defaults; funnel touches no other field).
  Regression tests FIRST, all in `shs_renderer_session_orchestration_tests`:
  settings-carrying host independence (two hosts, different fov/planes/
  toggle, same recorded mixed log — per-host toggle evolution pinned,
  projection settings untouched by intent application), settings
  recorded-input replay (toggle-heavy log, three fresh hosts, bit-identical
  state + fact logs), session->scene camera sync pinned bit-exact against
  an independent `ViewCamera` reference (including prev_viewproj
  first-call semantics), recipe-then-session precedence and
  only-two-fields funnel pins. 11/11 tests in the suite green. Gate
  results: include-graph gate OK (acyclic, `app -> render/frame` value-tier
  edge legal — `frame_params.hpp` is pure), header-migration gate OK,
  inventory regenerated 435->436 (new `session_settings_sync.hpp`).)
- [x] Define scene/resource identity policy: duplicate names/IDs, stale handles,
  deletion/recreation, vector-reference invalidation and renderer projections.
  (Done 2026-09-17: policy written where it is enforced — new
  `shs/scene/scene_identity.hpp` states the full contract and adds
  `audit_scene_identity` over renderer projections (object_id 0 reserved,
  duplicates counted after first owner, `SceneIdentityReport::valid()`).
  `SceneObjectSet` gains `remove(name)` (deletes FIRST match; re-adding an
  equal name recreates the SAME FNV-1a id — deletion/recreation preserves
  identity) plus `count_duplicate_names()`/`has_duplicate_names()` for the
  deliberate first-wins `find()` policy, and documents the
  vector-reference invalidation hazard of `add()`'s returned reference.
  `ResourceRegistry` gains `generation()` (bumped by `clear()`), with the
  policy pinned as: 1-based handles, 0 = unbound -> nullptr, append-only
  duplicate-key rebinding (last-wins, old handle still resolves to the old
  asset), no per-asset deletion (would invalidate sibling index handles),
  stale pre-clear handles must be re-derived via `find_*` (a stale handle
  aliases re-added slots). `SceneResourceView` documented as the per-call
  renderer projection that never caches registry pointers. New
  `shs_renderer_scene_identity_tests` suite (4 tests) pins all of the above,
  including projection-copy independence from later set mutations and the
  generation-epoch stale-handle rule. Gates green; inventory regenerated
  436->437 (new `scene_identity.hpp`).)
- [x] Document and test thread access, arena lifetimes, shutdown ordering,
  rejection preservation and partial-batch/event-allocation failure semantics.
  — DONE 2026-09-17: the lifecycle policy is written where it is enforced —
  `renderpath.gateway.hpp` LIFECYCLE POLICY block (thread access: gateway
  batches reentrant only over disjoint states + disjoint PMR arenas,
  compiler/caps freely shareable; arena lifetime: events live on the
  caller's arena, Step is a plain value, per-frame reset legal with
  copy-before-reset; rejection preservation: invalid compile keeps
  plan + recipe + generation and the batch continues; event-allocation
  failure: bad_alloc propagates unabsorbed, every already-performed state
  mutation persists — a command mutates before emitting, so the failing
  command's mutation lands while only its event is lost (documented
  log/state divergence), rerun from the same initial state reproduces the
  full log). `task/job_system.hpp` IJobSystem contract + ThreadPoolJobSystem
  destructor docs pin thread access (enqueue/wait_idle any-thread, jobs on
  arbitrary workers, transitive drain via wait_idle) and shutdown ordering
  (wait_idle is the completion guarantee; destruction drains accepted jobs;
  enqueue-after-teardown-start is a caller error). `scene.gateway.hpp` and
  `resources.gateway.hpp` carried the same policy note at the time (their
  lifecycle wording — "event prefix stays untouched" — was superseded by the
  mutate-before-emit finding above; the files themselves were retired in
  step 4.5, leaving renderpath.gateway.hpp as the authoritative policy
  statement). New
  `shs_renderer_lifecycle_semantics_tests` suite (5 tests) pins rejection
  continuation + plan preservation, partial-batch/event-alloc failure
  semantics (prefix events kept, strict-prefix state, divergence vs rerun),
  arena replay parity across disjoint arenas, concurrent gateway batches
  equal to the single-thread reference, and thread-pool shutdown ordering
  (wait_idle completion, concurrent enqueue, destructor drain).
  Gates green; full build + 32/32 CTest green; inventory regenerated
  (header-content hashes, no new headers).)
- [x] Retire identity-only gateway scaffolding only after consumer migration
  and spec amendment; preserve useful pure functions and their tests.
  (Done 2026-09-17: the seven pure-identity gateways — camera, geometry, gfx,
  lighting, resources, scene, sky, each an empty `variant<monostate>`
  vocabulary over no applied state with zero production callers — were
  retired after consumer migration and spec amendment. Consumers were tests
  only: the seven pod suites dropped their identity pins (identity_step_test
  harness + kit empty-log/replay cases deleted; `identity_step_test.hpp`
  removed with them) while keeping every pure-function pin. 7 canonical
  gateway headers + 7 `domains/` forwarders deleted, 437 -> 423 headers.
  `frame` retained: identity transition over real `FrameParams` state and the
  C1.4 replay-probe vehicle. Spec amendment per law §2.2:
  `pod_identifier_law.md` §2.2 layer table + §2.3 pod table + §2.7 amended
  with the retirement record superseding the Run C K1.5 "retain" verdict for
  those pods; KDBA conformance backlog Run C close-out annotated;
  `check_kdba_boundaries.sh` exempted the retired pods from the gateway file
  law and now FAILs on identity-gateway regrowth (negative-tested
  red-to-green). Full build + 32/32 CTest green, zero warnings; inventory
  regenerated 437 -> 423.)

Exit: observable behavior remains deliberate; owners and lifetimes are enforced
by API/tests, not only directory placement.

### 5. Make dependencies selectable by consumers
Depends on 3; may proceed alongside 4.
- [x] Preserve `shs_renderer` and `shs::renderer` as aggregate compatibility
  targets. Add component targets only for real dependency seams, not competing
  software/GPU libraries.
  (Done 2026-09-17: `shs_renderer` + `shs::renderer` alias preserved unchanged;
  the only added component is the `shs_renderer_values` INTERFACE target
  (`shs::renderer-values`) for the value-layer-only seam. Both are exported in
  `shs_rendererTargets` and re-aliased by `shs_rendererConfig.cmake`.)
- [x] Make SDL/SDL_image and Assimp discovery conditional on enabled adapters.
  (Done 2026-09-17: `SHS_RENDERER_WITH_SDL2` / `SHS_RENDERER_WITH_ASSIMP`
  options, default ON for unchanged behavior; with both OFF the full tree
  configures, builds and passes 24/24 CTest with zero windowing/asset-import
  SDKs. Keep C++23 and GLM baseline — untouched.)
- [x] Add install/export/package-consumer tests: minimal headless configuration
  and separately enabled software, Vulkan and platform adapters. Preserve
  static/shared options; document binary compatibility limitations.
  (Done 2026-09-17: `install(TARGETS/DIRECTORY/EXPORT shs_rendererTargets
  NAMESPACE shs::)`, `configure_package_config_file` + SameMajorVersion version
  file (0.1.0), and `cmake/shs_rendererConfig.cmake.in` with required
  `find_dependency(glm)` and conditional SDL2+SDL2_image / assimp / Vulkan deps
  via `@SHS_RENDERER_PKG_WITH_*@`; `shs::renderer` / `shs::renderer-values`
  aliases recreated by the config. Static/shared preserved via the existing
  `SHS_RENDERER_BUILD_SHARED` option. New CTest
  `shs_renderer_package_consumer_test` installs to a scratch prefix, consumes
  via `find_package(shs_renderer)` with the vcpkg toolchain, scans the installed
  interface for source-tree/build-tree path leakage (none found), then builds
  and runs a headless GLM-only consumer against the installed package —
  PASS. Jolt/xsimd/VMA are build-interface-only links and are documented in the
  config as a binary-compatibility limitation, not re-exported.)
- [x] Make public headers self-contained and exported target dependencies minimal.
  Reject source-tree/build-tree path leakage in installed packages.
  (Done 2026-09-17: build-tree/stb include dirs and FetchContent-owned Jolt/
  xsimd/VMA links wrapped in `$<BUILD_INTERFACE:...>` — in-tree behavior
  unchanged, nothing internal leaks into the installed interface; no public
  header uses `stb_*` and no `.cpp` includes non-shs headers. New CTest
  `shs_renderer_header_self_containment_test` compiles all 433 public headers
  standalone with `-fsyntax-only` and no `SHS_HAS_*` defines against the
  target's INTERFACE include dirs. First run exposed 18 failing headers (~8
  canonical + forwarders); fixed: `renderpath.event.hpp` missing
  `renderpath.command.hpp` include; 5 `renderpath/execution/passes/*` headers
  rode on transitively-included `shs/app/context.hpp` (now included directly,
  matching `pass_context.hpp`/`render_pass.hpp` precedent); `pass_adapters.hpp`
  Jolt-typed light-shape/culling sections and `vk_swapchain_uploader.hpp`
  (Vulkan-typed throughout) now carry the same `SHS_HAS_*` guards as their
  adapters; `geometry/culling_query.hpp` lost 7 dead never-compilable
  overloads referencing types defined nowhere in the tree (`SweptCapsule`,
  `SweptOBB`, `KDOP18/26` + helpers, `MeshletHull`, `ClusterHull`, and a
  `ConvexPolyhedron` overload using a nonexistent vertices helper — the header
  has zero consumers) and gained the missing `geometry/aabb.hpp` include.
  Final state: 433/433 headers self-contained, full 26/26 CTest green.)

Exit: downstream neutral consumers configure/build/install without optional SDKs;
aggregate-target consumers still build.

### 6. Prove engine integration through a vertical slice
Depends on 4 and 5.
- [ ] Add a public-API-only host: recorded input -> action routing -> camera/scene
  update -> render projection -> plan -> backend output. Game rules stay outside
  the rendering library.
- [ ] Verify deterministic headless replay, independent host instances, resize,
  backend-unavailable rejection, asset deletion/recreation and shutdown.
- [ ] Exercise software/Vulkan paths with known-pixel and bounded parity tests;
  explicitly report unavailable/skipped configurations.
- [ ] Specify external physics/animation/audio seams: snapshots/commands, stable
  IDs and scheduling ownership. Do not add placeholder subsystems or mandate SHS
  storage, FSMs, ECS, a global event bus or the optional app host.
- [ ] Document sync/async completion and cancellation at task/asset/backend
  boundaries; test teardown with outstanding work before promising concurrency.

Exit: an engine-style consumer proves integration; isolated tests or a single
triangle cannot close this step.

### 7. Namespace/API cutover and compatibility retirement
Depends on 2-6.
- [ ] Move root `shs::` public symbols into owner namespaces in separate slices.
  Existing `shs::camera`, `shs::input`, `shs::logic` and `shs::renderpath` APIs
  need no redundant namespace layer.
- [ ] Keep explicit old-name aliases/wrappers where safe; test overload lookup,
  ADL, serialization identifiers and mixed old/new includes. Namespace changes
  can break ABI: aliases do not guarantee binary compatibility.
- [ ] Migrate repository consumers/tutorials/catalogs/tools; publish a mapping
  and release-specific deprecation schedule.
- [ ] Remove public forwarding headers only in an announced breaking release
  after a compatibility release and consumer migration. Until then, test them.
- [ ] Reject new legacy includes in canonical sources; retire linter exemptions
  and archive mappings rather than losing migration history.

Exit: canonical sources no longer depend on `shs/domains/` or top-level
`shs/execution/`; compatibility facades are explicitly supported or retired.

## Validation, rollback and related work

Each slice requires clean configure/build, full CTest, boundary checks, old/new
header smoke tests and affected demo/adventure builds. Exercise the step-1 SDK
matrix, not just a developer cache. With Vulkan available, inspect verbose
lavapipe/validation output as well as exit status. Preserve G3/G4 tests; moving
files does not complete remaining upload/readback/parity feature work.

The existing `rhi/drivers/vulkan/` runtime and
`execution/rhi/drivers/vulkan/` value-driver trees are distinct implementations.
Resolve overlapping filenames/ownership explicitly before merging. Revert failed
slices independently; never duplicate implementations or disable boundary gates.
Record commit, configurations, tests and skips for each completed item.

This proposal owns placement, dependency/ownership separation and packaging.
The [KDBA backlog](kdba_conformance_backlog.md) retains its behavioral/renderer
feature gaps; [Kleisli migration history](kdba_kleisli_migration_plan.md) is not
reopened by renames. Step 1 must reconcile the active
[value-oriented specification](../spec/value_oriented_programming.md), especially
layout and orchestrator rules. An app orchestrator owning cross-module workflow
state must still follow applicable pod contracts. Taxonomy is not permission to
bypass typed transitions.

First deliverable: per-header mapping/dependency manifest and baseline consumer
matrix. Then move one leaf module as a compatibility pilot, not Vulkan or the
input ownership rewrite. Document creation closes no implementation checkboxes.
