# Engine-ready module boundaries: proposal and migration backlog

Status: proposed target architecture; implementation not started.
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

### 1. Inventory, decision record and baseline
- [ ] Create a machine-readable old-header -> canonical-header manifest covering
  every header, namespace owner, public/private status and build dependency.
  Resolve pipeline/RHI splits, existing `rhi/` overlap and cross-module cycles.
- [ ] Record accepted changes to KDBA layout/Core-4 requirements in the active
  specs and catalogs. Preserve behavioral laws and historical completion notes.
- [ ] Capture clean configure/build/CTest and boundary-check results; inventory
  demos, adventures, umbrella headers, install/export rules and optional SDKs.
- [ ] Record existing public API compatibility expectations and compiler/SDK
  matrix; unsupported configurations must be explicit, not silently skipped.

Exit: complete mapping, reproducible baseline and dependency DAG; no source moves.

### 2. Relocate headers without changing behavior
Depends on 1.
- [ ] Move foundations and leaf feature modules first, then rendering/RHI,
  then platform/task/app. Update all repository consumers in each slice.
- [ ] Leave old public includes as single-hop forwarding headers to canonical
  definitions. No copied types, parallel implementations or facade cycles.
- [ ] Update CMake source lists, umbrella includes, tools and documentation paths.
  Teach boundary checks to recognize old/new paths during the transition.
- [ ] Compile old-include and new-include smoke consumers together to detect
  duplicate definitions; compare test outcomes against step 1.

Exit: canonical top-level modules compile; compatibility headers work; no semantic
or public namespace changes mixed into relocations.

### 3. Enforce actual dependency separation
Depends on 2.
- [ ] Put recipe/compiler/plan definitions under renderpath ownership; remove
  the sanctioned domain-to-execution re-export workaround.
- [ ] Extract Jolt/Assimp/SDL/backend adapters from neutral headers and separate
  scene/resource values from storage/runtime integration where necessary.
- [ ] Define public include/dependency manifests and narrowly scoped exceptions.
  Replace path-token checks with rules covering transitive includes and cycles.
- [ ] Add negative gate fixtures: a pure contract including a driver or app
  header must fail. Restore checks immediately for every migrated directory.

Exit: neutral-header compile tests need no optional SDKs; dependency violations
are detected, not merely hidden under new paths.

### 4. Clarify state ownership and harden contracts
Depends on 3; behavior changes require regression tests first.
- [ ] Split input translation from camera/render/session application through
  explicit app orchestration. Preserve move/look/clamp and toggle behavior.
- [ ] Establish one authoritative owner for camera and render settings; test
  multiple independent host instances and recorded-input replay.
- [ ] Define scene/resource identity policy: duplicate names/IDs, stale handles,
  deletion/recreation, vector-reference invalidation and renderer projections.
- [ ] Document and test thread access, arena lifetimes, shutdown ordering,
  rejection preservation and partial-batch/event-allocation failure semantics.
- [ ] Retire identity-only gateway scaffolding only after consumer migration
  and spec amendment; preserve useful pure functions and their tests.

Exit: observable behavior remains deliberate; owners and lifetimes are enforced
by API/tests, not only directory placement.

### 5. Make dependencies selectable by consumers
Depends on 3; may proceed alongside 4.
- [ ] Preserve `shs_renderer` and `shs::renderer` as aggregate compatibility
  targets. Add component targets only for real dependency seams, not competing
  software/GPU libraries.
- [ ] Make SDL/SDL_image and Assimp discovery conditional on enabled adapters.
  Today CMake requires them; header moves alone cannot make headless consumers
  independent of those packages. Keep C++23 and GLM baseline.
- [ ] Add install/export/package-consumer tests: minimal headless configuration
  and separately enabled software, Vulkan and platform adapters. Preserve
  static/shared options; document binary compatibility limitations.
- [ ] Make public headers self-contained and exported target dependencies minimal.
  Reject source-tree/build-tree path leakage in installed packages.

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
