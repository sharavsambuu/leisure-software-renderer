
# SHS Renderer & Engine Constitution II: KDBA Kleisli Domain Boundary Architecture (Supreme Law, 2026-09-16)

This document is the second constitutional specification of the SHS Engine & Renderer.

- **Constitution I**: `docs/spec/conventions.md` (Units, Coordinate Systems, Physics Bridge, Lighting Semantics, Backend NDC Laws)
- **Constitution II (This Document)**: Value-Oriented Programming (VOP) & Data-Oriented Design (DOD) Architecture
- **Constitution III**: `docs/spec/dod_ecs_architecture.md` (Entity Component System & Memory Chunking)

---

## 1. Purpose

Value-Oriented Programming (VOP) combined with Data-Oriented Design (DOD) is adopted to make the engine's behavior explicit, deterministic, mechanically sympathetic to modern hardware, and trivially scalable across multi-threaded CPU and GPU compute pipelines.

### Expected Outcomes
- **Zero Lock Contention**: Elimination of mutexes, spinlocks, and read/write locks in hot simulation and rendering loops.
- **Predictable Execution & Determinism**: Bit-for-bit reproducible state transitions enabling instant rollback netcode, headless CI balance testing, and time-travel debugging.
- **Hardware Mechanical Sympathy**: Elimination of pointer-chasing and cache misses via Structure of Arrays (SoA) and $\mathcal{O}(1)$ Frame Memory Arenas.
- **Strict Separation of Concerns**: Pure mathematical simulation in the center; hardware drivers, GPU submission, audio DAC, and OS I/O isolated strictly at execution edges.
- **Bounded Domain Navigation**: A Glimmer/Ember-style Domain Pod structure that keeps massive game codebases modular, navigable, and free from cross-domain callback spaghetti.

---

## 2. Constitutional Principle

> **"Keep pure value transformations in the center. Keep side effects at execution boundaries."**

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                 1. INPUT / OS EDGE                                       │
│    [Hardware Poller] ────────► [Action Tokenizer] ────────► std::span<const Action>      │
└────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                             │
┌────────────────────────────────────────────▼─────────────────────────────────────────────┐
│                                2. PURE VALUE CENTER                                      │
│    Current State Snapshot + Actions + Delta Time                                         │
│         │                                                                                │
│         ▼                                                                                │
│    [Pure Reducers: reduce_domain()] ────────────────► New Snapshot + Discrete Event Log  │
│         │                                                                                │
│         ▼                                                                                │
│    [Pure Batch Planners: to_render_items()] ────────► PipelineExecutionPlan (Tokens)     │
└────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                             │
┌────────────────────────────────────────────▼─────────────────────────────────────────────┐
│                               3. EFFECT EXECUTION EDGES                                  │
│    ├─ Multi-Threaded Tiled Rasterizer / Vulkan Submission                                │
│    ├─ SPSC Lock-Free Audio Dispatcher                                                    │
│    └─ Swapchain Upload & Presentation                                                    │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 The KDBA Domain Pod Constitution (Supreme Law, 2026-09-16 — supersedes 2026-09-15 Redux form)

> **"Everything is a Domain Pod. Every stateful subsystem — engine module or demo
> domain alike — is expressed as a pure reducer over four components:
> Types (contract), Command (action), Reducer, Event. Nothing else mutates state."**

This constitution binds **all code in this repository**: every module of
`shs-renderer-lib` and every demo domain (tetris, snake, fps, …) alike.
Concretely:

1. **Core 4, always** — each stateful subsystem declares the four components in
   their own files: `*.contract.hpp` (types), `*.action.hpp` (closed command
   vocabulary), `*.reducer.hpp` (pure transition), `*.event.hpp` (closed event
   vocabulary). Components are never omitted; an empty vocabulary is an explicit
   closed type (Constitution II §6.1/6.2).
2. **Kleisli arrows are the only state transitions (KDBA, 2026-09-16)** — every transition is a composition of atomic arrows `A -> expected<B, DomainError>` via `.and_then()` / `.transform()` / `.or_else()`; switch-case reducer monoliths are forbidden. Deterministic (Rule 4.1); side effects exist only at execution edges.
3. **Cross-domain interaction is Commands in, Events out (KDBA gateway)** (Rule 8.1, Rule 11) — no direct POD writes across boundaries; sagas compose sub-domain Kleisli gateways synchronously inside an orchestrator pod, with immutable Domain Events as egress.
4. **Physical layout mirrors the law** — domain logic lives in `shs/domains/<pod>/`
   (library) or `domains/<pod>/` (demos); execution edges live in the edge zone
   (`shs/execution/…`); primitives in `shs/core|memory|containers`. The structure
   linter enforces this mechanically (roadmap P0.5/P5).
5. **Orchestrators are pods (KDBA saga)** — a multi-domain workflow is coordinated by
   an orchestrator that is itself a Domain Pod with its own contract, actions,
   reducer, and events, composing sub-domain Kleisli gateways (`reserve:and_then(charge)`)
   speculatively in-memory with `.or_else()` compensation. No
   non-pod controller may own cross-domain state (god-object ban).

No subsystem is exempt: the renderer's render path, the scene, input, camera,
lighting, and every game domain are Domain Pods. "Pod-shaped by analogy" modules
(pure-transform planners) are pods whose command/event vocabularies are explicitly
empty, per §6.1 — they still carry all four files.

### 2.2 Law Precedence & Single-Source Rule

The laws above are restated in several places (principle, rules, tables, demos).
To keep one authority per provision:

1. **Precedence order**: §2.1 states *intent*; **§3 numbered rules are the
   enforceable minimum**; §6.2 is the **authoritative Core 4 suffix table**;
   §7.1/§7.2 are the **authoritative memory/layout laws**; the roadmap is
   *schedule*, never law. Where a restatement and a numbered rule disagree, the
   numbered rule wins; where two rules conflict, **the stricter applies**.
2. **Naming**: "Constitution II", "§2.1", and the historical name "Glimmer/Ember
   Pod Standard" denote the same body of law; citations should prefer
   "Constitution §2.1" or "Rule N (§3)".
3. **Restatements must reference, not re-number**: downstream docs (demo
   ARCHITECTURE files, glossary, arch addenda) cite "Rule N (§3)" or "§6.2"
   instead of producing parallel numbering — parallel numbering creates dangling
   citations (this rule exists because one already did: a "Rule 3.2" reference in
   §7.1 was corrected to Rule 2).
4. **KDBA commit rule (replaces the in-place reconciliation)**: Rule 1's "immutable inputs" governs *intent inputs and views* (`std::span<const Action>`); arrows never mutate persistent state in place — purity means deterministic, side-effect-free *decisions* with atomic boundary commit, not owned-buffer copying.


---


---

## 3. Mandatory Rules

1. **Explicit Structs by Value**: All planning, simulation, and query APIs must accept immutable inputs and return explicit value structs by value.
2. **Side-Effect Free Center**: Simulation reducers, AI evaluators, and batch planners must be pure functions with zero hidden globals, zero singleton reads, and zero dynamic heap allocations.
3. **Pre-Resolved Edge Inputs**: Side-effect execution edges (GPU submission, Audio DAC, Disk I/O) must consume pre-resolved, complete execution plans; they must not recalculate planning decisions or query simulation state internally.
4. **Deterministic Reduction (Rule 4.1)**: Kleisli stages/arrows (`verify_*`, `apply_*`, composed pipelines) must be strictly deterministic. Identical initial state snapshots and identical action spans must produce identical resulting snapshots across all target platforms. Non-deterministic factors (RNG seeds, system clocks, hardware inputs) must be tokenized at input edges and passed in explicitly.
5. **Dual-Tier Memory Separation (Rule 5.1)**:
   - **Transient Frame Arena (`FrameMemoryResource`)**: A linear bump allocator reset in $\mathcal{O}(1)$ at frame boundaries. Used exclusively for per-frame command streams, active render batches, temporary polygon clips, and UI draw tokens.
   - **Persistent State Storage (`std::pmr::get_default_resource()`)**: Used for world snapshots, player stats, and persistent entity tables that survive across frame boundaries.
   *Violation*: Assigning persistent state objects from the transient frame arena is strictly forbidden.
6. **Data-Oriented Memory Layout (SoA) (Rule 6.1)**: Hot-path data (physics bodies, bot tables, particles, light grids) must use Structure of Arrays (SoA) and generational index handles (`uint32_t`), avoiding pointer-chasing and Array of Structures (AoS).
7. **Wait-Free Span Contract (Rule 7.1)**: Multi-threaded jobs must be pure functions that take an immutable `std::span<const T>` and write exclusively to a non-overlapping `std::span<U>`. No mutexes, atomics, or spinlocks are allowed inside worker threads.
8. **KDBA Gateway (Rule 8.1)**: Domains never write another domain's PODs. Interaction is Commands in, Events out; sagas compose typed Kleisli gateways synchronously inside orchestrator pods, with immutable **Discrete Event Values** (`CombatEvent`, `QuestEvent`, `InventoryEvent`) as egress to downstream reducers/edges.
9. **C++23 Value Abstractions**: Core APIs must leverage standard value types (`std::span`, `std::string_view` with `constexpr` hashing, `std::variant`, `std::pmr`, `std::expected`) to enforce safety and zero allocation overhead. The library baselines C++23 (Constitution I §10); the pod-idiomatic subset is defined in §8.
10. **Universal Domain Pod Law (Constitution §2.1)**: Every stateful subsystem — in
    `shs-renderer-lib` **and** in every demo — is a Domain Pod with the
    mandatory Core 4 (`*.contract.hpp`, `*.action.hpp`, `*.reducer.hpp`,
    `*.event.hpp`, each in its own file) and all state transitions through its pure
    reducer. Domain logic lives under `shs/domains/<pod>/` (library) or
    `domains/<pod>/` (demos); edges live in the edge zone. Compliance is
    mechanically verified: the structure linter checks Core 4 completeness and zone
    include-direction in CI (roadmap P0.5/P5).
11. **Bounded Contexts (KDBA gateway)**: A bounded context is a suite of cohesive Kleisli pipelines
    operating over a shared set of Domain PODs, bound by one ubiquitous
    language (one error-enum family, one event vocabulary). Stages compose synchronously as Kleisli chains
    (`.and_then()` / `.transform()` / `.or_else()`) — including across contexts inside a saga orchestrator pod (typed gateway); direct POD writes across boundaries stay forbidden (Rule 8.1).
12. **KDBA Saga (Transient Context vs Persistent Invariants)**: Multi-domain workflows execute speculatively in-memory over a transient `SagaContext` (stack/arena, in-flight tokens only) and commit atomically on 100% success; on failure the transient evaporates and persistent PODs stay pristine. Persistent PODs carry zero phantom flags (`is_pending`, `is_locked`, `retry_count`). Validate-before-mutate is preferred; the `.or_else()` compensator consumes the emitted receipt/fact — never ad hoc done-flags. A workflow that takes payment and then aborts without a refunding fact violates this rule (the wallet-leak shape).

---

## 4. Forbidden Patterns

1. **Mixing Planning and Backend Submission**: Invoking GPU/driver calls (`vkCmd...`, `glDraw...`, `SDL_Render...`) or audio DAC writes inside a planning pass or reducer.
2. **Hidden Singleton Mutation**: Reading or writing global state (`Context::Get()`, `AudioEngine::Instance()`, static local caches) inside reducers, AI evaluators, or planners.
3. **Per-Frame Heap Allocation**: Calling standard `malloc`, `new`, `std::vector::push_back` (without a PMR arena), or dynamic memory allocators inside the per-frame update/render loop.
4. **Dynamic Polymorphism in Hot Paths**: Using virtual method dispatch (`vtable`), `dynamic_cast`, or pointer-to-base switching inside simulation entities or rasterizer loops.
5. **Side-Effect Out-Parameters**: Passing mutable references (`&out_projectiles`) to functions that secretly mutate caller state instead of returning explicit value bundles (KDBA: event/fact out-params retired — return `Step` by value; exclusive span out-buffers for hardware kernels per §5(1) remain).
6. **Unbounded Frame Retainers**: Retaining pointers or references to memory allocated within the transient Frame Arena across frame boundaries.

---

## 5. Allowed Exceptions

1. **PMR Output Buffers for Hardware Fast-Paths**: Allocation-sensitive hot paths may write directly into pre-allocated `std::span<T>` or output buffers (`out` params) when memory ownership is explicit and deterministic (event/fact logs excluded — they ride `Step`, never out-params).
2. **Execution Edge Polymorphism**: Virtual interfaces are permitted strictly at the driver boundary (e.g., `IRenderPass::execute_resolved(...)`, `ISwapchainPresenter`) where backend switching occurs outside the value center.
3. **Atomic Ring Queues at Boundaries**: Single-Producer Single-Consumer (SPSC) lock-free atomic ring buffers are allowed exclusively at execution edges (e.g., streaming discrete audio events to the audio thread).

---

## 6. Domain Pod Architecture & Module Directives

To maintain modularity, cognitive clarity, and zero-leak encapsulation across complex projects, all gameplay features and engine modules must follow the **Glimmer/Ember Pod Standard**. Per the Domain Pod Constitution (§2.1, Rule 10), this is **supreme law for the entire repository**: engine library modules and demo domains (tetris, snake, fps, …) are held to the identical Core 4 standard — the library is not exempt, and neither are the demos.

### 6.1 Canonical Domain Pod Structure
Gameplay features are organized as self-contained vertical slices in `domains/<domain_name>/` using standardized file suffixes. Every Domain Pod **must explicitly define the four core components** — **Types (contract), Command/Action, Reducer, Event** — each in its own file. A pod never omits a core component: if a vocabulary is trivially small, it is still declared as an explicit closed type (e.g., `using FooAction = std::variant<std::monostate>;`) so the pod's full state-transition surface remains greppable, auditable, and mechanically checkable.

Multi-domain workflows add a constrained fifth element — the orchestrator/saga
recipe — required only where a workflow spans bounded contexts (Rule 11). The
orchestrator must itself be a Domain Pod (own contract/action/reducer/event);
it composes sub-domain Kleisli gateways synchronously and emits its own
facts as egress. A workflow controller that is not a pod is forbidden.

```text
domains/combat/
├── combat.contract.hpp   # CORE 1. TYPES: plain data structs (ProjectileTableSoA, DamagePacket)
├── combat.action.hpp     # CORE 2. COMMAND: intent tokens (FireIntent, ReloadIntent)
├── combat.event.hpp      # CORE 3. EVENT: emitted event values (EventPlayerFired, EventBotHit)
├── combat.reducer.hpp    # CORE 4. REDUCER: atomic Kleisli arrows (verify_* / apply_* chained via and_then)
├── combat.plan.hpp       # EXT 5.  Pure batch compiler (plan_projectile_mesh, plan_tracers)
└── scripts/
    └── blaster_rules.lua # EXT 6.  Mirrored stateless Lua decision rules
```

The four core components are bound together by the mandatory KDBA Kleisli pipeline (no switch-case monoliths):

```text
validate_cmd(cmd)
  .and_then(verify_state) .and_then(compute_diff) .transform(apply_commit)
  : (State, span<const Action>, dt) -> expected<Step{NextState, Events}, DomainError>
```

**Extension suffixes** (`*.plan.hpp`, `scripts/*.lua`) are conditional: add them only when a Domain Litmus Test (see demo-level pod theory, e.g. tetris `ARCHITECTURE.md` Part I) demands them. The core 4 are not conditional.

### 6.2 The Pod Suffix Laws

**Core 4 — mandatory for every Domain Pod:**

| File Suffix | Component | Required Contents | Strict Restrictions |
| :--- | :--- | :--- | :--- |
| `*.contract.hpp` | **Types** | Value Schemas & Snapshots | Plain data structs only. **No methods, no mutation, no logic.** |
| `*.action.hpp` | **Command / Action** | Intent Tokens / Commands | `std::variant` and enums representing caller intent. Closed vocabulary; `std::monostate` expresses "no intents accepted". |
| `*.reducer.hpp` | **Reducer** | Atomic Kleisli Arrows | `Ctx -> expected<Ctx, DomainError>` arrows via `.and_then()` / `.transform()` / `.or_else()`; no switch-case monoliths. **No globals, no side effects, no in-place persistent mutation.** |
| `*.event.hpp` | **Event** | Discrete Event Log | Immutable records of occurrences emitted by reducers. Closed vocabulary; an event-free pod still declares an explicit (possibly empty) event type. |

**Extension suffixes — added only when a litmus test demands them:**

| File Suffix | Required Contents | Strict Restrictions |
| :--- | :--- | :--- |
| `*.plan.hpp` | Batch & Scene Compilers | Pure functions: `(WorldSnapshot, Assets) -> RenderPlan`. **No GPU/driver calls.** |
| `*.edge.hpp` | Impure Execution Edges | Hardware drivers, SDL windows, audio DAC submission, and disk I/O. |
| `scripts/*.lua` | Mirrored Decision Rules | Stateless Lua mirrors of reducer decision rules; never authoritative over C++ reducers. |

**Conformance note (2026-09-15):** Pods created before this canon (e.g. `mission`, which lacks `*.action.hpp` / `*.event.hpp`) are nonconforming and must be brought up to the Core 4 by adding explicit closed vocabularies. New pods must conform from day one.

### 6.3 Inter-Pod Encapsulation Rules
1. **Public API Restriction**: A domain pod may only expose its `*.contract.hpp` and `*.event.hpp` to outside systems.
2. **Private Arrows, Public Gateways**: Domain A must never call Domain B's internal arrow bodies directly; it composes B's public typed gateway inside an orchestrator saga.
3. **KDBA Gateway (no ping-pong)**: Cross-domain sagas compose gateways synchronously; event logs are egress, not intermediate choreography:
   - `Combat` emits `CombatEvent::BOT_KILLED`.
   - `Quest` consumes `CombatEvent::BOT_KILLED` and updates its active objective counters.
   - `AudioEdge` consumes `CombatEvent::BOT_KILLED` and triggers the explosion sound on the SPSC ring buffer.

### 6.4 Core Engine Module Directives
- **Render Path Orchestration**: The dynamic render path system is the engine's first
  formal Domain Pod (`domains/renderpath/`, Core 4). Recipe/plan types are the contract;
  `RenderPathCommand` intents (`SelectPathPresetIntent`, `SetRenderingTechniqueIntent`, …)
  are the command vocabulary; the `renderpath` Kleisli pipeline compiles and hot-swaps plans as a
  pure railway (invalid compile ⇒ keep previous plan + `PATH_SWAP_REJECTED` event);
  `PATH_COMPILED` / `PATH_SWAP_REJECTED` are the only triggers for executor/GPU (re)builds.
  See `docs/arch/render_path_domain_pod_architecture.md` and
  `docs/roadmap/domain_pod_engine_rollout_roadmap.md`.
- **RHI Drivers**: Execution edges only. Drivers translate value descriptors
  (`resource_desc`, `command_desc`, `pipeline_desc`, `sync_desc`) into backend objects;
  no driver handle (`Vk*`, `GL*`) crosses upward, and the renderpath pod compiles with
  zero driver includes.
- **Scene**: Canonical transform: `SceneObjectSet::to_render_items(view, proj, &arena) -> RenderItemSpan`.
- **Lighting**: Canonical transform: `LightSet::to_cullable_gpu(...)` producing flat GPU-ready tile buffers.
- **Input / Controls**: OS events are tokenized into `UserCommand` streams and reduced via `reduce_user_commands()`.
- **Module classification**: each engine module is either a formal Domain Pod or
  "pod-shaped by analogy" (pure transforms named per the VOP pipeline above); the
  per-module mapping is maintained in `docs/roadmap/domain_pod_engine_rollout_roadmap.md` (P5).
---

### 6.5 Vocabulary Law — Semantic Surface for Renderer Authors

"Cosmetics are the API." In a value-oriented design, state lives in plain structs, so
the *names* carry all the meaning. Renderer authors must be able to compose new render
paths by selecting well-defined vocabulary, not by reading implementation files. The
following naming law is mandatory for all new public vocabulary (recipe/plan/contract
types, enums, factories):

1. **Closed menus as `enum class : uint8_t`** — every decision axis (technique, light
   volume provider, culling mode, post stack) is a closed enumeration with a companion
   `*_name()` reflection function. No `bool` pairs, no magic ints.
2. **`make_*` factories, never out-of-range literals** — construction of non-trivial
   values goes through `inline` `make_*` functions with defaulted safe fields.
3. **Pure `with_*` transforms for customization** — free functions of the form
   `Recipe with_technique(Recipe, Technique)` returning a modified copy. Recipe tuning
   reads as a sentence and never mutates a shared instance:
   `auto r = with_light_tile_size(with_technique(preset::DeferredPlusVk, Clustered), 8);`
4. **Named constants over raw knobs** — resource knobs get strong semantic aliases
   (`LightTileSize`, `ClusterZSlices`) and named `inline constexpr` values
   (`k_light_tile_small = 8u`) instead of bare `uint32_t` literals at call sites.
5. **Capability predicates, not comments** — questions the compiler/reducer asks are
   `inline` predicates with readable names: `render_path_preset_supports_taa(preset)`,
   `render_path_culling_allows_occlusion(mode)`.
6. **A named catalog of complete recipes** — every "well-defined renderer" is exposed
   once as a named value (preset header), so building a known renderer is picking one
   identifier, and a novel renderer is a preset + `with_*` deltas.
7. **The Domain Glossary** — `docs/pods/DOMAIN_GLOSSARY.md` maps each concept →
   type → header → owning pod, and is the single entry point for "which struct do I
   select when building a renderer".

Rationale: these rules make the renderer author's experience a *menu* (closed enums +
named presets + capability predicates) rather than an archaeology of demo code, which
is exactly the "select well-defined structs and definitions" requirement, and they cost
nothing at runtime (all `inline constexpr`/`inline` free functions on value types).

## 7. Dual-Tier Memory Specification

```
+-----------------------+-----------------------------+-----------------------------+
│                           MEMORY TIER ALLOCATION MATRIX                           │
+-----------------------+-----------------------------+-----------------------------+
| Attribute             | Transient Frame Arena       | Persistent State Storage    |
+-----------------------+-----------------------------+-----------------------------+
| Backing Resource      | FrameMemoryResource (Bump)  | get_default_resource()      |
| Lifetime              | Single Frame (Tick)         | Entire Session / Level      |
| Allocation Cost       | O(1) Bump Pointer           | Standard Heap Alloc         |
| Deallocation Cost     | O(1) Offset Reset           | Standard Free               |
| Contents              | Commands, Events, Plans, UI | WorldSnapshot, Stats, SoA   |
| Failure Policy        | Fallback to default heap    | Standard error handling     |
| Safety Invariant      | Never retained past frame   | Safe across frame ticks     |
+-----------------------+-----------------------------+-----------------------------+



### 7.1 Hot-Path Performance Law — ECS Speed Under Domain Pods

**Orthogonality principle.** The Domain Pod pattern governs *state transitions*
(vocabulary, reducer, events); ECS-grade performance comes from *data layout and
iteration* (SoA, contiguity, handles, chunked spans). The two axes are independent:
a pod's contract type may be — and for hot domains must be — a chunked SoA table
(see `combat.contract.hpp` → `ProjectileTableSoA`). A reducer's purity means
deterministic, side-effect-free *decisions*, not full-buffer copying on every
transition:

- **Hot pods stream, then commit (KDBA)**: arrows never mutate persistent SoA tables mid-chain; hot loops run branchless transforms over chunk spans and the boundary commits `Step` atomically. Edges apply committed change events. Snapshotting a whole SoA table is reserved for rollback/save points, not per-tick.
- **Reducers are configuration-time, not per-frame**: path/recipe reducers run on
  change only; per-frame hot loops are pure batch transforms over
  `std::span` chunks (Rule 7.1 wait-free contract) — pods never sit in the hot loop.
- **Cache-streaming techniques are mandatory where measurable** on hot loops
  operating over SoA chunks:
  1. **Software prefetch** of the next chunk's arrays (`__builtin_prefetch` /
     `std::experimental::prefetch`) while processing the current chunk;
  2. **Non-temporal (streaming) stores** for write-once outputs that will not be
     re-read soon (tile-buffer clears, final-blit writes, streaming GPU upload
     staging), avoiding cache pollution;
  3. **Chunk sizing to cache hierarchy** — job chunks sized so the chunk's active
     SoA arrays fit L2, enabling hardware-stride-friendly linear walks;
  4. **SIMD gather/scatter over SoA columns** where the compiler's auto-vectorizer
     stalls (light binning, particle integration);
  5. **Arena-backed command/event logs** — pod event logs are flat PMR vectors or
     SPSC rings (never node-based), so even the *decision* tier stays
     cache-resident.
- **Monad granularity law**: the monad rides at chunk/batch/span level —
  the inner loop streams flat arrays branchlessly and the pipeline monad
  evaluates the chunk outcome. Never wrap hot-loop scalar elements in
  `std::expected` (e.g. `std::vector<std::expected<...>>`): discriminant
  padding breaks cache alignment and disables auto-vectorization.
- **Verification**: hot-path kernels must be benchmarkable headlessly
  (chunked span jobs are GPU/OS-free), and any pod whose reducer executes per frame
  is a design smell to be flagged in review — per-frame work belongs to batch
  planners (plan-extension) or execution edges.

Rationale: this law makes explicit that adopting pods *never* trades away
mechanical sympathy (Constitution Principle: Hardware Mechanical Sympathy). ECS
architectures get their speed from layout, not from their API shape; pods keep the
layout laws (Rule 6.1, Rule 7.1) and add determinism and dynamism on top.

```


### 7.2 Contiguous Backing-Store Law — Dynamically Allocated Continuous Arrays

§7.1's cache-streaming techniques only pay off if the streamed columns are physically
contiguous at whatever size the workload demands. Pod hot-state therefore uses
**dynamically allocated, contiguous column arrays** under the following mandatory rules:

1. **Two tiers, one law of shape** — every hot column is a contiguous array in either
   tier:
   - *Persistent tier*: `std::pmr::vector<T>` (or a dedicated `SoaTable<Ts...>` owning
     one `pmr` allocation per column), **`reserve()`d up front** from a capacity
     estimate; growth is geometric and is a cold-path event that emits a compaction
     event — never a per-frame occurrence.
   - *Frame tier*: arena-backed spans carved from `FrameMemoryResource` (bump
     allocator); no reallocation exists by construction.
2. **Stability by handle, not pointer** — because columns can grow, all cross-frame
   references into pod state are generational `uint32_t` handles (Rule 6.1); raw
   pointers/references into a growable column are never stored.
3. **Density preservation** — order-independent columns use swap-and-pop removal so
   live elements stay dense and contiguous; ordering-sensitive columns use explicit
   compaction passes (streaming copies) instead of per-element `erase`.
4. **Cache-line discipline** — column bases are 64-byte aligned; columns of
   hot/cold fields are separated into distinct arrays (no AoS smuggling); a table's
   per-iteration working set (the columns a single pass touches) must be inspectable
   in review.
5. **No node-based containers in hot state** — `std::list`/`std::map`/`std::set` are
   forbidden in pod hot state and frame-tier structures; keyed lookup uses flat
   open-addressing maps over contiguous storage (pmr-backed).
6. **Library ownership** — the containers above are *shared lib utilities*
   (`shs/memory/`, `shs/containers/`), promoted from the demo implementations, so
   every pod gets identical streaming-friendly semantics; demos must not define
   private copies of them.

Rationale: "dynamically allocated" and "cache-friendly" are not in tension if
allocation is batched, reserved, handle-stabilized, and aligned. This law turns
Rule 6.1 (SoA) and §7.1 (cache streaming) into concrete, enforceable container
requirements instead of aspirations.

---

## 8. KDBA Kleisli Pipeline Doctrine (Supreme Law, 2026-09-16)

### Mandatory Standards
- `std::span<const T>`: For immutable non-owning views across reducers, AI evaluators, and tile jobs.
- `std::pmr::vector`: For all transient vectors backed by `FrameMemoryResource`.
- `std::variant` & `std::visit`: For typed, closed sets of user commands and game events.
- `std::string_view` & `constexpr` hashing: For zero-allocation ID lookups and asset tag resolution.
- `std::expected` (C++23 / `tl::expected`): For fallible planning and resource loading; planners must return explicit error types instead of crashing or throwing exceptions.

### KDBA Universal Primitive (amendment, 2026-09-16 — supersedes tier doctrine)

The atomic Kleisli arrow `A -> expected<B, DomainError>` is the sole unit of logic (2–5 lines, pure, isolated). State transitions are flat railway compositions via `.and_then()` / `.transform()` / `.or_else()` — switch-case reducer monoliths are forbidden. KDBA is adopted **universally, including the reducer core**, via the bundled house signature `(S_old, A) -> expected<Step{NextState, Events}, ClosedEnumError>`: success carries state + facts, failure keeps state and materializes a rejection fact; the Writer-on-both-rails arena out-param is retired. R3/R5b event-count evidence is preserved as HISTORY (§12) — new sagas prove fact preservation via the KDBA spike.

**Adopt everywhere a fallible transition exists (KDBA):**
- `try_swap_plan` / render-path compile → resolve chain: replace the
  `(plan, valid, errors[])`-then-classify pipeline with
  `expected<RenderPathExecutionPlan, PathSwapRejectionReason>` chained via
  `and_then`/`transform`/`or_else` (removes the stringly-typed
  `classify_plan_rejection` intermediate).
- `RenderPathResolvedState` and similar `(payload, bool valid)` pods: prefer
  `expected<Payload, ClosedEnumError>` so "did anyone check `valid`" bugs cannot
  exist.
- The demo input bridge (`map_action_to_renderpath_command` returning
  `std::optional<RenderPathCommand>`): compose with `and_then` at edge call
  sites instead of get-if/early-return noise.
- Diagnostics/logging: `std::format` formatters for the closed enums so logs
  stop hand-casting (`static_cast<unsigned>(ev.reason)`).

**Reducer core (KDBA):** `reduce_render_path` is a Kleisli composition over a command span emitting N facts per command into `Step.events`; the closed `RenderPathCommand` / `RenderPathEvent` variant vocabularies stay, but ride inside `expected<Step, PathSwapRejectionReason>` (keep-previous-plan + `PATH_SWAP_REJECTED` invariant). Single-value/single-error is expressed per-step, multi-event fidelity is preserved in `Step.events`.

**Why this matters (benefits):**
1. **Honest failure types** — `expected<T, ClosedEnumError>` makes accept/reject
   explicit in the signature; the compiler enforces that callers handle
   rejection, replacing boolean `valid` flags and assert/crash paths.
2. **Error-channel composition** — compile → classify → accept/reject pipelines
   become linear `and_then`/`or_else` chains instead of early-return ladders,
   keeping the pure-planning layers pure and assertion-free.
3. **Constitution-compatible** — closed-enum error payloads (no `std::string`
   diagnostics inside intents), vocabulary types only, no exceptions/RTTI
   required on the value path.
4. **Replay-friendly** — monadic transforms are pure functions of their inputs,
   matching the determinism requirement (§9.4).

**Prerequisites & constraints:**
- Toolchain: C++23 is baseline for `shs-renderer-lib` (Constitution I §10,
  GCC 13.3+); all trees baseline C++23 since 2026-09-16 (L2 closed).
- Error types must stay closed enums (no `std::string` in `expected` payloads
  inside pod/intent vocabulary).
- Introduce incrementally in Run 2 (GPU-free demo mode) and the Task 3 pure
  planner extraction — both are leaf-level plumbing where it pays; prototype as
  a small reversible spike (e.g. `try_swap_plan`) before committing.
- `std::expected` (C++23 / `tl::expected`): For fallible planning and resource loading; planners must return explicit error types instead of crashing or throwing exceptions.

### KDBA saga law (amendment, 2026-09-16 — supersedes compensation note)

Saga stages are Kleisli arrows over a transient `SagaContext`; the compensator is an
`.or_else()` continuation consuming the emitted receipt/fact, never ad hoc
done-flags or phantom POD flags. Persistent PODs stay pristine until atomic commit; on failure the transient evaporates. Validate-before-mutate is preferred; where mutation precedes a fallible step, every mutated stage must have emitted a fact. The wallet leak (restoring flagged stages while a prior debit leaks) is non-conforming (Rule 12). The 7 KDBA dimensions apply: PODs own state, arrows own logic, monads own flow, boundaries own writes, sagas own consistency, ECS owns layout, structured concurrency owns time — never mix abstractions.

### Forbidden in Planning and Reducer Layers
- `std::shared_ptr` / `std::make_shared` (Hidden atomic reference-counting contention).
- `dynamic_cast` / Runtime Type Information (RTTI) branching.
- Raw pointer switching with ambiguous ownership semantics.

---

## 9. Compliance Checklist & Static Verification

Before submitting new features or major refactors, verify the following:

1. **Planning/Execution Split**: Is the feature split into pure value planning/reduction and isolated side-effect execution?
2. **Zero Heap Allocation**: Does the per-frame loop run with zero standard `malloc`/`new` calls, using the PMR Frame Arena for transients?
3. **Memory Isolation**: Are persistent state snapshots strictly allocated using persistent memory, and transients on the arena?
4. **Deterministic Behavior**: Do identical state snapshots and action spans produce bit-for-bit identical outputs?
5. **Wait-Free Span Contracts**: Do multi-threaded jobs take immutable spans and write exclusively to non-overlapping target buffers?
6. **Encapsulation & Suffixes**: Does the domain follow the canonical file suffixes (`*.contract.hpp`, `*.action.hpp`, `*.reducer.hpp`, `*.plan.hpp`, `*.event.hpp`)?
7. **No Mutexes in Hot Paths**: Are audio, simulation, and rasterization completely free of mutex locks and spinlocks?
8. **No Reducer Monoliths**: Is every transition a flat `.and_then()` / `.transform()` / `.or_else()` chain of named arrows (no switch-case monoliths)?
9. **No Phantom Flags**: Do persistent PODs carry zero transitional flags (no `is_pending` / `is_locked` / `retry_count` — transient lives in `SagaContext`)?
10. **Gateway-Only Cross-Domain**: Do cross-domain sagas compose public gateways inside orchestrator pods (no direct POD writes, no event ping-pong)?

---

## 10. Automated Boundary Verification

The automated CI boundary checker (`cpp-folders/src/shs-renderer-lib/tools/check_vop_boundaries.sh`) enforces these rules on every commit:
- [x] Scan all `*.contract.hpp` and `*.reducer.hpp` files for banned includes (`#include <vulkan/...>`, `#include <SDL2/...>`, `#include <GL/...>`).
- [x] Reject any `*.reducer.hpp` containing `mutable`, `static` local variables, or `std::mutex`.
- [x] Validate that all planning passes require registered descriptor hints and return explicit execution plans by value.
- [x] KDBA Kleisli doctrine (§8): atomic `A -> expected<B, DomainError>` arrows via `.and_then()` / `.transform()` / `.or_else()`; phantom flags in `*.contract.hpp` FAIL; per-element `expected` in hot loops FAIL (§7.1 granularity); `std::string` members in `*.event.hpp` facts FAIL (closed payloads only, Rule 12); switch-case sites in `*.reducer.hpp` are INFO-tracked on the monolith-decomposition backlog (lib: renderpath ×3 closed-enum dispatches, input ×1 `action.type` — next breaking-parts phase).

---

## 11. Scalability & Architectural Benefits

1. **Multiplayer & Rollback Ready**: Pure reducers allow client-side prediction, instant snapshot rollback ($< 0.2\,\text{ms}$), and delta-compressed networking out of the box.
2. **Multi-Threaded Lua Scalability**: Lua scripts act as pure stateless functions evaluated across isolated thread-local `lua_State*` pools over chunked SoA entity spans.
3. **Zero-Glitch Real-Time Audio**: Lock-free SPSC event rings isolate audio synthesis from CPU rendering spikes.
4. **Hardware Portability**: The simulation center is 100% decoupled from graphics backends, allowing seamless swapping between the multi-threaded software rasterizer and modern GPU-driven Vulkan compute pipelines.

---

## 12. Adoption Snapshot

- Added value transform for scene object conversion (`SceneObjectSet::to_render_items`).
- Added value transform for light culling GPU payload generation (`LightSet::to_cullable_gpu`).
- Added value-style render path resolution object (`RenderPathResolvedState`).
- Added value-style pipeline execution planning object (`PipelineExecutionPlan`) and plan builder in `PluggablePipeline`.
- Migrated human/bot controller helpers to pure value-action emission helpers (`UserCommand` / `RuntimeAction`).
- Standardized command processing on pure collection and reduction (`collect_runtime_actions`, `reduce_all`).
- Hardened `IRenderPass` to explicit pass execution requests (`build_execution_request` + `execute_resolved`).
- Removed mutable validity flags from shared context (`Context::forward_plus`) and promoted depth/light readiness into request-scoped capabilities.
- Removed dynamic polymorphism (`dynamic_cast`) from depth-attachment and pass policy planning.
- Added automated boundary check script (`tools/check_vop_boundaries.sh`) and CMake target `shs_renderer_vop_boundary_check`.
- Codified Dual-Tier Memory Lifecycle Separation (`FrameMemoryResource` vs persistent state storage).
- Codified Glimmer/Ember Domain Pod standard (`domains/<domain>/`) with strict suffix naming contracts.
- Codified Lock-Free SPSC Audio Edge for glitch-free procedural sound synthesis.
- Decided C++23 monadic **targeted adoption** (2026-09-15): `std::expected` at compile/resolve error seams, monadic `std::optional` in input bridges, enum formatters for diagnostics; reducer/command variant + event-stream core explicitly out of scope (see §8).
- Adopted the C++23 monadic pipeline doctrine (2026-09-16): channel-based
  monadic law (§8 tier doctrine, Rules 11–12, Core 4+1 orchestrator, A.7
  F-DOD-DDD correspondence); bundled `expected<(State,Events)>` reducer
  signature explicitly not adopted, campaign evidence preserved.
- Adopted KDBA Kleisli Domain Boundary Architecture (2026-09-16): atomic `A -> expected<B, DomainError>` as universal primitive; bundled `expected<Step{NextState, Events}, ClosedEnum>` reducer ADOPTED (supersedes S8 tier doctrine + A.7 divergence); cross-context Kleisli gateways via orchestrator pods; transient SagaContext vs persistent PODs; phantom-flag ban; S8/S10/A.7 normative sections updated, S12 history above preserved as provenance.
- Recorded the functional-programming parallel (**Appendix A**, 2026-09-15): the Domain Pod architecture as The Elm Architecture in C++ — structural enforcement standing in for a type-system effect boundary; documents which guarantees are inherited from purity and which the backlog linters must hand-build. Lineage anchored in **"functional core, imperative shell"** (A.5) and the **actor model** (A.6: pods as deterministic actors, concurrency relocated to the edges): *a synchronous, deterministic actor system with event sourcing, running a functional core that speaks to imperative shells through effect-describing values.*

---

## Appendix A — Lineage (HISTORY; KDBA §8 + Rules govern, 2026-09-16)

> Observational note, not a new law: the VOP architecture is a rediscovery of
> pure functional programming's core discipline — specifically The Elm
> Architecture (Model / Update / Msg) — expressed in C++ (C++23 baseline for the lib, Constitution I §10) where no effect
> system exists. Recorded so authors recognize which *guarantees* are
> inherited from FP purity, and which are hand-enforced structurally.

### A.1 Correspondence table

| VOP / Domain Pod construct | Functional-programming equivalent |
| :--- | :--- |
| KDBA pipeline: `(State, Commands, dt) -> expected<Step{Next, Events}, Err>` (HISTORY: redux `f -> (NewState, Events)`) | Kleisli chain (HISTORY: `foldl update`) |
| `domains/` never including `execution/` | the `IO` boundary: pure functions cannot touch effects |
| Events on a caller-provided PMR arena | explicit effect channel (Elm's `(Model, Cmd)` return pair) |
| `std::expected` + `.transform()` / `.or_else()` (§8) | the `Either` monad; monadic chaining hand-rolled |
| Closed `Command` / `Event` `std::variant` vocabularies | sum types (algebraic data types) |
| Replay harness: same command log → same event log | referential transparency, tested |
| Time-travel debug overlay (P6) | the Elm debugger (possible *because* of the architecture) |

### A.2 Enforcement: type system vs. structural law

Haskell enforces the pure/impure boundary with the type system. C++ has no (even at the C++23 baseline)
effect system, so VOP enforces the same boundary *structurally*: directory law
(`domains/` vs `execution/`), include-direction linters, banned-token and
banned-pattern checks (`cpp-folders/src/shs-renderer-lib/tools/check_vop_boundaries.sh` §10), and the
Constitution itself. This is machine-checked in CI — a stronger artifact for
this project than compiler-only enforcement, and the only honest option in a
language without effects.

### A.3 Guarantees that fall out of purity (not bolted on)

Replayability and determinism gates, cross-backend pixel-parity proofs,
GPU-free reducer `ctest`s, time-travel debugging, seeded/replayable stochastic
sampling, rollback-ready networking (§11.1) — these are the standard benefits
of referential transparency. The architecture earns them by construction; no
per-feature test harness is required to believe them.

### A.4 What is kept that FP surrenders, and what is still hand-built

- **Kept:** PMR frame arenas, SoA hot tables (§7), zero-alloc frame budgets,
  cache-contiguous backing stores (§7.2) — layout and latency control that a
  GC'd lazy language gives up. "Pure functional core, data-oriented edges."
- **Still hand-built** (see Domain Pod roadmap backlog): variant exhaustiveness
  checking (GHC does this for free), immutability-by-default, semantic purity
  linters (no ambient entropy/time, no unordered-container iteration in
  reducers), and the pod replay test kit. The backlog is, in effect,
  compiling Haskell's compiler guarantees into this project's toolchain.

### A.5 Lineage: functional core, imperative shell

The closest named ancestor of this architecture is **"functional core,
imperative shell"** (Gary Bernhardt, ~2012): pure functions transform data in
the center; the imperative shell performs effects at the boundary. §2's
diagram is that pattern drawn. The FCIS testability claim — *test the core
without mocks, because it never touches the shell* — is realized here as the
GPU-free reducer `ctest`s (zero Vulkan/SDL links).

This architecture extends classic FCIS in three directions it does not go:

1. **Effects reified as first-class data, not shell code.** Classic FCIS keeps
   imperative effects as shell *code*; here effects are values — closed
   `Command`/`Event` variants in, `PipelineExecutionPlan` describing what the
   edges will do — so the shell is a generic interpreter of
   effect-descriptions, not bespoke glue. (FCIS toward Elm/event-sourcing.)
2. **The boundary is machine-checked.** Classic FCIS is convention and code
   review; here it is include-direction linters and banned-token gates (§10) —
   the core physically cannot call the shell.
3. **One core, many shells.** Classic FCIS usually has one shell per
   application. Here many edges (SW rasterizer, Vulkan, audio, platform) share
   one functional core — which is exactly what makes cross-shell parity
   provable: the core is the invariant, so the shells can be diffed against
   each other.

What FCIS does not supply, this spec does: the data-oriented half. Bernhardt's
formulation lives in GC'd languages where effect descriptions are cheap; the
Dual-Tier Memory Specification (§7), SoA hot tables, and PMR arenas make the
pure core also the *fast* core.

Lineage: **functional core, imperative shell** (the boundary) → **Elm
Architecture** (the command/event vocabulary, time travel) → **Domain Pod
Constitution** (effects as data, structurally enforced boundary, many shells
over one proven core, systems-level memory control).

### A.6 The actor-model parallel (Erlang / Akka)

Each Domain Pod is semantically an **actor**: private state reachable only
through its own gateway, commands in via typed Kleisli gateways, events out as egress facts,
one batch processed at a time (HISTORY: mailbox + frame-arena log). `Rule 8.1`'s "consume only raw-fact events;
never touch the grid" is tell-don't-ask / no-shared-state. The boundary
linter's ban on `std::mutex` in reducers is the actor invariant made
structural: an actor's state is only ever touched by its own message handler.
`PATH_SWAP_REJECTED` is actor supervision in miniature — a bad message does
not crash the pod; state survives and the failure is observable as an event.

**The one deliberate divergence — determinism.** Erlang/Akka actors are
concurrent, so cross-actor message order is unspecified: fault-tolerance and
scaling at the cost of reproducibility. Domain Pods are actors with a **global
deterministic scheduler**: the frame loop drains every mailbox in a fixed
order, making the whole system a pure function of (initial state, command
logs). Concurrency is not lost — it is relocated to the edges, where real
message-passing belongs: the lock-free SPSC audio ring is a literal mailbox
between processes, `WaitGroup` job stages are supervised workers.
Deterministic actors inside, Erlang-style message passing at the edges.

Lineage, complete: **functional core, imperative shell** (the boundary) →
**Elm Architecture** (the command/event vocabulary, time travel) → **actor
model** (state privacy, message-only communication, supervision) → **Domain
Pod Constitution** — composes all three and adds what none had: a
deterministic global scheduler, effects-as-data plans, cross-shell parity
proofs, and systems-level memory control. One-line identity: *a synchronous,
deterministic actor system with event sourcing, running a functional core
that speaks to imperative shells through effect-describing values.* Erlang
optimizes for uptime, Akka for distribution, Elm for UI correctness — this
architecture optimizes for provability, which is what the parity and replay
gates require.

### A.7 The F-DOD-DDD correspondence — SUPERSEDED by KDBA (2026-09-16)

| F-DOD-DDD construct | VOP equivalent |
| :--- | :--- |
| State monad (KDBA pipeline) | `(State, span<Action>, dt) -> expected<Step{Next, Events}, Err>` (§8; HISTORY: bare tuple) |
| Writer monad (facts in Step) | facts ride `Step.events` on success, rejection reasons on failure (HISTORY: caller-arena both rails) |
| Either monad (railway) | `expected<T, ClosedEnum>` everywhere including the reducer core (§8) |
| Kleisli arrow (pipeline stage) | atomic `Ctx -> expected<Ctx, Err>`; `.and_then()` chains (§8, Rule 11 gateway) |
| Saga orchestrator | orchestrator-is-a-pod, compensator consumes the fact log (Rule 12) |
| Bounded context | cohesive pipelines over shared PODs + one error/event language (Rule 11) |

**KDBA adoption (2026-09-16, supersedes the divergence):** the bundled reducer `(S_old, A) -> expected<(S_new, Events), Error>` is ADOPTED as the KDBA house signature (`expected<Step{NextState, Events}, ClosedEnum>`; failure keeps state + materializes a rejection fact). Facts ride inside `Step` on success, rejection reasons on failure; the Writer-on-both-rails arena out-param is retired. R3/R5b evidence is HISTORY — new sagas prove fact preservation via the KDBA spike.
