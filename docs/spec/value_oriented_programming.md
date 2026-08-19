
# SHS Renderer & Engine Constitution II: Value-Oriented Programming & Data-Oriented Architecture

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

---

## 3. Mandatory Rules

1. **Explicit Structs by Value**: All planning, simulation, and query APIs must accept immutable inputs and return explicit value structs by value.
2. **Side-Effect Free Center**: Simulation reducers, AI evaluators, and batch planners must be pure functions with zero hidden globals, zero singleton reads, and zero dynamic heap allocations.
3. **Pre-Resolved Edge Inputs**: Side-effect execution edges (GPU submission, Audio DAC, Disk I/O) must consume pre-resolved, complete execution plans; they must not recalculate planning decisions or query simulation state internally.
4. **Deterministic Reduction (Rule 4.1)**: State reducers (`reduce_world`, `reduce_player`, `reduce_combat`) must be strictly deterministic. Identical initial state snapshots and identical action spans must produce identical resulting snapshots across all target platforms. Non-deterministic factors (RNG seeds, system clocks, hardware inputs) must be tokenized at input edges and passed in explicitly.
5. **Dual-Tier Memory Separation (Rule 5.1)**:
   - **Transient Frame Arena (`FrameMemoryResource`)**: A linear bump allocator reset in $\mathcal{O}(1)$ at frame boundaries. Used exclusively for per-frame command streams, active render batches, temporary polygon clips, and UI draw tokens.
   - **Persistent State Storage (`std::pmr::get_default_resource()`)**: Used for world snapshots, player stats, and persistent entity tables that survive across frame boundaries.
   *Violation*: Assigning persistent state objects from the transient frame arena is strictly forbidden.
6. **Data-Oriented Memory Layout (SoA) (Rule 6.1)**: Hot-path data (physics bodies, bot tables, particles, light grids) must use Structure of Arrays (SoA) and generational index handles (`uint32_t`), avoiding pointer-chasing and Array of Structures (AoS).
7. **Wait-Free Span Contract (Rule 7.1)**: Multi-threaded jobs must be pure functions that take an immutable `std::span<const T>` and write exclusively to a non-overlapping `std::span<U>`. No mutexes, atomics, or spinlocks are allowed inside worker threads.
8. **Discrete Event Sourcing (Rule 8.1)**: Gameplay domains must never directly invoke methods or mutate state in other domains. Cross-domain interaction must occur exclusively through immutable **Discrete Event Values** (`CombatEvent`, `QuestEvent`, `InventoryEvent`) emitted by pure reducers and consumed by downstream domain reducers or execution edges.
9. **C++20 Value Abstractions**: Core APIs must leverage standard value types (`std::span`, `std::string_view` with `constexpr` hashing, `std::variant`, `std::pmr`, `std::expected`) to enforce safety and zero allocation overhead.

---

## 4. Forbidden Patterns

1. **Mixing Planning and Backend Submission**: Invoking GPU/driver calls (`vkCmd...`, `glDraw...`, `SDL_Render...`) or audio DAC writes inside a planning pass or reducer.
2. **Hidden Singleton Mutation**: Reading or writing global state (`Context::Get()`, `AudioEngine::Instance()`, static local caches) inside reducers, AI evaluators, or planners.
3. **Per-Frame Heap Allocation**: Calling standard `malloc`, `new`, `std::vector::push_back` (without a PMR arena), or dynamic memory allocators inside the per-frame update/render loop.
4. **Dynamic Polymorphism in Hot Paths**: Using virtual method dispatch (`vtable`), `dynamic_cast`, or pointer-to-base switching inside simulation entities or rasterizer loops.
5. **Side-Effect Out-Parameters**: Passing mutable references (`&out_projectiles`) to functions that secretly mutate caller state instead of returning explicit value bundles.
6. **Unbounded Frame Retainers**: Retaining pointers or references to memory allocated within the transient Frame Arena across frame boundaries.

---

## 5. Allowed Exceptions

1. **PMR Output Buffers for Hardware Fast-Paths**: Allocation-sensitive hot paths may write directly into pre-allocated `std::span<T>` or output buffers (`out` params) when memory ownership is explicit and deterministic.
2. **Execution Edge Polymorphism**: Virtual interfaces are permitted strictly at the driver boundary (e.g., `IRenderPass::execute_resolved(...)`, `ISwapchainPresenter`) where backend switching occurs outside the value center.
3. **Atomic Ring Queues at Boundaries**: Single-Producer Single-Consumer (SPSC) lock-free atomic ring buffers are allowed exclusively at execution edges (e.g., streaming discrete audio events to the audio thread).

---

## 6. Domain Pod Architecture & Module Directives

To maintain modularity, cognitive clarity, and zero-leak encapsulation across complex projects, all gameplay features and engine modules must follow the **Glimmer/Ember Pod Standard**.

### 6.1 Canonical Domain Pod Structure
Gameplay features are organized as self-contained vertical slices in `domains/<domain_name>/` using standardized file suffixes:

```text
domains/combat/
├── combat.contract.hpp   # 1. Plain data structs (ProjectileTableSoA, DamagePacket)
├── combat.action.hpp     # 2. Command intents (FireIntent, ReloadIntent)
├── combat.event.hpp      # 3. Emitted event values (EventPlayerFired, EventBotHit)
├── combat.reducer.hpp    # 4. Pure simulation rules (reduce_combat, resolve_hitscan)
├── combat.plan.hpp       # 5. Pure batch compiler (plan_projectile_mesh, plan_tracers)
└── scripts/
    └── blaster_rules.lua # 6. Mirrored stateless Lua decision rules
```

### 6.2 The Pod Suffix Laws

| File Suffix | Required Contents | Strict Restrictions |
| :--- | :--- | :--- |
| `*.contract.hpp` | Value Schemas & Snapshots | Plain data structs only. **No methods, no mutation, no logic.** |
| `*.action.hpp` | Intent Tokens / Commands | `std::variant` and enums representing caller intent. |
| `*.event.hpp` | Discrete Event Log | Immutable records of occurrences emitted by reducers. |
| `*.reducer.hpp` | Pure Simulation Reducers | Pure static functions: `(State, Actions, dt) -> (NewState, Events)`. **No globals, no side effects.** |
| `*.plan.hpp` | Batch & Scene Compilers | Pure functions: `(WorldSnapshot, Assets) -> RenderPlan`. **No GPU/driver calls.** |
| `*.edge.hpp` | Impure Execution Edges | Hardware drivers, SDL windows, audio DAC submission, and disk I/O. |

### 6.3 Inter-Pod Encapsulation Rules
1. **Public API Restriction**: A domain pod may only expose its `*.contract.hpp` and `*.event.hpp` to outside systems.
2. **Private Reducers**: Domain A must never call Domain B's internal `*.reducer.hpp` functions directly.
3. **Decoupled Event Bus**: Cross-domain communication occurs strictly by emitting and consuming event logs:
   - `Combat` emits `CombatEvent::BOT_KILLED`.
   - `Quest` consumes `CombatEvent::BOT_KILLED` and updates its active objective counters.
   - `AudioEdge` consumes `CombatEvent::BOT_KILLED` and triggers the explosion sound on the SPSC ring buffer.

### 6.4 Core Engine Module Directives
- **Scene**: Canonical transform: `SceneObjectSet::to_render_items(view, proj, &arena) -> RenderItemSpan`.
- **Lighting**: Canonical transform: `LightSet::to_cullable_gpu(...)` producing flat GPU-ready tile buffers.
- **Pipeline Orchestration**: Uses `PipelineExecutionPlan` built in a pure planning stage and executed in a disjoint effect stage.
- **Input / Controls**: OS events are tokenized into `UserCommand` streams and reduced via `reduce_user_commands()`.
---

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


```

---

## 8. C++20 / C++23 Guidance (VOP-Aligned)

### Mandatory Standards
- `std::span<const T>`: For immutable non-owning views across reducers, AI evaluators, and tile jobs.
- `std::pmr::vector`: For all transient vectors backed by `FrameMemoryResource`.
- `std::variant` & `std::visit`: For typed, closed sets of user commands and game events.
- `std::string_view` & `constexpr` hashing: For zero-allocation ID lookups and asset tag resolution.
- `std::expected` (C++23 / `tl::expected`): For fallible planning and resource loading; planners must return explicit error types instead of crashing or throwing exceptions.

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

---

## 10. Automated Boundary Verification

The automated CI boundary checker (`tools/check_vop_boundaries.sh`) enforces these rules on every commit:
- [x] Scan all `*.contract.hpp` and `*.reducer.hpp` files for banned includes (`#include <vulkan/...>`, `#include <SDL2/...>`, `#include <GL/...>`).
- [x] Reject any `*.reducer.hpp` containing `mutable`, `static` local variables, or `std::mutex`.
- [x] Validate that all planning passes require registered descriptor hints and return explicit execution plans by value.

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
