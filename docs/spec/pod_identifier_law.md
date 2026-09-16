# Pod Identifier Law — Annex to Constitution II §6.6

> Status: **normative annex (2026-09-17)**. Companion to
> [Constitution II §6.6](value_oriented_programming.md) "Pod Identifier Law".
> Precedence: §6.6 states the *rules*; this annex states the *model, the history and the
> decision procedure* behind them. Where the two ever disagree, §6.6 + §2.2 govern.
>
> Purpose: in a value-oriented design **the names are the repository's only navigation
> surface**. An engineer must be able to grep any noun this constitution uses and land on
> real code — and must never read a Domain Pod as a Redux-style reducer container.

---

## Part 0 — The one-sentence answer

**A pod is named after the four KDBA axioms it implements: a `Contract` (types), a
`Command` (what may be asked), a `Gateway` (the single pure boundary), and an `Event`
(what happened) — never after an operation on data.**

The abandoned paradigm named pods after an *operation* (`reduce_*`, `*.reducer.hpp`).
KDBA names them after an *architecture* (a boundary, a vocabulary, a fact log). That is the
whole difference, and it is why the old names were not merely cosmetic.

---

## Part 1 — The abandoned paradigm: how "reducer-based domain PODs" named things

### 1.1 What the old vocabulary encoded

The pre-KDBA pods were written in the Redux/Elm lineage: one store, one reducer per slice,
actions dispatched into it, reducers composed into a single root reducer. The names were
faithful to that theory:

| Layer | Old identifier / file | Count |
| :--- | :--- | :--- |
| File suffix (entry point) | `*.reducer.hpp` | 11 pods |
| File suffix (vocabulary) | `*.action.hpp` | 11 pods |
| Shared vocabulary header | `value_actions.hpp` | 1 |
| Entry point function | `reduce_camera`, `reduce_frame`, `reduce_geometry`, `reduce_gfx`, `reduce_input`, `reduce_lighting`, `reduce_resources`, `reduce_scene`, `reduce_sky` | 9 |
| Entry point function (aliases) | `reduce_render_path`, `reduce_fsm`, `reduce_runtime_state`, `reduce_runtime_input_latch` | 4 |
| Combinators | `reduce_via_pod`, `reduce_all` | 2 |
| Variant type | `<Pod>Action` (`CameraAction`, `FsmAction`, `GfxAction`, …) | 11 |
| Shared variant | `RuntimeAction` / `RuntimeActionType` / `RuntimeActionPayload` | 3 |
| Variant alternatives | `MoveLocalAction`, `LookAction`, `ToggleFlagAction` | 3 |
| Environment type | `<Pod>ReduceInputs`, `FsmInputs` | 12 |
| Source banner | `CORE 4. REDUCER: …` | every gateway |
| Gate / build surface | `check_vop_boundaries.sh`, `shs_renderer_vop_*`, `tests/vop_*_tests.cpp`, `[vop-boundary]` | 1 script + N targets |
### 1.2 What each old name promised vs. what was actually there

This table is the *reason* for the rename. Read it as "the name lied, and here is how":

| Old name | What a reader expects | What was actually there | KDBA axiom it obscured |
| :--- | :--- | :--- | :--- |
| `*.reducer.hpp` | a Redux store module owning slice state + a reducer | the pod's Core 4 entry point (the pure boundary) | §6.2 Core 4 is the **Gateway**. The law said "gateway" **zero** times in code. |
| `reduce_<pod>(...)` | `(state, action) => state` — an *operation on data* | the pod's single Kleisli boundary; the only legal mutation gate | Rule 10 (pod ownership). A verb-on-data reads as a transform, not as a boundary. |
| `reduce_all` | a global root-reducer composition | a *sequence* of independent pod transitions | Implies central authority over all state. KDBA has no global store. |
| `reduce_via_pod` | delegation of reduction to a slice | gateway invocation (an inter-pod boundary crossing) | Hides the boundary — the one thing worth seeing in a call graph. |
| `<Pod>Action` | an imperative command to execute (Redux actions may carry thunks/effects) | an inert intent token value | Invites effect-carrying payloads inside the pure center (§2.1). |
| `RuntimeAction` (`…Type`, `…Payload`) | a runtime-dispatchable action | the runtime **command** vocabulary — in fact the input pod's vocabulary (`InputCommand = shs::RuntimeCommand`) | Findability: the law noun `Command` had no counterpart in code. |
| `MoveLocalAction` / `LookAction` / `ToggleFlagAction` | edge-level imperative actions | intent payloads *inside* a pod variant | **Name collision** with the real edge classes `LookCommand`/`MoveCommand` (see §2.4). |
| `<Pod>ReduceInputs` | argument luggage for a reducer function | the Kleisli **environment** (`Context`), carrying `dt` | Collides with the pass layer's `struct Inputs` — a *different* concept (see §3.1). |
| `CORE 4. REDUCER` banner | a section header for the reducer | Core 4 is the Gateway (§6.1/§6.2) | Doc-vs-code drift inside the same file. |
| `check_vop_boundaries.sh`, `shs_renderer_vop_*`, `tests/vop_*_tests.cpp` | a value-oriented-programming *technique* check | the **architecture** conformance gate for KDBA | Made the gate look like an optional technique, not supreme law. |

### 1.3 Why the names were worse than "just ugly" — three measurable costs

1. **Broken grep navigation (Rule N2).** Constitution prose said *gateway*; code said
   *reducer*. Both `gateway` and `Step` appeared **zero** times in the library. An engineer
   following the law could not find the implementation, and an engineer reading the code
   could not find the law.
2. **Paradigm misreading.** "These are reducer-based domain PODs" was the honest reading of
   the tree, and it is recorded as such in the audit trigger of the KDBA conformance
   backlog. The names actively misled review.
3. **Law-drift invisibility.** With the entry point called `reduce_*`, no mechanical check
   could enforce "exactly one gateway per pod" — the checker had no stable noun to look for.
   The gate could only count `*.reducer.hpp` files, i.e. it enforced the abandoned paradigm.

### 1.4 Where the old paradigm is still preserved (deliberately)

Nothing is hidden: the pre-KDBA theory stays readable in the archives, which are **never
rewritten** (Rule N5):

- `docs/education/kdba_history/kdba_manifesto_v1.md`, `kdba_manifesto_v2.md`
- `docs/education/monadic_domain_architecture_lessons.md`
- `docs/outdated/vop-track-leaf-seams-scope-2026-09-16.md`
- `docs/backlog/domain_pod_hardening_backlog.md` (FROZEN banner)
- `docs/outdated/state_orchestration_architecture.md`

---

## Part 2 — The KDBA naming model: how we name things now

### 2.1 The two-level naming rule (the core of the model)

This is the single most useful rule in the annex, and it is what makes the vocabulary
mechanically checkable:

> **The container gets the law noun; the alternatives get verb phrases.**

Applied uniformly at all three pod vocabulary layers:

| Layer | Container (law noun) | Alternative (verb / fact phrase) | Real examples |
| :--- | :--- | :--- | :--- |
| Core 2 — Command | `<Pod>Command` | `<VerbPhrase>Intent` | `RenderPathCommand = std::variant<SelectPathPresetIntent, SetRenderingTechniqueIntent, SetRuntimeToggleIntent, ...>` |
| Core 3 — Event | `<Pod>Event` | `<FactPhrase>Event` | `InputEvent = std::variant<CameraTranslatedEvent, CameraRotatedEvent, RuntimeFlagToggledEvent, QuitRequestedEvent>` |
| Failure rail | `<Pod>RejectionReason` | `<WhyPhrase>` enum value | `PathSwapRejectionReason { CompileInvalid, EmptyPassChain, BackendUnavailable, MissingRequiredPass, DepthUnsupported, OcclusionUnsupported }` |

Note the shape: `RenderPathEvent = std::variant<PathCompiledEvent, ...>` — the *container*
carries "which layer am I", the *alternative* carries "what happened". Same for
`<Pod>Command = std::variant<...Intent>`: container = layer, alternative = request.

### 2.2 The layer table (Rule N1) with code reality

| Layer | Law noun | Canonical identifier | Lives in | Reality (2026-09-17) |
| :--- | :--- | :--- | :--- | :--- |
| Core 1 state | Snapshot / State | `<Pod>State`, `<Pod>Snapshot` | `<pod>.contract.hpp` | present in all 11 pods |
| Core 2 vocabulary | Command | `<Pod>Command = variant<...Intent>` | `<pod>.command.hpp` | 11/11 (`variant<monostate>` when empty) |
| Kleisli environment | Context (`Ctx`) | `<Pod>Context` (carries `dt`) | `<pod>.gateway.hpp` | `FrameContext`, `GfxContext`, `GeometryContext`, `SceneContext`, `ResourcesContext`, `LogicContext<TStateId>`, ... |
| Core 3 facts | Event | `<Pod>Event` + `<FactPhrase>Event` | `<pod>.event.hpp` | 11/11 + `*_event_name()` name tables + `variant_size` pins |
| Core 4 entry point | Gateway | `<pod>_gateway(State, span<const Command>, Context, events)` | `<pod>.gateway.hpp` | present and unique, still `inline void` — see 2.7 |
| Internal arrows | Arrow | verb phrases in `shs::<pod>::detail` | `<pod>.gateway.hpp` | `verify_*`, `apply_*`, `compile_plan`, `select_rule` |
| Failure rail | Error / Rejection | `<Pod>Error`, `<Pod>RejectionReason` (closed enums) | `<pod>.event.hpp` | only `PathSwapRejectionReason` exists — see 2.7 |

### 2.3 The Core 4 mapping, pod by pod (verified)

| Pod | Command variant | Event variant | Gateway entry point | Namespace note |
| :--- | :--- | :--- | :--- | :--- |
| `renderpath` | `RenderPathCommand` | `RenderPathEvent` | `renderpath_gateway` | `shs::renderpath`; first formal pod |
| `input` | `InputCommand = shs::RuntimeCommand` | `InputEvent` | `input_gateway` | alias to root vocabulary; legacy `runtime_state_gateway` + `input_latch_gateway` delegate |
| `logic` | `FsmCommand<TStateId>` | `FsmEvent` | `logic_gateway` | **parameterized** pod; `TrafficCommand = shs::FsmCommand<TrafficLight>` |
| `frame` | `FrameCommand` (`monostate`) | `FrameEvent` | `frame_gateway` | empty vocabulary, still explicit |
| `geometry` | `GeometryCommand` (`monostate`) | `GeometryEvent` | `geometry_gateway` | |
| `camera` | `CameraCommand` (`monostate`) | `CameraEvent` | `camera_gateway` | camera *math* still lives in input's edge; K1.4 decides home |
| `lighting` | `LightingCommand` (`monostate`) | `LightingEvent` | `lighting_gateway` | |
| `resources` | `ResourcesCommand` (`monostate`) | `ResourcesEvent` | `resources_gateway` | edge-owned asset stores |
| `sky` | `SkyCommand` (`monostate`) | `SkyEvent` | `sky_gateway` | |
| `scene` | `SceneCommand` (`monostate`) | `SceneEvent` | `scene_gateway` | edge-owned stores |
| `gfx` | `GfxCommand` (`monostate`) | `GfxEvent` | `gfx_gateway` | edge-owned registry |

**Empty-vocabulary law.** `using FrameCommand = std::variant<std::monostate>;` is not a
placeholder. §6.1 requires a pod to *declare* even a trivially small vocabulary so its
state-transition surface stays "greppable, auditable, and mechanically checkable". A missing
file and an empty vocabulary are semantically different; only one of them is legal.

**Name-table convention.** Every event variant ships with a reflection function
(`input_event_name(const InputEvent&)`) and a pin:

```cpp
static_assert(std::variant_size_v<InputEvent> == 4,
    "input event vocabulary changed: update name table + gateway pins");
```

Adding a fact therefore *forces* the author to update the name table and the gateway pins —
the vocabulary cannot drift silently (this is what the §6.2 drift gates consume).

### 2.4 The `Command` vs `Intent` split, and the `edge/` carve-out

`Command` is the **name of the variant type**; `Intent` is the **name of its alternatives**.
The split is not stylistic — it exists because the edge layer already owns the word
`Command`:

- `ICommand` and its subclasses (`LookCommand`, `MoveCommand`, `ToggleLightShaftsCommand`,
  `ToggleBotCommand`, `QuitCommand`) are **executable edge objects**: they carry
  hardware-derived data and know how to be applied at the boundary.
- Pod vocabulary is **inert intent**: plain values, no behaviour, arena-safe.

Evidence that this is not hypothetical: the renames were first attempted *before* this rule
existed, and produced a genuine redefinition of `shs::LookCommand` (pod alternative vs. edge
class). The rule was derived from a compiler error, which is why it is law and not taste.

**Carve-out (decision, not debt):** inside `edge/`, `Command` means *"executable edge
object"*. `ICommand` and its subclasses keep their names. An engineer reading `LookCommand`
under `edge/` is in the imperative shell; an engineer reading `LookIntent` under `domains/`
is in the pure center. The layer disambiguates, the suffix does not have to.

### 2.5 Namespaces: ownership, aliases, parameterization

- A pod's vocabulary lives in `shs::<pod>` (`shs::input`, `shs::renderpath`, `shs::logic`).
- Namespace `shs` (root) is reserved for **named orchestrators / cross-pod aggregates**:
  `RuntimeState` (owns camera state — genuinely spans pods), `RuntimeCommand`,
  `FsmCommand<TStateId>`. Root-level *pod* vocabulary is a known migration item
  (Rule N4), not a pattern to copy.
- An alias is legal when a pod's vocabulary genuinely *is* an orchestrator's vocabulary:
  `InputCommand = shs::RuntimeCommand`.
- Parameterized pods are legal and expected: `FsmCommand<TStateId>` +
  `LogicContext<TStateId>`, with `TrafficCommand = shs::FsmCommand<TrafficLight>` as a
  concrete instance. A generic pod names its parameters by role (`TStateId`), not by `T`.

### 2.6 The verb law (Rule N6) applied

| Verb shape | Axiomatic role | Examples in code |
| :--- | :--- | :--- |
| returns `expected<...>` | arrow or gateway (fallible rail) | `try_swap_plan`, `verify_*` |
| returns a plain value | transform | `to_render_items`, `to_cullable_gpu`, `compile_plan`, `select_rule` |
| touches hardware / OS / disk / wall-clock | edge | everything under `*/edge/`, `*Drivers`, pollers |
| `*_name(...)` | reflection for logging / HUD / replay | `input_event_name`, `render_path_*_name` |
| `make_*` | value factory | `make_default_soft_shadow_culling_recipe`, `make_*_intent` |
| `collect_*` / `apply_*` | batch planner / boundary commit | `collect_runtime_commands`, `apply_commands` |

**Banned on pure functions:** mutation-implying verbs — `reduce_*`, `set_*` meaning "mutate
state in place", and `apply_*` when it means "mutate in place" rather than "commit the
batched step at the boundary".

### 2.7 Honest status: law nouns not yet realized

Rule N2 requires every law noun to exist as an identifier. Two do not, yet. They are listed
here so nobody mistakes law for inventory:

- **`<Pod>Error` / `<Pod>RejectionReason`** — only `PathSwapRejectionReason` exists. All 11
  gateways still return `inline void` with a `pmr::vector<Event>&` out-parameter (the
  "writer shape"). The closed-enum failure rail is *law* (§6.6 N1) but unrealized; the
  K1.x port work in the [KDBA conformance backlog](../backlog/kdba_conformance_backlog.md)
  realizes it pod by pod.
- **`Step`** — the house-signature return type exists **only in demo pods**
  (`SessionStep`, `MissionStep`, ...), never in the library. Naming it is K1.1; defining it
  is K1.2/K1.4 — behavioural work, not cosmetic.

### 2.8 Header banner convention

Every pod source file opens with its KDBA role stated, so a reader who lands mid-file still
knows what they are holding (Rule N5: stable paths, loud headers):

```text
    FILE: input.event.hpp
    MODULE: domains/input
    PURPOSE: CORE 3. EVENT — raw facts emitted by input_gateway (R3).
```

The `CORE n. <LAW NOUN>` prefix is mandatory and must match §6.2. A banner that says
`CORE 4. REDUCER` in a file named `*.gateway.hpp` is a contradiction the review must catch.

---

## Part 3 — The migration ledger (2026-09-17)

Cosmetic pass, one atomic change set. Behaviour was deliberately untouched: every rename was
reverted-and-redone until the build and the full test suite proved it neutral.

| Was | Is | Files affected |
| :--- | :--- | :--- |
| `*.reducer.hpp` | `*.gateway.hpp` | 11 |
| `*.action.hpp` | `*.command.hpp` | 11 |
| `value_actions.hpp` | `value_commands.hpp` | 1 |
| `reduce_<pod>(...)` (9 pods) | `<pod>_gateway(...)` | 11 gateway headers |
| `reduce_render_path`, `reduce_fsm` | `renderpath_gateway`, `logic_gateway` | — |
| `reduce_runtime_state`, `reduce_runtime_input_latch` | `runtime_state_gateway`, `input_latch_gateway` | — |
| `reduce_via_pod`, `reduce_all` | `run_gateway`, `apply_commands` | — |
| `<Pod>Action` (11 variants) | `<Pod>Command` | 11 headers |
| `RuntimeAction` / `RuntimeActionType` / `RuntimeActionPayload` | `RuntimeCommand` / `RuntimeCommandKind` / `RuntimeCommandPayload` | lib + 11 demos |
| `MoveLocalAction` / `LookAction` / `ToggleFlagAction` | `MoveLocalIntent` / `LookIntent` / `ToggleFlagIntent` | pod vocabulary |
| `<Pod>ReduceInputs`, `FsmInputs` | `<Pod>Context`, `LogicContext` | 11 gateway headers |
| `collect_runtime_actions` | `collect_runtime_commands` | orchestrator |
| `CORE 4. REDUCER` banners | `CORE 4. GATEWAY` banners | all pod sources + docs |
| `check_vop_boundaries.sh` | `check_kdba_boundaries.sh` | script + CMake + docs |
| `shs_renderer_vop_*` targets | `shs_renderer_*` | CMake |
| `tests/vop_*_tests.cpp` | `tests/*_tests.cpp` | tests |
| `[vop-boundary]` / `[vop-tests]` | `[kdba-boundary]` / `[kdba-tests]` | script output |

Result: **41 files recorded by git as renames**, 100 files touched in the working tree
(lib + tests + gates + 35 docs). Verification: `cmake --build build_vcpkg` exit 0 with zero
compiler errors, `ctest` **16/16 passed**, boundary gate exit 0.

### 3.1 Scope discipline — the rename that had to be *un*-made

The first sweep was repo-wide and corrupted the execution layer: in
`include/shs/execution/passes/`, `struct Inputs` became `struct Context`, producing nonsense
like `execute(Context& ctx, const Context& in)`. The rename was fully reverted and redone
scoped to the pod layer only.

**The lesson is a naming law, not an anecdote:** `Inputs` in the pass layer and
`<Pod>ReduceInputs` in the pod layer were *different concepts that happened to share a
substring*. A mechanical rename that ignores the layer boundary creates lies. This is why
§6.6 N1 fixes one word *per layer* rather than one word per repository.

---

## Part 4 — Enforcement: how the naming law is kept true

A naming law that is not machine-checked decays into a style guide. §6.6 is therefore backed
by `tools/check_kdba_boundaries.sh` (CMake target `shs_renderer_boundary_check`, run in
ctest). The gate fails loudly rather than warning:

| Gate | Enforces |
| :--- | :--- |
| **Non-vacuity** *(new)* | every pod carries `<pod>.{contract,command,event,gateway}.hpp`, and each per-role glob matches a **non-zero** number of files |
| **Paradigm-token ban** *(new)* | no `reduce_*`, `reducer`, `*Action`, `*.reducer.hpp`, `*.action.hpp` under `domains/` or `tests/` |
| **Gateway presence** *(new)* | each `<pod>.gateway.hpp` exposes a `<pod>_gateway` entry point |
| Monolith tracker | no `switch(<command>.type)` dispatch monoliths in pods |
| Banned-token scan | no `shared_ptr` / `dynamic_cast` / `mutex` / `std::function` in pod cores; no `SDL_*` / `fopen` in `domains/` |
| Catalog drift | event/enum catalogs match `EVENT_FLOW.md`; cross-domain includes limited to sanctioned seams |

Run locally:

```bash
cd cpp-folders/src/shs-renderer-lib && bash tools/check_kdba_boundaries.sh
# or, from cpp-folders:
ctest --test-dir build_vcpkg -R shs_renderer_boundary_check -V
```

### 4.1 The vacuity hazard (Rule N3) — observed, not theoretical

The gate locates pods **by filename glob** (`find domains -name '*.gateway.hpp'`). A rename
that lands without updating those globs makes the glob resolve an **empty file set** — and
GNU `grep -r` with no file operand then silently re-scopes to the *current working
directory*, enforcing the rule against the **wrong tree**.

This was observed during this migration, in this order:

1. `*.reducer.hpp` -> `*.gateway.hpp` landed in code, globs not yet updated.
2. The boundary test reported `switch` monoliths in `execution/pipeline/`, `execution/passes/`
   and `rhi/drivers/` — files with no pod vocabulary at all.
3. The suite went **15/16 failing**: a checker that had been enforcing the law was now
   enforcing nonsense, while *looking* green-ish and authoritative.

The failure mode is the dangerous kind: **a vacuously-passing gate and a vacuously-failing
gate look the same in CI output as a real one.** Hence the non-vacuity gate exists now: an
empty glob is a hard FAIL, not a pass.

### 4.2 The safe-rename recipe (Rule N3 procedure)

One commit, in this order:

1. `git mv` the files (pod layer only — never repo-wide; see §3.1).
2. Sweep symbols, **scoped to `include/shs/domains/` + `tests/`**.
3. Update every gate glob, target name and log prefix in the *same* commit.
4. Grep the *old* tokens and confirm zero hits in live code/docs.
5. Run the gate and the full ctest suite; confirm non-vacuity output names all 11 pods.
6. Only then update prose (`docs/`), never before the code has proven neutral.

---

## Part 5 — Carve-outs and open items

### 5.1 Deliberate carve-outs (decisions, not debt)

- **`VOP`** survives as the *technique* layer name (memory tiering, SoA, handles,
  contiguous backing stores). It is not a substitute for KDBA as the *architecture* name:
  that is exactly why `check_vop_boundaries.sh` became `check_kdba_boundaries.sh`.
- **`*.contract.hpp` / `*.event.hpp`** keep their suffixes — they match §6.2's Core 1 / Core 3
  nouns and are pinned by three drift gates.
- **`edge/` `ICommand`** keeps `Command` (see §2.4): there it means "executable edge object".
- **Archives are never rewritten:** `docs/outdated/` and `docs/education/kdba_history/`.
  `docs/backlog/domain_pod_hardening_backlog.md` is FROZEN and likewise untouched.
- **`docs/arch/action_based_input_architecture.md` and `docs/roadmap/action_based_input_roadmap.md`**
  keep their filenames: their "Action Registry" / "Action Priority" refer to the
  hardware-to-*action* binding *design concept* (a separate concern from pod vocabulary),
  not to a pod identifier.
- **Generic math notation**: tables that describe the house signature for a general audience
  (e.g. the F-DOD-DDD correspondence) may write `span<Command>`/`Context` in the abstract;
  the concrete law form is §6.1 + §6.6 N1.

### 5.2 Open items (tracked, not forgotten)

| Item | Where tracked |
| :--- | :--- |
| Root-level pod vocabulary (`shs::Fsm*`, input's root vocabulary) -> `shs::<pod>` | §6.6 N4; backlog migration note |
| `<Pod>Error` / `<Pod>RejectionReason` closed enums (gateways still `void`) | backlog K1.2-K1.5, K2.x |
| `Step` return type in the library (exists only in demo pods) | backlog K1.1 (name) / K1.2, K1.4 (define) |
| ~11 `exps-gpu-renderer` demos including the long-dead `shs/input/...` path | pre-existing breakage, needs its own repair item; symbols were swept for grep-consistency, but they are **not** build-verified |

---

## Part 6 — Decision procedure: how to name a NEW thing

Ask these in order. The first "yes" names the artifact.

1. **Is it state?** -> `<Pod>State` / `<Pod>Snapshot` in `<pod>.contract.hpp`.
2. **Is it something a caller may ask for?** -> a `<VerbPhrase>Intent` added to the
   `<Pod>Command` variant in `<pod>.command.hpp`.
3. **Is it a fact that happened?** -> a `<FactPhrase>Event` struct added to the `<Pod>Event`
   variant in `<pod>.event.hpp`, **plus** a name-table entry, **plus** the `variant_size` pin.
4. **Is it a reason a request failed?** -> a value in the closed `<Pod>RejectionReason` enum.
   Never a string, never a bool.
5. **Is it an internal step that can fail?** -> a `detail::` arrow returning
   `expected<Ctx, Err>`, named as a verb phrase (`verify_*`, `select_*`, `compile_*`).
6. **Does it touch hardware, OS, disk or wall-clock?** -> it is **edge**: it lives under
   `edge/`, may be a class, and only there may `Command` mean "executable object".
7. **Is it shared across pods?** -> it is an orchestrator noun at namespace `shs` (like
   `RuntimeState`), and the pod that owns its vocabulary re-exports it (`InputCommand`).
8. **Does its name contain a mutation verb on data?** -> reject it and go to step 2 or 5
   (Rule N6).
9. **Added/renamed a file?** -> update the gate globs, target names and the prose **in the
   same commit** (Rule N3, §4.2).

### Worked examples

**A new pod `combat`.** `domains/combat/{combat.contract,combat.command,combat.event,combat.gateway}.hpp`;
`CombatState`; `CombatCommand = std::variant<FireIntent, ReloadIntent>`;
`CombatEvent = std::variant<ProjectileFiredEvent, BotHitEvent>`; `combat_gateway(CombatState&,
std::span<const CombatCommand>, const CombatContext&, std::pmr::vector<CombatEvent>&)`;
namespace `shs::combat`; banner `CORE n. <LAW NOUN>`. Empty vocabulary ->
`std::variant<std::monostate>`, never an omitted file.

**A new request.** "Players may reload." -> `struct ReloadIntent { };` added to
`CombatCommand`. Not `ReloadAction`, not `Reload`, not `OnReload`.

**A new fact.** "A projectile was spawned." -> `struct ProjectileFiredEvent { ... };` added to
`CombatEvent`, a `combat_event_name()` entry, and the count pin bumped. The compiler
forces the pin; the reviewer forces the name table.

**A new failure.** "Reloading while empty." -> `CombatRejectionReason::NoAmmoInReserve`,
surfaced as a rejection *fact*, with state kept (the renderpath `PathSwapRejectedEvent` model).

**A new internal step.** "Check that reload is possible." -> `detail::verify_reload(...)`
returning `expected<CombatContext, CombatRejectionReason>`.

**A new edge capability.** "Open the save file." -> a class under `edge/`; it may carry
`Command` in its name; it may not enter `domains/`.

### 6.1 Smells this law exists to catch

- An identifier that names an **operation on data** instead of an architectural role.
- A pod with **two** public gateways (`camera` briefly had a stub beside input's path).
- A pod whose vocabulary types are named `*Action`.
- A `set_*` that mutates state instead of producing an intent value.
- A bare `float dt` parameter instead of `<Pod>Context`.
- An error expressed as a `std::string`, a bool, or a phantom validity flag.
- A gate whose glob matches nothing — **the vacuous pass** (§4.1).

---

## Cross-references

- [Constitution II §6.6 — Pod Identifier Law](value_oriented_programming.md) (normative)
- [Constitution II §6.1/§6.2 — Pod structure and suffix laws](value_oriented_programming.md)
- [DOMAIN_GLOSSARY](../pods/DOMAIN_GLOSSARY.md) — concept -> type -> header catalog
- [EVENT_FLOW](../pods/EVENT_FLOW.md) / [ERROR_FLOW](../pods/ERROR_FLOW.md) — catalogs
- [KDBA conformance backlog](../backlog/kdba_conformance_backlog.md) — K1.x-K6.x work
- `tools/check_kdba_boundaries.sh` — the executable form of Parts 4
