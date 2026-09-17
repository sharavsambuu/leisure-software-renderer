# Building a Domain Bounded Context — the full tutorial (teaching, 2026-09-17)

> **Explanatory, not normative.** Constitutions legislate (`value_oriented_programming.md`
> = Constitution II; Const. II §2.2 wins all ties). This file is the
> new-contributor walkthrough: how a feature becomes a bounded context folder,
> how DVOs and contracts are defined, why the style exists, and what to watch
> out for while coding. Prerequisites in reading order: §8 of this file.
> Companion teaching: `kdba_kleisli_composition.md` (the monad primer),
> `domain_value_objects.md` (the DVO rename),
> `cpp26_contract_guardrails.md` (the contract bridge).

---

## 1. The mental model in ten sentences

1. **The universe is data and arrows.** Data lives in plain structs; behavior
   lives in small pure functions between them. Nothing else exists.
2. A **Domain Value Object (DVO)** is a plain, identity-free data value that
   belongs to exactly one bounded context and is **always valid at every
   module edge**. Law: `docs/spec/domain_value_object_law.md`.
3. A **bounded context** ("pod") is a folder with one ubiquitous language:
   its own commands, events, state, and one gateway that owns all decisions.
4. **Arrows are the only unit of logic**: 2–5 line functions of the shape
   `A -> std::expected<B, DomainError>` — pure, `constexpr`-friendly, no
   side effects.
5. The house arrow signature is
   `(State&, span<const Command>, ...) -> expected<Step{Next, Events}, Err>`:
   commands go **in**, facts/events come **out**, state is returned, never
   mutated in place mid-arrow.
6. **One writer per context**: the gateway is the only code that transforms
   state. Callers dispatch commands and consume events — they never write
   state directly.
7. **Validate before mutate.** A rejected command leaves state untouched and
   materializes a rejection fact. There is no "half-applied" success.
8. **No exceptions.** Errors are closed `enum class : uint8_t` values on the
   error rail — 16 bytes, register-passable, exhaustive by construction.
9. **DVOs are always valid; identity lives outside.** A DVO has no
   `is_pending`/`needs_rebuild` phantom flags — invalid is expressed as
   *structure* (or not existing), never as a flag apologizing for it.
10. **The compiler is the reviewer.** Closed enums, exhaustive dispatch,
    static_assert pins, contract rims — the architecture is designed so that
    mistakes are build errors, not 3 a.m. bugs.

## 2. Anatomy of a bounded context folder

Every feature pod follows the same four-file spine (see `logic/`,
`renderpath/`, `render/frame/`):

```
<pod>/
├── <pod>.contract.hpp   # Core 1: DVOs — State, Snapshot, Desc, closed enums
├── <pod>.command.hpp    # the closed command vocabulary (std::variant)
├── <pod>.event.hpp      # the facts this pod emits (closed, too)
├── <pod>.gateway.hpp    # the arrows: validate → dispatch → commit → report
└── <subfolder>/         # optional: pure leaves + execution, kept off the rim
```

Real example — `renderpath/`:

| File | Role | Tier (P1) |
| :--- | :--- | :--- |
| `renderpath.contract.hpp` | DVOs: state/snapshot, invariants documented | value |
| `renderpath.command.hpp` | the commands callers may send (closed variant) | edge vocabulary |
| `renderpath.event.hpp` | the facts the pod can emit | edge vocabulary |
| `renderpath.gateway.hpp` | arrows, validate-before-mutate, rim contracts | edge |
| `planning/` | pure leaves: `pass_id`, presets, capabilities, the compiler | pure leaf |
| `execution/` | registries, executor, renderer seams | edge (side-effectful) |

**The P1 placement law in one line**: edge contracts at seams, value
contracts in pure leaves. Invariants that are *structural* (a table maps
mode → technique) belong in the pure leaf; *translation* work (compiler
errors → pod rejection reasons) belongs in the gateway. The placement gate
(`check_kdba_boundaries.sh`) enforces this mechanically — if you put an
invariant in the wrong tier, the gate tells you before review does.

Subfolders have a convention too: **planning/pure** subfolders hold pure
leaves (no includes from the edge), **execution** subfolders hold the
side-effectful rims. The include-graph gate polices the directions.

## 3. Worked walkthrough — a new feature pod, start to finish

Scenario: we add a camera-shake effect. Here is the whole path, in order,
with the reasoning a senior contributor would apply out loud.

### 3.1 Name the context and its language first

Before any code: what is this context called, what can *happen* in it, what
can it *say about it*? One ubiquitous language per folder (Rule 11):
`camera_shake` — commands `AddShake`, `Tick`, `ClearShake`; facts
`ShakeStarted`, `ShakeSettled`. If you cannot write the command/event
vocabulary down in five minutes, the feature is not understood yet — that is
a signal, not a blocker.

### 3.2 Define the DVOs (`camera_shake.contract.hpp`)

```cpp
// A DVO: plain, identity-free, ALWAYS valid. Produced only by the gateway's
// validate-before-mutate step. No phantom flags, no "half-ready" states.
struct CameraShakeState
{
    float amplitude{};       // 0 = settled (a value, not a flag)
    uint32_t remaining_ms{}; // decay is data, not a `needs_update` bit
    uint32_t seed{};
};
static_assert(std::is_trivially_copyable_v<CameraShakeState>);
```

Rules in force here:

- **No phantom flags** (`is_pending`, `needs_rebuild`, `was_rejected`) — if
  you are writing one, you are encoding an invariant into a bool that the
  *structure* should express (amplitude == 0 *is* the settled state).
- **No methods that make decisions** — DVOs are passive; logic goes in the
  gateway. A helper that is *pure and total* (e.g. a table lookup) may live
  beside the DVO as a `constexpr` free function.
- Document the invariant in a comment at the definition — this is the law
  the gateway's `SHS_PRE`/`SHS_POST` will enforce.

### 3.3 Define the command vocabulary (`camera_shake.command.hpp`)

```cpp
struct AddShake { float amplitude; uint32_t duration_ms; };
struct ShakeTick { uint32_t dt_ms; };
using CameraShakeCommand = std::variant<AddShake, ShakeTick>;
```

The variant is **closed**: adding an alternative later must break every
gateway dispatch at compile time (that is the P5 exhaustiveness pin, §5
below). If a command needs to be conditional, that is usually a sign it is
two commands.

### 3.4 Write the gateway arrow (`camera_shake.gateway.hpp`)

The shape every arrow follows (abridged from the real `logic.gateway.hpp`):

```cpp
inline ShakeStep camera_shake_gateway(
    CameraShakeState& state,
    std::span<const CameraShakeCommand> commands,
    std::pmr::vector<CameraShakeEvent>& events)
{
    SHS_PRE(amplitude_is_finite(state.amplitude));   // rim precondition
    ShakeStep step{};
    for (const CameraShakeCommand& command : commands)
    {
        std::visit([&](const auto& cmd) {
            // validate-before-mutate → apply → emit facts
            // (2–5 line pure helpers; nothing grows here)
        }, command);
    }
    SHS_POST(state.amplitude >= 0.f);                // rim postconditions
    SHS_POST(applied + rejected == commands.size()); // zero-signal-loss
    return step;
}
```

Read the real one (`logic.gateway.hpp` `logic_gateway`) — it shows all four
moves: **rim precondition** (trusted data validated at entry), **closed
variant dispatch**, **P5 static_assert tail**, **rim postconditions** after
the batch commits (state integrity + zero-signal-loss accounting: every
command consumed is counted by exactly one of applied/facts/rejected).

The `SHS_PRE`/`SHS_POST` macros are the contract bridge
(`core/contract_guardrails.hpp`): debug builds run the check and route
violations through a pluggable handler; release builds fold to C++23
`[[assume]]` (or a no-op below GCC 14). They document *and* enforce the rim.
Placement law applies: gateway rims only — never inside a pure leaf's data.

### 3.5 Closed errors, 1:1 rejection mapping

When a command can fail, the rejection is **materialized**: state stays
untouched, a rejection fact is emitted, the count increments. Native errors
are translated 1:1 into the pod's own closed enum at the boundary by a named
single-purpose arrow (see `map_rejection` in `renderpath.gateway.hpp` —
exhaustive switch, total fallback, no `default:` swallowing). Error strings
belong to logs, never to the error rail.

### 3.6 Register, test, gate

1. **Test twins**: negative-test the rims (a hand-broken input must fire
   kind=assertion / kind=rejection); for stateful pods, replay the same
   recorded command span through both build twins and diff the output.
   Canonical patterns: `tests/contract_guardrails_tests.cpp`,
   `contract_guardrails_replay_probe.cpp`.
2. **Zero-signal-loss** (K3.2): every consumed command is counted by exactly
   one of applied / facts / rejected. This single postcondition has caught
   more silent-drop bugs than any review.
3. **Gates, same commit**: `check_kdba_boundaries.sh` (placement, rails),
   `check_include_graph.py` (via `python3` — bare exec gives exit 126),
   header inventory regenerated if you added a header (Rule 15: same commit).
4. **Naming law** (§6.6): gateways are `*_gateway`, never `reduce_*`;
   files are `<pod>.contract.hpp` / `.command.hpp` / `.event.hpp` /
   `.gateway.hpp`. The name is the repository's navigation surface — a lying
   name is a defect, not a style nit.

## 4. Why this beats switch-case nesting hell

The honest history (`domain_value_objects.md` §1): the earlier reducer era
wrote one pure function owning the **Cartesian product of state shapes ×
action types × failure modes**. Every new action touched every state; every
failure mode nested one `if` deeper; an unvalidated early-exit corrupted a
different branch's assumptions. That is the "switch-case nesting hell" this
architecture exists to make impossible.

Three mechanical reasons the house shape cannot regress into it:

1. **`std::expected` flattens the error dimension.** `.and_then()` on the
   success rail, `.or_else()` as the compensator — failure handling is a
   flat chain, not a nesting level. A rejected command *keeps state and
   emits a fact*; the caller never writes an `if (ok)` ladder.
2. **Closed variants replace open dispatch.** The command vocabulary is a
   `std::variant`; the dispatch is one `std::visit` with a
   `static_assert(sizeof(T) == 0, ...)` tail (P5 exhaustiveness). Adding an
   alternative later **fails compilation** in every gateway — there is no
   silently-swallowed `default:` branch to grow.
3. **Always-valid DVOs delete the validity dimension.** The monolith's deep
   nesting was mostly "check which half-valid shape the state is in". With
   validate-before-mutate and zero phantom flags, that entire nesting
   evaporates — state has one shape, always.

And the honest limits (KDBA primer §4): monads are for **batch level, never
per-element** — `vector<expected<T>>` per entity in a 60Hz chunk is a FAIL.
KDBA is for transactions/sagas/ingress; hot per-pixel/per-pass math stays a
boring flat loop. "Boring, 5-second-readable functions" is the target.

## 5. The working checklist (what to watch out for)

The five bans (each is load-bearing):

- **No phantom flags** — structure expresses state, never a `needs_rebuild`
  bool.
- **No exceptions** — `expected` + closed error enums only.
- **No cross-domain writes** — a pod owns its state; composition happens in
  orchestrator pods.
- **No in-arrow side effects** — return the `Step`; the boundary commits.
- **No monoliths** — arrows stay 2–5 lines; helpers stay pure and total.

Continuing the checklist:

- **Data hygiene**: `DomainError` is `enum class : uint8_t` (never a string
  in the error rail); house `FlatMap`/pmr for registries; `std::span` for
  command input; `std::pmr::vector` for event output; identity
  (generational handles) lives in context-owned registries, never inside a
  DVO.
- **Contract hygiene**: single-expression, side-effect-free conditions only
  (bridge rule 2); a rim comment saying *why* the data is trusted; P1
  placement — structural invariants in pure leaves, translation in
  gateways (the placement gate will tell you if you got it wrong).
- **Exhaustiveness**: no `default:` in command dispatch; the
  `static_assert(sizeof(T) == 0, ...)` tail is the pin that turns "someone
  added an enum value" into a compile error in every gateway, not a
  runtime mystery.
- **Validate before mutate**: a rejected command keeps state untouched and
  materializes a rejection fact — "half-applied success" is the bug class
  this codebase does not have.

## 6. The 30-second flow

```
caller ──commands──▶ <pod>.gateway.hpp ───▶ Step{Next, Events}
                        │  SHS_PRE: trusted-input rim
                        │  validate-before-mutate per command
                        │  closed std::visit + static_assert tail
                        │  SHS_POST: state integrity + zero-signal-loss
single writer ◀──Step──┘
events ──▶ other pods' gateways (composed in orchestrators only)
```

## 7. What good looks like (recognize it fast)

- A gateway reads in one screen: precondition → dispatch → postcondition →
  return. If it doesn't fit, it is two arrows.
- Every failure path is a value on the error rail, never control-flow
  by exception, never a swallowed branch.
- A new enumerator breaks the build, not the weekend.
- The invariant you care about is written exactly once — at the rim that
  enforces it — and the negative test proves the enforcement fires.

## 8. Reading order for a new contributor

1. This file top-to-bottom (it references everything else).
2. `kdba_kleisli_composition.md` — the arrow primitive and its laws.
3. `domain_value_objects.md` + `docs/spec/domain_value_object_law.md` —
   what a DVO is, why "POD" retired.
4. A real pod spine, smallest first: `logic/logic.gateway.hpp` (canonical
   four moves), then `renderpath/renderpath.gateway.hpp` (the pilot, with
   `map_rejection` and the transition assert).
5. `cpp26_contract_guardrails.md` — the bridge, the placement law, and the
   C++26 switch path (the bridge header itself is 65 lines — read it once).
6. The constitutions when you need the law verbatim:
   `docs/spec/value_oriented_programming.md` (Constitution II),
   `domain_value_object_law.md` (terminology).



