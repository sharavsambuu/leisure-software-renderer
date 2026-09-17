# C++26 Contract Guardrails — what they are and why they fit DVOs (teaching, 2026-09-17)

> **Explanatory, not normative.** Adoption status: **law** since 2026-09-17 —
> Constitution II Rule 17 + Forbidden Pattern 7, Constitution I §11; adopted
> from
> [docs/backlog/contract_guardrails_adoption_proposal.md](../backlog/contract_guardrails_adoption_proposal.md)
> (see also its todo list). This file teaches the concept, the project fit,
> and the C++23 emulation bridge. Companion: `domain_value_objects.md`.

## 1. Where contracts sit in the discipline arc

The project's self-discovery arc has three layers, and each solves a *different*
problem. Contracts are the third layer — they do not replace the first two:

| Layer | Mechanism | Question it answers | Failure class it owns |
| :--- | :--- | :--- | :--- |
| 1. Transitions | pure reducers → Kleisli gateways (KDBA §8) | what does this state change do? | — |
| 2. Failure composition | `std::expected` + `and_then`/`or_else` (§8) | how do failures compose and compensate? | **domain failure** — outcomes the world can legitimately produce |
| 3. Invariants | C++26 contracts (P2900) | what must be true at every seam? | **programmer error** — states that mean the code is already wrong |

The category line is the most important sentence in this file:

> The railway (`expected`/`or_else`) handles *expected* failure — typed,
> replay-safe, compensable (Rule 12). Contracts handle *unreachable* states —
> if one fires, the system is already wrong and no compensator can help.
> Never route a domain rejection through a contract, and never route a
> programmer error through `expected`.

This division is what keeps Rule 4.1 (determinism) intact: a contract violation
is not a replay-relevant transition, because in a correct system it never
happens on any platform.

## 2. What the feature is (P2900, "contract guardrails")

C++26 adds three checked assertions, evaluated at defined points:

```cpp
int scale(int v, int factor)
  pre (factor != 0)                       // precondition: checked at call site
  post (r : v * factor / factor == v)     // postcondition: `r` names the return value
{
    contract_assert (v <= max_scalable);  // internal assertion: checked inline
    return v * factor;
}
```

- **`pre(...)`** — obligation on the *caller*, checked before the body runs.
- **`post(r: ...)`** — obligation on the *callee*, checked on every return path,
  with the return value named.
- **`contract_assert(...)`** — internal invariant, checked where written.

Each check has an **evaluation semantic** chosen per build, which is the
feature's real power:

| Semantic | Behavior | SHS use |
| :--- | :--- | :--- |
| `ignore` | nothing (may still be ill-formed to reference) | none |
| `observe` | evaluate, report violation to handler, **continue** | debug/profile builds |
| `quick_enforce` | evaluate; violation ⇒ terminate | hardened/test builds |
| `assume` | **no check**; hands the condition to the optimizer | release builds |

Violations invoke a *contract-violation handler* (`<contracts>`), a single
well-defined seam — not an exception, not `assert()`. Violations are not
catchable in the usual sense and must not be used for control flow.

## 3. Toolchain reality (2026-09)

| Compiler | Contracts (P2900R14) | `<contracts>` library |
| :--- | :--- | :--- |
| GCC | **16** (first support) | 16 |
| Clang / Apple Clang | not yet | — |
| MSVC | not yet | — |
| **SHS baseline** | **GCC 13.3, C++23** — three major versions short | — |

Consequence: the *syntax* is not adoptable today; the *discipline* is
(see §6 and the proposal). The feature-test macro is `__cpp_contracts` — all
bridge code must key on it, never on compiler sniffing.


---

## 4. Why contracts fit Domain Value Objects exactly

Recall what a DVO promises (see `domain_value_objects.md`): plain aggregate,
identity-free, **always valid at every module edge**, no phantom flags. Recall
what currently *enforces* that: convention, review discipline, and the
validate-before-mutate shape of gateways. Contracts turn that convention into
executable law — and the fit is exact for a reason that surprises people:

- **Contracts attach to functions, not to types.** `pre`/`post` express
  *transition* invariants — "given verified input, the committed snapshot
  satisfies I". That is precisely where the SHS philosophy puts law: DVOs stay
  plain aggregates; the **gateway** owns the invariants. Contracts do not turn
  DVOs into "invariant types" (validating constructors, non-aggregate design —
  a direction that would break trivial copyability, persistence, and replay).
  They formalize the edge validation §2.1 already mandates.
- **The no-phantom-flags rule becomes executable.** Today "persistent DVOs
  carry zero phantom flags" is review discipline. As a postcondition on every
  committing gateway arrow — `post(s : !has_pending_flags(s))` — it becomes a
  checked, documented, machine-verified fact.
- **The wait-free span contract (Rule 7.1) is a textbook `pre`:**
  `pre (!overlaps(input, out))`, `pre (input.size() == out.size())` — written
  once at the job entry, checked in every debug build, folded to an optimizer
  assumption in release.
- **Invariants are context law, not type law** (DVO annex Part 4). The same
  data shape in two bounded contexts carries different invariants — contracts
  living on *that context's* gateway express exactly that.

Concrete pilot candidates in this tree:

| Seam | Contract | Replaces |
| :--- | :--- | :--- |
| `renderpath_gateway` commit | `post`: compiled plan references only registered techniques; pass chain non-empty | scattered defensive checks + review memory |
| `apply_*` arrows | `post`: committed DVO has no pending flags, nonzero stable IDs | comments-as-invariants |
| parallel job entry (Rule 7.1) | `pre`: input/output spans non-overlapping, sizes equal | trust + debugging sessions |
| `RenderPathCompiler` | `contract_assert`: mode-transition table hits only legal transitions | silent-fallthrough risk |

## 5. What the feature buys an architecture like this (speculation, marked as such)

1. **Executable documentation.** A `post` on a gateway is simultaneously the
   doc, the test, and the debug check. Constitutions currently quote code;
   contracts make code quote the constitution.
2. **Optimizer leverage.** Under assume-like semantics, release builds get real
   information for free — bounds, non-aliasing, non-empty invariants. For a
   software rasterizer with hot inner loops, that is the rare check that
   *costs nothing* in the build that ships.
3. **The end of the boolean-apology pattern.** `is_valid` flags exist because
   types cannot state their own law and functions cannot state their callers'
   obligations. Contracts remove the need for the flag by moving the law to
   the seam.
4. **Trustable boundaries without runtime tax.** Bounded contexts today rely on
   gates (linters, CI). Contracts extend that trust model *into the build*,
   per-build, with a language-level handler seam — the boundary checker gains
   a runtime sibling.
5. **Design pressure in the right direction.** A precondition that keeps
   growing is a function asking its caller to do too much; writing `pre/post`
   surfaces design smells earlier. The feature is a design instrument, not
   just a safety net.

Caveat: items 1 and 5 depend on culture, not the compiler. The feature pays
only where invariants are actually written down — which is exactly the
C++23 bridge's purpose.


---

## 6. The C++23 bridge: emulating the semantics today

The goal of the bridge is **not** to fake the syntax — it is to (a) adopt the
discipline now, (b) hold a shape that makes the future C++26 switch mechanical,
and (c) get the release-build semantics natively, because **`[[assume]]` is
already C++23** (P1774). The bridge is a ~60-line header, nothing more:

```cpp
// shs/core/contract_guardrails.hpp (proposed — see the adoption proposal)
#if defined(SHS_CONTRACTS_ENFORCED)                 // debug/profile builds
    #define SHS_PRE(...)             /* check -> violation handler */
    #define SHS_POST(...)            /* check -> violation handler */
    #define SHS_CONTRACT_ASSERT(...) /* check -> violation handler */
#else                                                // release: native C++23 assume
    #define SHS_PRE(...)             SHS_ASSUME(!!(__VA_ARGS__))
    #define SHS_POST(...)            SHS_ASSUME(!!(__VA_ARGS__))
    #define SHS_CONTRACT_ASSERT(...) SHS_ASSUME(!!(__VA_ARGS__))
#endif
```

with one project-level violation handler shaped like P2900's from day one
(`kind, location, comment`), so the future switch replaces the handler, not the
handler's call sites.

### The four bridge design rules (these make the switch mechanical)

1. **Mirror the three keywords 1:1** — `SHS_PRE` / `SHS_POST` /
   `SHS_CONTRACT_ASSERT`, named for what they become.
2. **Side-effect-free conditions, always.** In debug the condition *executes*;
   in release it folds to `[[assume]]` and may be *discarded*. A side effect
   would make debug and release observably different — a direct collision with
   Rule 4.1. Conditions are pure reads of state. (Native contracts want the
   same discipline, so nothing is thrown away later.)
3. **Single-expression conditions.** P2900's `pre`/`post` are statement blocks
   (`post(r): ...` names the return value); macros cannot replicate that
   exactly. Keep every condition one trivially-translatable boolean expression
   and the per-site rewrite stays mechanical.
4. **No framework.** If the bridge grows a mini-DSL or control-flow tricks,
   the C++26 migration has to *delete* it — negative value. The bridge holds a
   shape; it is not a feature.

### Boundary rules inside this project

- **Public headers**: contract macros only (they compile away in release),
  never raw checks — the self-containment gate (step 5 packaging, 433/433)
  keeps installed headers portable to consumers on any compiler.
- **Replay parity**: one CTest asserts debug and release builds produce
  identical replay-relevant output with contracts enabled — proving the bridge
  never gates control flow (Rule 4.1). It keeps earning its keep after the
  C++26 switch too.
- **Feature detection**: everything keys on `__cpp_contracts`, never on
  compiler version sniffing.

## 7. The eventual C++26 switch (what it looks like)

When a GCC 16+ baseline reaches the project (a toolchain ruling, same shape as
the C++20→C++23 bump):

1. Where `__cpp_contracts` is defined, the macros expand to native
   `pre`/`post`/`contract_assert` blocks — or sites are rewritten; rule 3
   keeps each rewrite trivial.
2. The violation handler moves to `<contracts>` plumbing; the project policy
   (observe in debug, assume in release) becomes standard build semantic flags.
3. The bridge header shrinks to the feature-test shim, then retires.
4. Acceptance gate: the replay-parity CTest stays green across the switch.

## 8. Reading order

1. The adoption proposal: `docs/backlog/contract_guardrails_adoption_proposal.md`.
2. The adoption todo: `docs/backlog/contract_guardrails_adoption_todo.md`.
3. `domain_value_objects.md` — why the invariants belong to gateways (DVOs).
4. cppreference "C++26 compiler support" — track GCC 16/17 and Clang progress
   before any baseline ruling.
