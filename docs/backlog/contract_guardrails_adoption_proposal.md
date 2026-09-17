# Proposal — Contract Guardrails adoption for shs-renderer-lib (C++23 bridge, C++26-ready)

> Status: **ADOPTED AS LAW (2026-09-17)** — project owner granted all three
> rulings: the bridge + handler are sanctioned, the renderpath pilot is approved,
> and the §3 design rules are binding conventions (Constitution II **Rule 17** +
> Forbidden Pattern 7; Constitution I §11). Implementation proceeds per the todo
> (C0.4 resolved → C1 unblocked).
> Teaching companion: `docs/education/cpp26_contract_guardrails.md` (what the
> feature is, why it fits DVOs, what the bridge looks like).
> Law precedence: Constitution II §2.1 + §2.2 + Rules 4.1/7.1/15; Constitution I §10
> (C++23 baseline — unchanged by this proposal).
> Related: DVO terminology annex `docs/spec/domain_value_object_law.md` (contracts
> are the sanctioned enforcement mechanism of the gateway role, Part 3 — now
> Constitution II Rule 17).

---

## 1. Summary

Adopt **contract-style invariant enforcement** for gateway/edge seams now, via a
small C++23 emulation header, and migrate mechanically to **native C++26
contracts (P2900)** when the toolchain baseline reaches GCC 16+. The proposal
changes no gate, no header move, and no C++ baseline; it adds one header, a
handler, pilot annotations, and two tests.

**What we get now**: every bounded-context invariant written down at the seam
where it is true (executable documentation), debug-build checking of the
no-phantom-flags rule and the Rule 7.1 span contract, and release builds that
already emit native C++23 `[[assume]]` optimizer information.

**What we get later**: a per-site mechanical rewrite to native `pre`/`post`/
`contract_assert` when compilers mature — designed in from day one.

## 2. Motivation (project-specific)

- DVO invariants ("committed plan references only registered techniques",
  "no pending flags", "spans non-overlapping" per Rule 7.1) live today as
  comments and review discipline. Contracts give them a checked home at
  module edges — exactly where §2.1 already puts validation.
- The category line stays intact: `expected`/`or_else` remains the only
  error path (domain failure, replay-safe, compensable); contracts handle
  *unreachable* states only. Rule 4.1 determinism is never routed through a
  contract.
- Toolchain facts (2026-09): P2900 is GCC 16-only; our baseline is GCC 13.3 /
  C++23; Clang/MSVC have no support. Emulation is the only way to adopt the
  discipline early, and the early-stage advantage is real: the codebase is
  small enough to annotate before more bounded contexts accrete.

## 3. The bridge (deliverable C1)

`include/shs/core/contract_guardrails.hpp` — header-only, self-contained,
zero required dependencies beyond `<cstdlib>`/`<cstdio>` level:

| Build type | `SHS_PRE/POST/CONTRACT_ASSERT` expansion |
| :--- | :--- |
| Debug / profile (`SHS_CONTRACTS_ENFORCED`) | check → `shs::contract_violation(...)` handler |
| Release (default) | `[[assume(!!(cond))]]` — native C++23, zero cost |
| C++26 (`__cpp_contracts` defined) | native `pre`/`post`/`contract_assert` (deferred rung, see §6) |

Handler shape (fixed now, P2900-shaped, so the switch is a swap):

```cpp
namespace shs {
enum class contract_kind { pre, post, assertion };
[[noreturn]] void contract_violation(contract_kind kind,
                                     const char* expr,
                                     const char* file, int line);
}
```

Inviolable design rules (full rationale in the education doc §6):

1. **Side-effect-free conditions only** — debug executes them, release may
   discard them; anything else breaks debug/release parity (Rule 4.1).
2. **Single-expression conditions** — keeps the future per-site C++26 rewrite
   mechanical (native `pre`/`post` are statement blocks, not macros).
3. **~60-line ceiling, no framework** — the bridge holds a shape; it is not a
   feature. Growth beyond that is scope creep by definition.
4. **Public headers**: macros only, never raw checks — the self-containment
   gate keeps installed headers compiler-portable (step-5 packaging law).


---

## 4. Pilot scope (deliverable C2)

One bounded context, chosen because its invariants are exactly the shape
contracts were designed for — **the renderpath pod**:

| Seam | Annotation |
| :--- | :--- |
| `renderpath_gateway` commit path | `SHS_POST`: plan pass chain non-empty; every pass entry references a registered technique |
| `RenderPathCompiler` | `SHS_CONTRACT_ASSERT`: only legal technique-mode transitions taken |
| Rule 7.1 job entry (wait-free span contract) | `SHS_PRE`: input/output spans non-overlapping, sizes equal |

Explicitly **out of pilot scope**: input pod edge queues, `logic` FSM internals,
Vulkan runtime headers, anything in `shs/domains/` beyond renderpath.

## 5. Acceptance gates (C2)

- `contract_guardrails_test`: violations abort in debug with the handler
  reporting kind + expression + location; release build performs no check.
- **Replay-parity CTest**: identical recorded input span through the same
  gateway on debug vs release builds produces byte-identical replay-relevant
  output (Rule 4.1). Permanent — survives the C++26 switch.
- Full suite green: all CTest targets + `check_kdba_boundaries.sh` +
  `check_include_graph.py`; header inventory regenerated in the same commit
  (Rule 15: 433 → 434).

## 6. Deferred rung — native C++26 switch (not scheduled)

Trigger: a toolchain ruling that baselines GCC 16+ (same shape as the
2026-09-16 C++23 baseline ruling). Procedure: flip the `__cpp_contracts`
branch in the bridge, rewrite sites (mechanical per bridge rule 2), move the
handler to `<contracts>` plumbing, keep the replay-parity CTest green, retire
the emulation. No date is committed; no code may depend on a date.

## 7. Risks & mitigations

| Risk | Mitigation |
| :--- | :--- |
| Bridge grows into a framework | 60-line ceiling in the todo; review-blocking |
| Conditions acquire side effects | design rule 1 + replay-parity CTest fails loud |
| Contract syntax leaks into installed headers | self-containment gate + bridge rule 4 |
| Debug/release divergence | contracts never gate control flow; parity CTest |
| Toolchain churn mid-pilot | bridge is plain C++23; no `-std=c++26` bump, no compiler jump |

## 8. Alternatives considered

- **Wait for native C++26** — rejected: loses the early-stage window; invariants
  stay unwritten while bounded contexts accrete.
- **Comment-only discipline** — subsumed: todo C0 keeps comment-form contracts
  where annotation isn't yet warranted; the bridge gives the discipline teeth
  for one ~60-line header.
- **`assert()`-style only** — rejected: no postconditions, no assume-folding,
  no migration-shaped handler seam, and `NDEBUG` semantics don't match the
  observe/assume split.

## 9. Decisions requested

1. Approve the bridge header + handler (C1).
2. Approve the renderpath pilot scope (C2) as listed in §4.
3. Adopt §3 design rules as review-blocking conventions once C1 lands.
