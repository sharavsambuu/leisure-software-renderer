# DVO semantics enforcement — proposal

> Status: **proposal (2026-09-17)** — awaiting owner ruling. Landing schedule:
> [`constitution_enforcement_plan.md`](constitution_enforcement_plan.md).
> Companion to
> Constitution II (§6, §8 Kleisli doctrine, Rule 17), the module-layout amendment
> (§6, 2026-09-17), and the adoption todo
> (`contract_guardrails_adoption_todo.md`). Once ratified, items land as
> boundary-gate extensions + law amendments, one commit each, shrink-only.
> Precedence: Constitution II §2.2; this proposal is schedule, not law.

## Motivation

The guardrails bridge (C1) and the monadic `std::expected` rails (Rules 8–12)
give the engine two new semantic axes — *what enforces an edge* and *what an
edge returns* — but current law pins neither the **placement** of guardrails
nor the **shape** of gateway return types mechanically. The DVO module migration
also moves headers without yet moving the checker's scan roots. This proposal
codifies those semantics so they are enforceable, not folklore.

## P1 — Guardrail placement by file role (Rule 17 extension)

Semantics: invariants live in the *type*, edge law lives at the *seam*.

- `SHS_CONTRACT_ASSERT` — value invariants; allowed in
  `*.contract.hpp` / `*.command.hpp` / `*.event.hpp` and pure leaf value
  headers (invariants of the value itself).
- `SHS_PRE` / `SHS_POST` — edge law only; allowed in
  `*.gateway.hpp` / `*.contract.hpp` (the seam files).
- Macro use anywhere else in `include/shs/**` is a gate failure.

DoD: checker gate (8) + negative fixtures (macro in a math leaf FAILs) +
retrofit scan of existing `SHS_*` sites; Constitution II Rule 17 amended with
the placement sentence; Conventions §11 restatement updated same commit
(single-source rule).

> **LANDED 2026-09-17 (W-A of the enforcement plan):** gate 8 =
> `tools/check_contract_placement.sh` (+ negative CTest fixture); Rule 17
> placement sentence + Conventions §11 restatement landed in the same commit.
> Pure-leaf allowlist currently: `renderpath/planning/render_path_compiler.hpp`
> (C2.2 pilot). Retrofit scan result: zero violations across 231 headers.

## P2 — Expected-rail exclusivity at gateways (§8 codification)

Gateway arrows return either a plain value or
`std::expected<T, closed-error-enum>`; no `bool`, status codes, out-params, or
exceptions as failure rails. Error enums already live beside their events
(ERROR_FLOW drift gate); this closes the *return-type* half.

DoD: grep/AST gate over `*.gateway.hpp` returning `bool`/integral status on
failure paths; violations enumerated and triaged (fix or documented exception
in §5 Allowed Exceptions) before the gate turns blocking; one pilot pod first
(`frame`, extending the C2 pilot).

## P3 — Checker follows the module migration (Rules 13–16)

`check_kdba_boundaries.sh` currently globs `domains/*/*.gateway.hpp` etc. When
headers move per manifest, the gate must follow: scan roots derive from the
migration manifest (`engine_header_migration_manifest.json`), not a hardcoded
`domains/` prefix; suffix globs (`*.contract.hpp`, `*.command.hpp`,
`*.event.hpp`, `*.gateway.hpp`) are the invariant.

DoD: checker reads the manifest; negative test moves one pilot header and the
gate still sees it; zero gate regressions on the camera-convention pilot path.

## P4 — C++26 switch runbook (C4.3 enforcement)

The sanctioned rewrite is one mechanical PR: bridge macros → native keywords
per the bridge's mapping, single-expression side-effect-free conditions only;
`shs_renderer_contract_replay_parity` byte-identical; self-containment gate
green in both modes; no new `__cpp_contracts` branches outside
`contract_guardrails.hpp` (already greppable today — keep it that way).

DoD: runbook recorded in the adoption todo C4.3; the parity + boundary gates
are the release blockers named in the commit template.

## P5 — Closed-variant exhaustiveness audit (backlog companion)

Commands/events are closed `std::variant`s; gateways must handle every
alternative (no silent `default:` swallow in seam code — pairs with the
existing `continue;` gate).

DoD: `-Wswitch`-clean seam files asserted in CI (flag already on?) plus a
checker probe banning `default:` in `*.gateway.hpp` command dispatch.

## Sequencing

P3 with the next migration manifest batch (it changes the same files); P1
standalone any time (small); P2 and P5 after the C2 renderpath pilot rules on
ergonomics; P4 stays standing until the toolchain lands GCC 16.
