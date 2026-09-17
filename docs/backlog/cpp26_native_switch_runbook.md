# C++26 native-switch runbook (C4.3 / P4 execution plan)

> **Status: DRAFT — execution blocked on the owner baseline ruling.**
> Facts governing the ruling live in the education doc §7.1
> (`cpp26_contract_guardrails.md`): native contracts (`__cpp_contracts`) and
> the `<contracts>` library open together at **GCC 16**; no Clang release
> supports them yet. On the current baseline (GCC 13.3.0) this runbook is
> inert — execute nothing until the ruling lands.

## 1. Scope: complete site inventory (enumerated 2026-09-17)

Fourteen `SHS_PRE`/`SHS_POST`/`SHS_CONTRACT_ASSERT` sites exist outside the
bridge. Every one is a single-expression condition (bridge rule 2), so the
rewrite is fully mechanical — no site needs restructuring.

| # | Site (file:line) | Macro |
|---|---|---|
| 1 | `logic/logic.gateway.hpp:244` | `SHS_PRE` |
| 2 | `logic/logic.gateway.hpp:284` | `SHS_POST` |
| 3 | `logic/logic.gateway.hpp:285` | `SHS_POST` |
| 4 | `renderpath/renderpath.gateway.hpp:199` | `SHS_POST` |
| 5 | `renderpath/renderpath.gateway.hpp:364` | `SHS_PRE` |
| 6 | `renderpath/planning/render_path_compiler.hpp:124` | `SHS_CONTRACT_ASSERT` |
| 7 | `render/frame/frame.gateway.hpp:69` | `SHS_POST` |
| 8 | `app/session_orchestrator.gateway.hpp:199` | `SHS_POST` |
| 9–11 | `tests/contract_guardrails_tests.cpp:15,19,25` | pre / post / assertion twins |
| 12–14 | `tests/contract_guardrails_replay_probe.cpp:39,40,43` | replay-probe twins |

(P1 placement note: every production site is already in a gateway or the
allowlisted pure leaf — the switch touches no placement law.)

## 2. Preconditions (verify before Step 1; abort if any fail)

1. Baseline compiler defines `__cpp_contracts` (compile a one-line probe;
   per §7.1 this means GCC 16+ — never a compiler-name check, the
   feature-test only, per C4.1).
2. libstdc++ 16 `<contracts>` available (library gate of the same version).
3. Full CTest green on the pre-switch commit; tree clean; `shs_renderer_contract_replay_parity`
   in the suite and green.

## 3. Steps (one mechanical commit, per P4: "one mechanical PR")

1. **Flip the branch** in `contract_guardrails.hpp`: the
   `__cpp_contracts`-keyed branch expands to native keywords instead of
   `SHS_CHECK`:
   `SHS_PRE(c)` → `pre (c)`, `SHS_POST(c)` → `post (c)`,
   `SHS_CONTRACT_ASSERT(c)` → `contract_assert (c)`.
   Macro indirection is kept during the run so the 14 sites are untouched —
   the rewrite is entirely inside the bridge.
2. **Move the handler** to `<contracts>` plumbing: install the project policy
   through `std::contracts::handle_contract_violation` (debug semantic
   `observe`; release semantic `ignore`/assume-equivalent), translating
   `contract_kind` from the native violation info. The `SHS_PRE`/`SHS_POST`/
   `SHS_CONTRACT_ASSERT` expression text and source-location capture move to
   the standard plumbing — kind capture is preserved via one dedicated
   handler per macro.
3. **Retire the emulation ladder**: the `[[assume]]` fold and the no-op
   fallback branches become dead under `__cpp_contracts`; delete them and
   shrink the bridge to the feature-test shim (the "bridge shrinks to a
   shim" DoD).
4. **Build-semantic flags**: debug/release policy becomes standard build
   semantic flags (education doc §7 step 2); `SHS_CONTRACTS_ENFORCED` either
   retires or maps onto the native semantics for test purposes — decide at
   execution, not before.
5. **Site check** (grep, not rewrite): all 14 sites still single-expression;
   if the native grammar rejects any (e.g. the `static_cast` in
   `renderpath.gateway.hpp:364` needs a parenthesized condition), parenthesize
   in place — no logic edits anywhere.

## 4. Acceptance gates (all must be green, same commit)

- `shs_renderer_contract_replay_parity` **byte-identical** (the release
  blocker named in the plan).
- Full CTest green (43+ at the 2026-09-17 baseline, including both contract
  test twins).
- Self-containment gate green in both modes (`-DSHS_CONTRACTS_ENFORCED`
  path retired or remapped — either way, both twins compile).
- `grep -rn "cpp_contracts" cpp-folders/src/shs-renderer-lib` → bridge only
  (C4.1 discipline unchanged).
- Boundaries/rails/placement gates + include-graph green.
- Header inventory regenerated same commit (Rule 15).

## 5. Rollback

Single `git revert` of the run commit restores the emulation ladder in full;
the replay-parity CTest is the arbiter of whether a partial state is
acceptable (it is not — any byte difference aborts the run).

## 6. Bookkeeping after execution

- Mark C4.3 DONE in the adoption todo with the commit hash.
- Re-snapshot the §7.1 toolchain table (Clang status may have moved by then).
- Record the P4 standing item as DONE in the DVO proposal.
- Standing Rule 7.1 deviations review is triggered by the first true dst/src
  job entry, not by this run.
