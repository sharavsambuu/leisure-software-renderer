# Contract Guardrails Adoption — todo list

> Status: **active (2026-09-17)**. Companion proposal:
> `contract_guardrails_adoption_proposal.md` (design + rulings requested).
> Traversal schedule (W-A…W-E, per-pod order):
> [`constitution_enforcement_plan.md`](constitution_enforcement_plan.md).
> Teaching: `docs/education/cpp26_contract_guardrails.md`.
> Law precedence: Constitution II §2.1 + §2.2 + Rules 4.1/7.1/13/15; Constitution I §10.
> Verification after every item: full `build/` CTest green + `check_kdba_boundaries.sh`
> green + `check_include_graph.py` green. Header inventory is content-hashed — any
> header edit regenerates it in the same commit (`inventory_headers.py --write`, Rule 15).
> Items may only be ticked with their DoD met; disputes are recorded inline, never
> silently rewritten.

## C0 — Groundwork (done with the proposal)

- [x] **C0.1 Education material** — DONE 2026-09-17: `docs/education/cpp26_contract_guardrails.md` (what contracts are, why they fit DVOs, the bridge, the C++26 switch story).
- [x] **C0.2 DVO terminology law** — DONE 2026-09-17: `docs/spec/domain_value_object_law.md` + teaching doc; "Domain Value Object" is the official term for the data contracts guard over.
- [x] **C0.3 Proposal drafted** — DONE 2026-09-17: this todo's companion proposal (§9: three rulings requested).
- [x] **C0.4 Owner ruling on proposal** — DONE 2026-09-17: owner ruled contracts law ("we should make this into law"). Rulings 1–3 granted; enacted as Constitution II **Rule 17** + Forbidden Pattern 7 + Constitution I §11. DoD: ruling recorded here and in the proposal status banner — both updated. **C1 is unblocked.**

## C1 — The bridge (one commit, after C0.4)

- [x] **C1.1 `shs/core/contract_guardrails.hpp`** — DONE 2026-09-17 (60 lines exactly): handler seam (`contract_kind`, `contract_violation`, `set_contract_violation_handler`) + macros; enforced → check + handler, release → `[[assume]]` via `__has_cpp_attribute(assume)`, `__cpp_contracts` branch reserved for C4.3. Self-containment gate green on GCC 13.3 / C++23 both ways. `SHS_PRE`/`SHS_POST`/`SHS_CONTRACT_ASSERT`; debug (`SHS_CONTRACTS_ENFORCED`) → check + handler; release → `[[assume]]`; C++26 branch keyed on `__cpp_contracts` (may compile to nothing until a compiler supports it). DoD: include-clean (IWYU), compiles on GCC 13.3 / C++23 with and without `-DSHS_CONTRACTS_ENFORCED`, passes the self-containment gate.
- [x] **C1.2 Violation handler + unit test** — `shs::contract_violation(kind, expr, file, line)`. DONE 2026-09-17: `tests/contract_guardrails_tests.cpp` compiled twice by CMake — enforcement twin asserts kind/expr/file/line capture (pre, post, assertion) + silence on valid conditions + default-handler restore; release twin compiles the assume path (checks discarded, verified by compilation + run). Both in CTest, green. DoD: `contract_guardrails_test` in CTest, green in debug; release build emits no check (verified by compilation, not by running).
- [x] **C1.3 Inventory + gates** — DONE 2026-09-17: inventory regenerated same commit (Rule 15), diff exactly +1 entry (433 → 434). `inventory_headers.py --write` same commit (Rule 15: 433 → 434); `check_include_graph.py` + `check_kdba_boundaries.sh` green; no public header includes contract internals — only the macros. DoD: all gates green, inventory diff exactly +1 line.
- [x] **C1.4 Replay-parity CTest** — DONE 2026-09-17: `shs_renderer_contract_replay_parity` runs the same recorded command span through the same gateway, debug vs release, byte-identical replay-relevant output. DoD: new CTest target green in CI; permanent (survives C++26 switch).

## C2 — Pilot annotations (renderpath pod only)

- [x] **C2.1 `renderpath_gateway` commit postconditions** — DONE 2026-09-17: `SHS_POST` on the committed plan in `try_swap_plan` (pass chain non-empty; every pass entry resolves to a registered standard pass id via the pure helper `renderpath_plan_pass_chain_registered`). Negative test: enforced twin commits a hand-broken empty-chain plan (compatibility rules relaxed) and observes kind=post. DONE in `tests/contract_guardrails_pilot_tests.cpp` (both CMake twins).
- [x] **C2.2 `RenderPathCompiler` transition asserts** — DONE 2026-09-17, P1-legal placement: the technique-mode transition table moved to its pure leaf home (`render_path_recipe.hpp`); the compiler asserts a compiled plan's technique is the table image of its mode (`SHS_CONTRACT_ASSERT` in `render_path_compiler.hpp`, allowlisted pure leaf). Negative test: hand-mismatched technique/mode pair fires kind=assertion. `map_rejection` (reason leg) and `apply_runtime_toggle` (toggle leg) stay INFO-tracked closed-enum dispatches — their leg invariants would need the mappings moved to event/contract headers; ruled at the C2.4 retro, not forced now.
- [x] **C2.3 Rule 7.1 span preconditions** — DONE 2026-09-17: `SHS_PRE` at the `renderpath_gateway` wait-free rim — the immutable commands span and the events output buffer must never alias. Deviation recorded: the *sizes-equal* half of Rule 7.1 has no dst/src job entry in the renderpath pod yet (no input/output span pair exists under `include/shs`); the non-overlap half landed, and the C2.4 retro places the sizes-equal half at the first true dst/src job entry when one lands. Negative test: aliased span storage fires kind=pre.
- [ ] **C2.4 Pilot retro** — measure: did the annotations catch anything? Did any condition need a helper (pure, allocation-free)? Update proposal §4 with findings before sweeping further pods. DoD: retro notes recorded; sweep is blocked on the ruling it produces.

## C3 — Conventions + documentation sweep

- [x] **C3.1 Codify the bridge conventions** — DONE 2026-09-17 (ahead of C2, by owner ruling): Constitution II Rule 17 + Forbidden Pattern 7 + Constitution I §11 enacted. Residual CLOSED 2026-09-17: boundary checker now enforces gate (7) — raw `assert`/`<cassert>` and native `pre(`/`post(`/`contract_assert(` syntax FAIL in Core 4 seam files (contract/command/event/gateway) until the C4.3 switch. DoD: amendment landed ✓; checker extension landed + negative-tested ✓.
- [x] **C3.2 DVO docs sweep (shrink-only)** — DONE 2026-09-17: swept `docs/arch/render_path_architecture.md`, `docs/arch/render_path_domain_pod_architecture.md`, `docs/roadmap/domain_pod_engine_rollout_roadmap.md`, `docs/roadmap/value_oriented_programming_first_class_roadmap.md`, `docs/roadmap/slang_utilization_plan.md`, `docs/pods/DOMAIN_GLOSSARY.md` body (amendment note kept — deliberate retirement quote), `docs/pods/ERROR_FLOW.md` title, and the `kdba_conformance_backlog.md` trigger line. Legacy-term live-doc match count 26 → 0 (T4 shrink-only). Archives untouched (T2); the only remaining live occurrences are the deliberate defining quotes in the law/teaching docs and DOMAIN_GLOSSARY.md's header amendment note.
- [x] **C3.3 Education index** — DONE 2026-09-17: `docs/education/README.md` lists both new teaching docs under "Live teaching" (done together with C0.1/C0.2). DoD: index updated.

## C4 — Native C++26 readiness (standing, not scheduled)

- [ ] **C4.1 Feature-test discipline** — every contract expansion keys on `__cpp_contracts`; no compiler sniffing anywhere. DoD: `grep -rn "cpp_contracts" cpp-folders/src/shs-renderer-lib` shows only the bridge; zero compiler-name branches.
- [ ] **C4.2 Track the toolchain** — note GCC 16/17 and Clang contract support in the education doc's support table when it changes; no baseline action until a toolchain ruling. DoD: table current at each docs sweep.
- [ ] **C4.3 Switch run plan (blocked on C4.2 ruling)** — flip the `__cpp_contracts` branch, rewrite sites (mechanical: single-expression conditions per bridge rule 2), move handler to `<contracts>` plumbing, replay-parity CTest green, retire emulation. DoD: run plan executed and marked DONE with commit hash; bridge shrinks to a shim.

---

## Order & constraints

- C0.4 gates C1; C1 gates C2; C2's retro (C2.4) gates any sweep beyond renderpath.
- C3.2/C3.3 are independent of code and can land before C1 if the user wants the
  terminology law shipped first (recommended: law + education in one commit,
  bridge in the next).
- Every item's commit also regenerates the header inventory when a header is
  touched (Rule 15) and keeps all gates green.
- Archive documents (`kdba_history/`, FROZEN backlogs, `docs/outdated/`) are
  never edited by any item here.

