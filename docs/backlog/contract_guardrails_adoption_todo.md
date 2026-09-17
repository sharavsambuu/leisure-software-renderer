# Contract Guardrails Adoption — todo list

> Status: **active (2026-09-17)**. Companion proposal:
> `contract_guardrails_adoption_proposal.md` (design + rulings requested).
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

- [ ] **C1.1 `shs/core/contract_guardrails.hpp`** — header-only, self-contained, ≤60 lines: `SHS_PRE`/`SHS_POST`/`SHS_CONTRACT_ASSERT`; debug (`SHS_CONTRACTS_ENFORCED`) → check + handler; release → `[[assume]]`; C++26 branch keyed on `__cpp_contracts` (may compile to nothing until a compiler supports it). DoD: include-clean (IWYU), compiles on GCC 13.3 / C++23 with and without `-DSHS_CONTRACTS_ENFORCED`, passes the self-containment gate.
- [ ] **C1.2 Violation handler + unit test** — `shs::contract_violation(kind, expr, file, line)`; test asserts debug aborts report correct kind/expression/location. DoD: `contract_guardrails_test` in CTest, green in debug; release build emits no check (verified by compilation, not by running).
- [ ] **C1.3 Inventory + gates** — `inventory_headers.py --write` same commit (Rule 15: 433 → 434); `check_include_graph.py` + `check_kdba_boundaries.sh` green; no public header includes contract internals — only the macros. DoD: all gates green, inventory diff exactly +1 line.
- [ ] **C1.4 Replay-parity CTest** — same recorded command span through the same gateway, debug vs release, byte-identical replay-relevant output. DoD: new CTest target green in CI; permanent (survives C++26 switch).

## C2 — Pilot annotations (renderpath pod only)

- [ ] **C2.1 `renderpath_gateway` commit postconditions** — `SHS_POST` on the committed plan: pass chain non-empty; every pass entry references a registered technique. DoD: annotations + a negative test (debug build rejects a hand-broken plan).
- [ ] **C2.2 `RenderPathCompiler` transition asserts** — `SHS_CONTRACT_ASSERT` on the technique-mode transition table. DoD: annotations + negative test.
- [ ] **C2.3 Rule 7.1 span preconditions** — `SHS_PRE` at the wait-free job entries: spans non-overlapping, sizes equal. DoD: annotations + negative test.
- [ ] **C2.4 Pilot retro** — measure: did the annotations catch anything? Did any condition need a helper (pure, allocation-free)? Update proposal §4 with findings before sweeping further pods. DoD: retro notes recorded; sweep is blocked on the ruling it produces.

## C3 — Conventions + documentation sweep

- [x] **C3.1 Codify the bridge conventions** — DONE 2026-09-17 (ahead of C2, by owner ruling): Constitution II Rule 17 + Forbidden Pattern 7 + Constitution I §11 enacted. Residual: boundary-checker extension (raw `assert`/contract syntax in seam code) still open — carried to the next gate pass. DoD: amendment landed ✓; checker extension listed ✓.
- [ ] **C3.2 DVO docs sweep (shrink-only)** — live docs adopting "Domain Value Object" terminology per T3 (`docs/spec/domain_value_object_law.md` Part 5): sweep `docs/arch/render_path_architecture.md`, `docs/pods/DOMAIN_GLOSSARY.md` body, roadmap docs at next edit. DoD: live-doc "domain pod" match count strictly decreased; archives untouched.
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

