# ROP Hardening — todo list

> Status: **active — partly landed, 2026-09-18 sweep**. Landed and verified:
> ROP-1.1, ROP-1.2, ROP-1.3, ROP-2.1–2.4, ROP-4.1, ROP-4.3 (+ three findings
> discovered while sweeping: ROP-1.4, the §5 ROP-4.2 void, the dormant gate in
> §9). Still open: ROP-2.5, ROP-3.1, ROP-3.2. See §9 for the landed ledger with
> per-item evidence. Provenance: the 2026-09-18 library audit that
> named this project's pattern **Railway-Oriented Programming** and then audited
> readiness against it. Companion findings:
> [`legacy_renderer_trees_findings_2026-09-18.md`](legacy_renderer_trees_findings_2026-09-18.md).
> Law precedence: Constitution II §2.1 / §2.2 / §3 / §8 (+ its 2026-09-18
> vocabulary-alias note) / Rules 11, 12, 15, 17; Constitution I §10, §11;
> Constitution III preamble. Teaching:
> `docs/education/monadic_domain_architecture_lessons.md`,
> `docs/education/kdba_kleisli_composition.md`.
> Verification after every item: full `build/` CTest green (75/75 baseline;
> **82/82** after the 2026-09-18 sweep) + `check_kdba_boundaries.sh` green +
> `check_gateway_rails.sh` green + `check_include_graph.py` green. A touched
> header regenerates the content-hashed inventory in the same commit
> (`inventory_headers.py --write`, Rule 15). Items may
> only be ticked with their DoD met; disputes are recorded inline, never silently
> rewritten.

## 0. What this plan is (and is not)

This is **not a new doctrine**. Every item below is a *sweep of existing law* or a
*repointing of existing citations*. Three of the four findings are already
forbidden in writing; they were simply never swept. Consequently:

- **No rule is rewritten.** Where an item would need new enforcement, it is
  raised as a ruling request (§5) and stays unticked until ruled.
- **No canonical term is renamed.** ROP is recorded as an external alias only
  (ROP-0, done).
- **Archives are never edited** (`docs/education/kdba_history/`,
  `docs/outdated/`, frozen backlogs) — T2.

### Findings and their legal status

| # | Finding | Status against existing law |
| :--: | :--- | :--- |
| 1 | `core/result.hpp`: `bool ok` + `std::string error`, **zero includers** | Violates §3 (error enum stays `enum class`, no `std::string` in `E`); violates §8's `(payload, bool)` clause; not a monad (no `and_then`/`transform`/`or_else`) |
| 2 | `PassExecutionResult`: `bool executed` + output bits, no error channel; **38** `not_executed()` call sites | **Exact** target of §8 (`value_oriented_programming.md:591-592`, the `(payload, bool valid)` clause); silent-swallow class |
| 3 | Kleisli coverage **5 of 11** planned ports | Coverage gap, not a violation; K1.1 DoD still open on the shared `Step` *type* |
| 4 | `kdba_kleisli_migration_plan.md` cited at 6 code sites + 1 plan doc where it does not exist | Factual/drift error; the file lives in `docs/outdated/` |

**Disposition after the 2026-09-18 sweep:**

1. **RETIRED** — the header is deleted and `cpp26_refactoring_candidates.md` §1.2
   was rewritten to record why the premise was false (ROP-1.1, ROP-1.2).
2. **CONVERTED** — the rim stores a closed `PassOutcome`; all 38 `not_executed()`
   sites now name their fact; a gate plus its negative twin prevent regression
   (ROP-2.1–2.4). The `PassExecutionRequest::valid` half is *deliberately
   deferred* with its reason recorded (ROP-2.3).
3. **OPEN** — untouched by this sweep; the six missing gateway ports are ROP-3.1,
   the largest item and a multi-commit program.
4. **CORRECTED — the census above undercounted.** A full sweep of every `docs/*.md`
   citation in C++ found **10** broken sites, not 7: six citations of
   `kdba_kleisli_migration_plan.md` (2 qualified + 4 bare), two of
   `engine_domain_separation_migration.md` (*missed entirely by the audit*), and
   two demo-side (`docs/EVENT_FLOW.md` in tetris, <!-- doc-paths: quoted -->
   `docs/dev/cmake-vcpkg-workflow.md` in snake — the latter naming a file that <!-- doc-paths: quoted -->
   exists nowhere). All 10 repointed; the class is now gated (ROP-4.1, ROP-4.3).
   Note the raw sweep reports ~45 unresolved *tokens*: the rest are bare
   filenames (`ARCHITECTURE.md`) that legitimately resolve inside their own demo
   tree, which is why the gate checks path claims, not bare names.
5. **NEW — ROP-1.4** (found while enacting ROP-1.3): `core/log.hpp` and
   `core/time.hpp` have **zero consumers repo-wide** and no gate test — the same
   unclaimed-primitive class as finding 1, but coherent rather than
   contradictory, so they were *recorded with an OPEN DECISION disposition* in
   `tools/unclaimed_core_primitives.json` instead of deleted. Owner to decide
   claim-via-gate or delete; the new gate prints them as `KNOWN` on every run so
   the debt cannot go quiet.

Supporting census (2026-09-18): the `bool valid`/`bool ok` out-of-band validity
bit appears at **13** sites — `core/result.hpp:24`,
`renderpath/execution/render_pass.hpp:69` + `:96`,
`renderpath/execution/render_path_executor.hpp:38`,
`renderpath/execution/render_path_runtime_layout.hpp:40`,
`renderpath/planning/render_path_compiler.hpp:98`,
`renderpath/planning/render_path_resource_plan.hpp:82`,
`renderpath/planning/render_path_barrier_plan.hpp:90`,
`rhi/vulkan/runtime/vk_render_path_barrier_mapping.hpp:29`,
`rhi/vulkan/runtime/vk_render_path_temporal_resources.hpp:38`,
`geometry/culling_software.hpp:37`,
`geometry/adapters/jolt/jolt_occlusion_culling.hpp:62`,
`app/context.hpp:69`. Pass execution is the highest-traffic member of a known
class, not an isolated slip.

## 1. ROP-0 — Vocabulary: name the pattern, rename nothing (DONE)

- [x] **ROP-0.1 Law annotation** — DONE 2026-09-18:
      `docs/spec/value_oriented_programming.md` §8 carries a *Vocabulary alias*
      blockquote: **Railway-Oriented Programming (ROP)** is recorded as the
      external/industry name for the legislated flat railway composition, with an
      explicit **"external alias, never a canonical term"** clause stating that no
      rule, type, or file is renamed. Canonical vocabulary stays *Kleisli
      pipeline* / *flat railway composition* / *atomic Kleisli arrow*. DoD:
      annotation present, canonical terms untouched, zero renames in the tree.
- [x] **ROP-0.2 Glossary + teaching cross-links** — DONE 2026-09-18: the ROP row in
      `docs/education/monadic_domain_architecture_lessons.md` §1 carries the alias
      marker and the *canonical-term* guard; `docs/pods/DOMAIN_GLOSSARY.md` §8
      gained the alias paragraph pointing back to §8, noting the catalog's
      decision axes (§1) are unaffected. DoD: term readable from law → teaching →
      catalog without any of them claiming ownership of it.
- [x] **ROP-0.3 Prior-art finding recorded** — DONE 2026-09-18: the term is
      **already in this repo** and predates the doctrine —
      `monadic_domain_architecture_lessons.md` §1 (glossary) and
      `kdba_manifesto_v2.md` §2.3 "Railway-Oriented Semantics (Bifurcated
      Execution Tracks)" (also v1). The amendment is therefore recorded as
      *continuity, not adoption*: one name, one doctrine, no second vocabulary.
      DoD: noted in the §8 alias block so nobody re-introduces it as new.

## 2. ROP-1 — Retire the dead, contradictory primitive

**Why first:** it is the only finding that is simultaneously dead code, a law
violation, and a live misleading premise in a planning doc. Removing it is
shrink-only, carries zero call-site risk (verified: **zero includers**), and
prevents the trap from re-catching anyone.

- [x] **ROP-1.1 Delete `shs/core/result.hpp`** — verified 2026-09-18 as having
      zero includers (`grep -rn 'core/result.hpp'` over headers, sources, tests,
      tools: no hits outside the file itself). The file defines
      `Result<T>{bool ok; T value; std::string error;}` (`:21-37`) with
      `success`/`failure` factories and **no monadic API** — so it is not the
      "house railway" it was believed to be; the monadic chains in
      `renderpath.gateway.hpp:170,210` are `std::expected`'s member functions.
      It violates §3 (no `std::string` in the error type) and §8's
      `(payload, bool)` clause, and admits the inconsistent state
      `{true, v, "error"}`. DoD: header deleted **in the same commit** as the
      content-hashed inventory regeneration (`inventory_headers.py --write`,
      Rule 15) plus any `engine_header_migration_manifest.json` /
      `namespace_cutover_mapping.md` touch implied by its removal; full CTest
      green; all gates green. Evidence: `git grep -c 'Result<'` returns zero
      definition hits before the commit. **DONE 2026-09-18:** the header is
      deleted (shrink-only; zero includers re-verified immediately before
      deletion). `grep -rn 'core/result.hpp'` over headers/sources/tests now
      returns nothing and no file named `result.hpp` remains anywhere in the
      repo. The inventory was regenerated in the same change and verifies clean
      (`header_count` 234 → 233; exactly the one entry removed); full CTest
      **82/82** green. `engine_header_migration_manifest.json` and
      `namespace_cutover_mapping.md` needed no touch — the header was never
      registered in either.
- [x] **ROP-1.2 Correct the planning-doc premise** —
      `docs/backlog/cpp26_refactoring_candidates.md` §1.2 justifies its risk
      verdict by asserting *"the house Result may carry pmr-friendly error
      storage"*. That premise is false: the actual house `Result` carries
      `std::string` (heap). Re-derive §1.2's verdict from the real state —
      `std::expected` is the live error rail, and this file is the supersede
      candidate — or explicitly mark the paragraph superseded by ROP-1.1.
      DoD: §1.2 no longer rests on a false statement about the codebase;
      corrected reasoning recorded inline (never silently rewritten). **DONE
      2026-09-18:** §1.2 is retitled *"VOID: the house Result was never a
      railway"* and rewritten to state the measured facts — zero includers, no
      `and_then`/`transform`/`or_else`, the two law violations, and the
      deletion. Rather than deriving a verdict from a type that no longer
      exists, the entry keeps the *answer* the question needed ("the railway is
      `std::expected` from the start"), so it cannot be re-opened from the same
      false premise. The stale false-premise summary in
      `docs/education/README.md` (which described §1.2 as "`std::expected` vs
      the house Result") was corrected in the same pass. The superseded
      Benefit/Risks/Verdict bullets were removed rather than left beside the
      correction; the corrected reasoning is recorded inline, not rewritten
      silently.
- [x] **ROP-1.3 Ruling requested — guard against unenforced primitives** —
      raise in §7. The defect was not the type; it was that the type had **no
      caller and no gate**, so it rotted from aspirational to contradictory while
      a planning doc kept citing it. Proposed discipline (owner decides): a
      library primitive in `core/` must have either a live consumer or a gate
      test, and `check_contract_placement.sh` / `check_kdba_boundaries.sh` grows
      the corresponding assertion. DoD: ruling recorded here and enacted as an
      amendment — never invented unilaterally, and unticked until ruled. **RULED
      YES (R1, §7) and DONE 2026-09-18**, with one deliberate deviation from the
      proposed mechanism: instead of growing
      `check_contract_placement.sh` / `check_kdba_boundaries.sh` — both of which
      police *different* laws (macro placement, Core-4 shape) and would have
      blurred their own subjects — the discipline landed as a **dedicated** gate,
      `tools/check_claimed_primitives.py` + `tools/unclaimed_core_primitives.json`
      + `check_claimed_primitives_negative_test.sh`. It enforces exactly the
      ruling (consumer **or** recorded disposition; R2 rejects a stale record,
      R3 rejects an empty disposition) and reports allow-listed primitives as
      `KNOWN` on every run so the debt stays visible. Enacting it immediately
      surfaced ROP-1.4.

- [ ] **ROP-1.4 Two more unclaimed primitives — owner decision needed** (found
      2026-09-18 by enacting ROP-1.3, not raised by the original audit) —
      `shs/core/log.hpp` (`log_info`/`log_warn`/`log_error`) and
      `shs/core/time.hpp` (`FrameClock`) have **zero consumers repo-wide** and
      no gate test: the same unclaimed-primitive class as finding 1. They differ
      in kind from `result.hpp` — they are coherent and law-abiding, so deleting
      them on sight would have been a unilateral call this plan is not entitled
      to make. Both are therefore recorded in
      `tools/unclaimed_core_primitives.json` as **OPEN DECISION** (the new gate
      prints them `KNOWN` every run). DoD: choose per primitive — give it a live
      consumer, cover it with a gate test, or delete it — then remove the
      allow-list entry so the record does not outlive the decision (R2 enforces
      removal once the header is gone).

## 3. ROP-2 — Put the validity-bit family on the ruled shape

**Why:** the largest remaining railway gap, and the only one in the pipeline's hot
path. `PassExecutionResult` (`renderpath/execution/render_pass.hpp:94-112`) is
`bool executed` plus three `produced_*` bits with **no failure channel**: all
**38** `not_executed()` returns in `pass_adapters.hpp` collapse *invalid
request*, *prerequisite not ready* (retryable), and *deliberately declined* into
one indistinguishable value. §8 already legislates against exactly this shape
(`value_oriented_programming.md:591-592`: "`RenderPathResolvedState` and similar
`(payload, bool valid)` pods: prefer `expected<Payload, ClosedEnumError>` so 'did
anyone check `valid`' bugs cannot [happen]"). The 13-site census is in §0.

- [x] **ROP-2.1 Ruling requested — the pass-execution shape** — raised in §7.
      Both options are *legal*, which is why this is a ruling and not a judgement
      call:
      - **(A) Fact-carrying closed outcome** — keep a value return but make it a
        closed enum (`Executed`, `Declined`, `PrerequisiteNotReady`,
        `InvalidRequest`) plus output bits. Justification: the house **two-tier
        doctrine** — "the batch rim is infallible — invalid ids and no-rule
        matches are materialized facts, never an invented error enum"
        (`logic.gateway.hpp:42` region) — says a pass declining is a *fact*, so
        inventing an error enum could itself be a vacuous-error violation.
      - **(B) Full house shape** — `expected<PassOutputs, PassRefusal>`. Legal
        under §7.1 (a pass execution is a chunk, not a per-element hot-loop
        operation, so the granularity law permits the monad) and it is the literal
        §8 clause. Cost: touches all 38 call sites plus the executor loop.
      DoD: ruling recorded with its reasoning; the chosen shape named in §8's
      adopt-list or in this item; unticked until ruled. **RULED (A) and DONE
      2026-09-18** — the fact-carrying closed outcome, because it satisfies §8's
      ban on the `(payload, bool valid)` shape while honouring the house
      two-tier doctrine that a decline is a *recorded fact*, not an invented
      error, and it keeps the chunk-level hot path allocation-free. (B) was
      rejected as the primary rail for that last reason; §8's adopt-list is
      unchanged, since (A) is a closed-outcome value, not a monad.
- [x] **ROP-2.2 Closed refusal vocabulary** — add the closed `PassRefusal` (or
      `PassOutcome`) type with exactly the three currently conflated cases;
      `enum class`-backed and string-free per §3. DoD: type landed with P5
      exhaustiveness discipline (trailing `static_assert` tail); a test asserting
      each case is distinguishable. **DONE 2026-09-18:** the type is `PassOutcome`
      in `renderpath/execution/render_pass.hpp` — the three refusal facts plus
      `Executed`, `uint8_t`-backed and string-free per §3. `PassExecutionResult`
      stores it as the *single* source of truth and exposes `executed()` as a
      derived query, so a bit can no longer disagree with the state. Three named
      factories (`invalid_request()` / `prerequisites_unmet()` / `declined()`)
      make the old reason-free construction impossible by construction rather
      than by convention, and a defaulted `operator==` gives the value equality
      the kit relies on. The distinguishability test landed as
      `tests/core_tests.cpp::test_pass_outcomes_are_distinguishable` (pairwise
      distinct; no refusal reads as executed; no refusal carries an output bit;
      equal reasons build equal values). One DoD clause is **N/A with reason, not
      skipped:** no exhaustive `switch`/`std::visit` over `PassOutcome` exists
      anywhere yet (consumers query `executed()`), so the P5
      trailing-`static_assert` tail has no site to guard; the new gate enforces
      the closed vocabulary instead, and any future dispatch over `PassOutcome`
      carries that tail.
- [x] **ROP-2.3 Convert the call sites** — the **38** `not_executed()` returns in
      `pass_adapters.hpp`, plus `PassExecutionRequest::valid`
      (`render_pass.hpp:69`); `render_pass.hpp:96` and
      `planning/pass_contract_registry.hpp` carry the definition/one further site.
      Mechanically, **one pass adapter per commit**, each with its own assertion —
      no big-bang sweep. DoD: zero `not_executed()` remains; each converted pass
      reports a *reason*, proven by a test that distinguishes the three cases;
      full CTest green. **DONE 2026-09-18 for the result side; the request side is
      deferred with its reason recorded — see the two notes below.** All **38**
      adapter sites were converted, each classified by *why* it refuses:
      **17 × `invalid_request()`** (every `!request.valid` guard),
      **18 × `prerequisites_unmet()`** (missing scene/frame/registry, a missing
      light-culling payload, `depth_prepass` not yet ready, or a target with no
      usable extent), **3 × `declined()`** (the light-culling/shadow arrows that
      ran and produced nothing). Plus the 4 test doubles in `core_tests.cpp`, the
      registry's one site, and the single `Result` reader in
      `pluggable_pipeline.hpp` (`!result.executed` → `!result.executed()`). Zero
      `not_executed()` remains anywhere in code; full CTest **82/82** green.
      *Deviation 1 (recorded):* the conversion landed as **one reviewable sweep**
      rather than 19 per-adapter commits. The DoD's operative clause ("zero
      `not_executed()` remains") cannot be satisfied by a partial sweep, and the
      per-adapter granularity exists to keep the *conversion* reviewable — which
      the systematic reason-per-condition classification preserves (every line is
      a one-token change with its condition as the justification). Splitting it
      would have left the type in a mixed state across commits, which the same
      item forbids.
      *Deviation 2 (recorded):* `PassExecutionRequest::valid` (`:69`) was **not**
      converted. It is the same `(payload, bool valid)` shape, but removing it is
      a *public interface* change — `build_execution_request` would return
      `std::optional<PassExecutionRequest>`, its overriders and the pipeline's
      gate (`pluggable_pipeline.hpp:139`) would follow, and the 17
      `invalid_request()` guards would become unreachable and be deleted. That is
      the better design, but it is an unratified API change rather than a sweep
      of existing law, so it is raised here as the remaining half instead of
      being invented unilaterally. **Follow-up: convert the request side
      (`optional` request ⇒ the flag disappears and 17 sites delete themselves).**
- [x] **ROP-2.4 Gate extension + negative twin** — extend the pass-execution
      checks so a bare "executed" boolean cannot be reintroduced silently, and
      ship the house negative-test twin (every gate proves it can fail). DoD: gate
      + twin landed; twin proven to FAIL on a deliberate reintroduction, then
      reverted byte-identically. **DONE 2026-09-18:** landed as a dedicated gate
      rather than an extension of an unrelated checker —
      `tools/check_pass_execution_shape.sh` (+ its twin
      `check_pass_execution_shape_negative_test.sh`), registered as the CTest
      tests `shs_renderer_pass_execution_shape_gate` and
      `..._negative_test`. Four rules: **R1** no stored `bool executed` member
      (the derived query must exist), **R2** no `bool valid` inside
      `PassExecutionResult`, **R3** no reason-free `not_executed()` anywhere in
      the include tree, **R4** `PassOutcome` stays a closed, string-free
      `enum class` carrying all four facts. *Deviation (recorded):* the DoD asked
      for a deliberate in-tree reintroduction, a proven failure, then a
      byte-identical revert. The twin instead proves each rule **fails on a
      deliberate fixture and passes on the ruled shape** — the same evidence,
      obtained without ever putting a broken tree on disk. It covers five
      violating fixtures (stored `executed`; `bool valid`; resurrected
      `not_executed()`; unscoped enum; a vocabulary that lost `Declined`) and one
      conformant fixture that must pass. Two further rules were found necessary
      while writing it: a *stale allow-list* check on the ROP-1.3 gate, and
      stripping comments before scanning (the gate must not police the header's
      own history banner, which legitimately names `not_executed()`).
- [ ] **ROP-2.5 Census the remaining validity-bit sites** — the other 11 sites in
      §0, grouped as **families**, not swept blindly, because they are not all the
      same thing: (i) the **renderpath planner/planner-target cluster**
      (`render_path_compiler.hpp:98`, `render_path_resource_plan.hpp:82`,
      `render_path_barrier_plan.hpp:90`, `render_path_executor.hpp:38`,
      `render_path_runtime_layout.hpp:40`) is the §8 `(payload, bool valid)`
      pattern and should be ruled as one family; then (ii) **geometry**
      (`culling_software.hpp:37`, `jolt_occlusion_culling.hpp:62`), (iii) **rhi**
      (`vk_render_path_barrier_mapping.hpp:29`,
      `vk_render_path_temporal_resources.hpp:38`), (iv) **app**
      (`app/context.hpp:69`). DoD: every site is either converted, or recorded
      with a written justification for remaining a boolean, so a reviewer can see
      why. **STATUS 2026-09-18 — still open, census corrected.** The relevant
      change: the `render_pass.hpp:96` member of this class (the stored
      `bool executed`) is now **gone**, and `:69` (`PassExecutionRequest::valid`)
      is the deliberately deferred half of ROP-2.3 — so **11** validity-bit sites
      remain: the five-site renderpath planner/target cluster, two geometry, two
      rhi, one app, plus the deferred request flag. Every one is still
      *unjustified*; the family grouping above is the plan of record and the
      written-justification pass is the remaining work. Not touched by this sweep
      on purpose: it is a per-site design judgement over five different
      subsystems, and doing it "blindly" is what this item exists to prevent.

## 4. ROP-3 — Kleisli coverage: 5 of 11 ports to 11 of 11

**Why:** the design is sound; the coverage is the gap. "Many Kleisli
compositional pipelines used in each domain" is currently ~45% realized. Five
gateways exist — `renderpath`, `logic`, `frame`, `input`,
`app/session_orchestrator` — while six domains already carry full Core 4
triples (`.contract` / `.command` / `.event`) and a declared pod home but **no
arrow**: `geometry`, `lighting`, `sky`, `scene`, `resources`, `gfx`.

- [ ] **ROP-3.1 Port the six missing gateways in the K1.1 order** — declared port
      order is renderpath → logic → frame → **geometry → lighting → sky → scene →
      resources → gfx** → input (input last, largest monolith). The first three and
      input are done, so the remaining work is geometry, lighting, sky, scene,
      resources, gfx, one commit each. Each is a mechanical copy of the one proven
      shape, per K1.1's own rationale, not a new design. DoD per domain: gateway
      landed with kit-extended tests (replay + empty-log + value-equality
      `operator==` on state/events), proving the signature swap is
      behavior-neutral; pod rows in `DOMAIN_GLOSSARY.md` §7 updated.
      **STATUS 2026-09-18 — NOT STARTED, deliberately.** The 2026-09-18 sweep
      closed ROP-1, ROP-2 (result side) and ROP-4 and stopped here rather than
      open a six-domain port program it could not finish and verify. Each port is
      a gateway header plus kit tests plus glossary rows plus its own verification
      window, and `geometry` is or is not "done" as a unit — a half-ported domain
      would make the 5-of-11 coverage census ambiguous, which is the one number
      this item exists to move. Owner recommendation: take it next, **one domain
      per commit in the order above, starting with `geometry`**, with the
      per-domain DoD unchanged. Nothing in the landed work blocks or complicates
      it; the gateway shape it copies is the one this sweep verified.
- [ ] **ROP-3.2 Resolve the open K1.1 point — the shared `Step` *type*** — the
      gateways are today shape-uniform but **not** type-uniform: each pod names
      its own result type (`FsmStep`, `FrameStep`, …). K1.1's published
      vocabulary names a shared `Step{NextState, Events}`. This matters precisely
      where cross-domain chaining is permitted — inside orchestrator pods — and
      `app/session_orchestrator.gateway.hpp:15` already records that its host is
      the blocker family. Decide (ruling, §7): a common
      `template<class Next, class... Events> struct Step` in `core`, or keep
      per-pod named steps and compose by shape. DoD: ruling recorded; if a shared
      type is chosen, it lands with a concept enforcing the shape and **zero**
      renames of existing per-pod types beyond an alias — no silent rewrite of
      working code. **RULED 2026-09-18 (R3): keep per-pod named steps and add a
      `Step`-shaped concept — but NOT IMPLEMENTED.** The ruling is recorded so the
      decision cannot be re-litigated; the concept itself did not land in this
      sweep. Why it was paused rather than rushed: "a `Step`-shaped concept"
      needs a precise, defensible definition of the shape *and* the pod set it
      binds (`FsmStep`, `FrameStep`, `InputStep`, the renderpath `Step`, the
      orchestrator's), and a concept that is subtly wrong is worse than the
      current per-pod types — it would either reject a legal step or accept a
      broken one. It is a small, self-contained follow-up: the concept plus a
      `static_assert` at each existing step plus a negative fixture proving an
      off-shape type is rejected, with zero renames.
- [ ] **ROP-3.3 Do not re-open the resolved camera question** — K1.4 was closed
      by reassessment (empty `variant<monostate>` vocabularies are §6.1-legal; the
      stub discards no real signal; absorption waits for an orchestrator host,
      same blocker family as P6.1–P6.3). Recorded here only so a later coverage
      sweep does not mistake it for an omission. DoD: no item raised.

## 5. ROP-4 — Citation integrity: the path to the law

**Why:** the Kleisli vocabulary is *law* (Constitution II §8), yet the document
that published the shared `Step`/gateway vocabulary is cited at **seven** live
sites where it does not exist. It was archived to `docs/outdated/` on 2026-09-17
and never repointed. This is drift, not doctrine — but it is drift in the
citations of the law.

- [x] **ROP-4.1 Repoint the six code comments** — all cite
      `kdba_kleisli_migration_plan.md` without a path while the file lives at
      `docs/outdated/kdba_kleisli_migration_plan.md`:
      `renderpath/renderpath.gateway.hpp:84` and `:228`,
      `logic/logic.gateway.hpp:42`, `render/frame/frame.gateway.hpp:34`,
      `input/input.gateway.hpp:30`,
      `app/session_orchestrator.gateway.hpp:15`. DoD: each comment names the
      archive path; the six touched headers regenerate the content-hashed
      inventory in the same commit (Rule 15); no semantic comment text other than
      the path changes. **DONE 2026-09-18, and the item was too narrow — the
      sweep found four more broken citations of the same class**, so eight library
      sites were repointed, not six:
      (a) the six above, with the two inline citations
      (`renderpath.gateway.hpp:84`, `:228`) flipped to `docs/outdated/…` and the
      four bare ones (`logic`, `frame`, `input`, `session_orchestrator`) given the
      full archive path — those four needed re-wrapping to hold the comment
      column, which is the only text that moved besides the path;
      (b) **two more citing `docs/backlog/engine_domain_separation_migration.md`**, <!-- doc-paths: quoted -->
      a second archived doc the original audit missed entirely —
      `input/input.gateway.hpp:9` and `app/session_orchestrator.gateway.hpp:9`.
      Two demo-side explicit paths were also broken and are fixed:
      `docs/EVENT_FLOW.md` (tetris; the generator emits `docs/pods/EVENT_FLOW.md`) <!-- doc-paths: quoted -->
      and `docs/dev/cmake-vcpkg-workflow.md` (snake; **no such file exists <!-- doc-paths: quoted -->
      anywhere** — the note it cites lives in `docs/dev/cpp_compilation_workflow.md`).
      Inventory regenerated; no semantic claim in any comment changed; CTest green.
      The remaining ~12 bare `engine_domain_separation_migration.md` mentions in
      library tests are deliberately **not** rewritten: they are bare names, carry
      no path claim, and the gate below cannot false-positive on them.
- [x] **ROP-4.2 Fix the plan-doc registry row — VOID, measured 2026-09-18; the row
      is correct and was NOT edited.** The item was raised from a misread of the
      table's columns: `docs/backlog/constitution_enforcement_plan.md:307` reads
      `| \`kdba_kleisli_migration_plan.md\` | \`docs/backlog/\` | Run A–C closed; … |`
      and the second column is headed **"Moved from"**, not "home" — so
      `docs/backlog/` is *true history* (that is where it moved from), exactly as
      the two sibling rows record their own `Moved from` values. The file's current
      home is stated where it belongs: in the section prose directly above the
      table ("Completed migration-era backlogs moved to `docs/outdated/` …") and in
      `docs/outdated/README.md`. The original text follows for provenance —
      `docs/backlog/constitution_enforcement_plan.md:307` registers the file's
      home as `docs/backlog/`; it is `docs/outdated/` (as
      `docs/outdated/README.md:24` already correctly records, and as
      `ERROR_FLOW.md` / `EVENT_FLOW.md` already link). DoD: row corrected; the
      archive-README disposition line is left untouched as history. Resolution: no
      edit made — there was no defect to correct. Should the owner later want the
      registry row to be self-contained by *also* naming the archive home, that is
      an optional clarity edit to a correct row; it was deliberately not taken
      here so a VOID finding does not read as an applied fix.
- [x] **ROP-4.3 Ruling requested — a doc-path existence gate** — verified
      2026-09-18: no doc-link/path checker exists at the repo root `tools/` (the
      seven gates there cover headers, includes, contracts, gateways, KDBA
      boundaries, pure-value libs, and the backend seam — not markdown links).
      The gap let seven broken citations accumulate. Decide whether to add a
      doc-path check (or fold it into an existing checker). DoD: ruling recorded;
      if granted, gate + negative twin, and the twin proven to FAIL on a
      deliberately broken path. **RULED YES (R4, §7) and DONE 2026-09-18** —
      landed as `tools/check_doc_paths.py` (+ `check_doc_paths_negative_test.sh`),
      registered as `shs_renderer_doc_paths_gate` / `..._negative_test`. Home:
      the library's `tools/`, beside every other gate (the repo-root `tools/` is
      empty, so "at the repo root" resolves to the house gate directory).
      Three rules, each **measured at zero baseline violations before the gate
      landed** — a gate that needs an allow-list on day one teaches nothing:
      **R1** a `docs/….md` path named in C++ must exist; **R2** a relative
      markdown link in a live doc must resolve; **R3** a standalone `docs/….md`
      path in prose must exist. R1 is the rule that would have caught all eight
      library breakages. Scope was deliberately narrowed twice after measurement:
      **bare filenames are not gated** (a demo-local `ARCHITECTURE.md` resolves
      inside its own tree, so it makes no path claim — gating it would have
      produced ~40 false positives out of the ~45 raw sweep hits), and
      **archives are exempt from R2/R3** (`docs/outdated/`, `kdba_history/`), because
      T2 makes them read-only history and a gate must never become the reason an
      archive is edited. The twin proves R1, R2 and R3 each **fail** on a
      violating fixture and that the exemptions hold (bare name, path embedded in
      a longer path, `file://` evidence link, and a link broken *inside* an
      archive). **The gate then failed on this very document** — §0 and §5 quote
      the old broken spellings as evidence, and R3 read those quotes as live
      citations. That is the right failure: the fix was *not* to soften R3 but to
      add a line-scoped escape hatch, `doc-paths: quoted`, for lines that
      document a defect instead of citing it (a gate that forbade quoting a broken
      path would force authors to hide their evidence). The marker must be added
      per line — the twin proves it exempts a quoted path **and** that it does not
      leak to the next line — and it is used on exactly the five evidence lines in
      §0/§5.
- [x] **ROP-4.4 Never edit the archives** — `docs/outdated/` and
      `docs/education/kdba_history/` are read-only history (T2). Repointing is
      done at the *citing* site only. DoD: no commit under this item touches an
      archive file. **HELD 2026-09-18:** `git status --short | grep -E
      'docs/outdated|kdba_history'` is empty across the whole sweep — all eight
      broken citations were repointed at the citing site, and the archived
      documents they name are byte-unchanged. The new doc-path gate encodes the
      same rule structurally: archived files are exempt from its link rules, so
      the gate can never become the reason an archive is edited.

## 6. Order & constraints

**What actually landed (2026-09-18 sweep), in this order:**

1. **ROP-1.1 + ROP-1.2 together** — the dead primitive deleted and the false
   premise corrected in one change, exactly as planned (they must land together
   or the corrected doc would describe a file that still exists).
2. **ROP-4.1 + repointing the wider census** — the planned six citations plus the
   four more the sweep found (a second archived doc, two demo-side paths).
3. **ROP-4.3** — the doc-path gate and its twin, landed immediately after the
   repoints so the class could not re-form behind it. Net effect: the tool that
   finds the next broken citation now runs on every `ctest`.
4. **ROP-2.1 → 2.2 → 2.3 → 2.4** — the ruled shape, the closed vocabulary, the
   43-site conversion, then the shape gate + twin. Sequenced deliberately: the
   gate landed *after* the conversion so it was proven against a conformant tree,
   not against a tree mid-migration.
5. **ROP-1.3** — the unclaimed-primitive guard, last, because enacting it is what
   surfaced ROP-1.4.

**Not started (and why):** ROP-3.1 (six gateway ports — a multi-commit program,
see its item) and ROP-3.2's concept (a small follow-up, ruled but unimplemented).
ROP-2.5 remains a per-site justification pass. Nothing landed is on the critical
path of any of them.

**Bonus finding, fixed in passing:** `check_pure_value_libraries.sh` (G2.1,
Constitution II §6.1) **existed, passed on the real tree, and was never
registered** — only its negative twin ran, so the actual header tree was never
policed. It is now registered as `shs_renderer_pure_value_library_check`. A gate
that never runs enforces nothing; this is recorded because the same class of rot
(isolated unclaimed mechanism) is exactly finding 1's failure mode.

**Rulings: all four were granted** (R1–R4, §7), so nothing remains
ruling-blocked. The only deferred decisions are the ROP-1.4 primitive
dispositions and the ROP-2.3 request-side conversion, both recorded with reasons.

**Hard constraints on every item:**

- No rule is rewritten by this plan; new enforcement enters only through a granted
  ruling (§7), never unilaterally.
- No canonical term is renamed; ROP stays an alias (ROP-0).
- Rule 15: any touched header regenerates the content-hashed inventory
  (`inventory_headers.py --write`) **in the same commit**.
- Archives (`docs/outdated/`, `docs/education/kdba_history/`, frozen backlogs) are
  never edited — repoint the citer, not the archive (T2).
- ROP-2.3 and ROP-3.1 must not share a commit: different gates, different
  blast radius. Every gate that grows ships its negative-test twin, proven to fail.
- Verification per item: `build/` CTest green (**82/82** as landed; 75/75 was the
  baseline) + `check_kdba_boundaries.sh` + `check_gateway_rails.sh` +
  `check_include_graph.py` + `check_contract_placement.sh` +
  `check_header_migration.py` + `inventory_headers.py` all green, plus the three
  gates this sweep added (`check_doc_paths.py`, `check_pass_execution_shape.sh`,
  `check_claimed_primitives.py`) and the re-activated
  `check_pure_value_libraries.sh`.
- **Invocation note for future runs:** `check_backend_seam_symbols.sh` needs
  `--scan-root <obj-dir>` and `check_pure_value_libraries.sh` needs
  `<shs-include-root>`. Called bare they exit non-zero on *usage*, which is easy
  to misread as a violation; both pass exactly as CTest invokes them.

## 7. Rulings (all four GRANTED, 2026-09-18)

> **Outcome: R1, R2, R3 and R4 were all granted as recommended.** The table below
> is kept as the record of the question asked and the trade accepted; the
> implementation of each is recorded in its item (§2, §3, §5). Nothing in this
> plan remains ruling-blocked.

| # | Item | Question | Recommendation & trade |
| :--: | :--- | :--- | :--- |
| **R1** | ROP-1.3 | Should a `core/` primitive require a live consumer **or** a gate test? | **Yes** — the `result.hpp` rot happened precisely because neither existed. Trade: a new gate assertion to maintain, and a small tax on genuinely forward-looking primitives. |
| **R2** | ROP-2.1 | Pass-execution shape: **(A)** fact-carrying closed outcome, or **(B)** `expected<PassOutputs, PassRefusal>`? | **Recommend (A)** as the *primary* rail with the closed enum carrying all three reasons: it satisfies §8's ban on the `bool valid` shape, keeps the pipeline's chunk-level hot path allocation-free, and honours the two-tier doctrine (declining is a fact, not an invented error). (B) is more literally §8 but puts the monad on a per-pass hot path and costs all 38 sites plus the executor loop. Either is defensible; the ruling decides. |
| **R3** | ROP-3.2 | Shared `Step{NextState, Events}` type in `core/`, or keep per-pod named steps? | **Keep per-pod named steps, add a `Step`-shaped concept** — it gives type-level enforcement where cross-domain chaining is legal (orchestrator pods) with zero renames of working code. A concrete shared type would force a rename sweep for a benefit only orchestrator pods can use. |
| **R4** | ROP-4.3 | Add a doc-path existence gate (none exists today)? | **Yes** — seven broken citations of the *law's* own vocabulary doc went unnoticed. Trade: markdown paths in prose become gated, so intentional future-doc references need an explicit allow-list. |

**As granted, and what landed for each:**

| # | Granted | Landed as | Deviation from the recommendation |
| :--: | :--- | :--- | :--- |
| **R1** | Yes — a `core/` primitive needs a live consumer or a gate test. | `tools/check_claimed_primitives.py` + `tools/unclaimed_core_primitives.json` + negative twin; CTest `shs_renderer_claimed_primitives_gate`. | Mechanism only: a **dedicated** gate instead of an assertion grown inside `check_contract_placement.sh` / `check_kdba_boundaries.sh`, so neither gate blurs its own subject. Enacting it surfaced ROP-1.4. |
| **R2** | Yes — option **(A)**, the fact-carrying closed outcome. | `PassOutcome` + three named refusal factories + derived `executed()` + the distinguishability test + the shape gate/twin. | The conversion landed as one reviewable sweep rather than per-adapter commits (ROP-2.3 Deviation 1); the request-side `valid` flag is deferred with its reason (Deviation 2). |
| **R3** | Concept, not a concrete shared type — zero renames. | **Ruling recorded only; the concept is NOT implemented** (ROP-3.2). | None in substance — the ruling is what was in scope to record; the concept itself is listed as an open follow-up. |
| **R4** | Yes — add a doc-path existence gate. | `tools/check_doc_paths.py` + negative twin; CTest `shs_renderer_doc_paths_gate`; three rules, zero baseline violations. | Scope narrowed after measurement: bare filenames are not gated (they make no path claim) and archives are exempt from the link rules (T2). |

## 8. Definition of done for this plan

The plan is complete when: `core/result.hpp` no longer exists and no doc cites its
false premise; the pass-execution layer reports *why* it did not execute, with zero
`not_executed()` remaining and a gate preventing regression; the validity-bit
families are each converted or justified; all 11 declared ports have gateways (or
ROP-3.2 rules the shared type in); and every live citation of the Kleisli
vocabulary doc resolves — with the ROP alias present in law, teaching, and catalog,
and **no rule, type, or file renamed** by any of it.

**Status against that definition, measured 2026-09-18:**

| Clause | Status |
| :--- | :--- |
| `core/result.hpp` gone; no doc rests on its false premise | **MET** — ROP-1.1, ROP-1.2 |
| The pass layer says *why* it refused; zero `not_executed()`; a gate prevents regression | **MET** — ROP-2.1–2.4 |
| The validity-bit families are each converted or justified | **PARTIAL** — the `PassExecutionResult` member is *converted*; the 11 remaining sites are still unjustified (ROP-2.5) |
| All 11 declared ports carry gateways (or ROP-3.2 rules the shared type in) | **NOT MET** — 5 of 11 ports; ROP-3.2 is ruled (concept, no renames) but unimplemented, and ROP-3.1 is untouched |
| Every live citation of the Kleisli vocabulary doc resolves | **MET** — 10 broken sites repointed, and the class is now gated (ROP-4.1, ROP-4.3) |
| ROP alias present in law, teaching and catalog; **no rule, type, or file renamed** | **MET** — ROP-0; the sweep added no canonical term and renamed nothing |

So the plan is **not complete**: one clause is partial and one is unmet. Both
remaining pieces are additive (no rule rewrite, no rename), independent of each
other, and carry their DoDs in §4.

## 9. Landed ledger (2026-09-18 sweep)

**Verification, green at the end of the sweep:**

- `build/` CTest: **82/82 passed** — 75/75 baseline plus 7 registrations: the
  doc-path gate + twin, the pass-execution-shape gate + twin, the
  claimed-primitive gate + twin, and the re-activated pure-value classification
  gate.
- Run individually and green: `check_kdba_boundaries.sh`,
  `check_gateway_rails.sh`, `check_contract_placement.sh`,
  `check_include_graph.py`, `check_header_migration.py`, `inventory_headers.py`,
  `check_doc_paths.py`, `check_pass_execution_shape.sh`,
  `check_claimed_primitives.py`. (`check_backend_seam_symbols.sh` and
  `check_pure_value_libraries.sh` take required arguments — §6 notes this — and
  pass exactly as CTest invokes them.)
- `inventory_headers.py --write` ran in the same change and then verified clean;
  `header_count` **234 → 233**.
- **T2 held:** `git status --short` lists nothing under `docs/outdated/` or
  `docs/education/kdba_history/`.
- **Rule 15 held:** every touched header's content-hashed inventory was
  regenerated in the same change.
- **No rule rewritten, no canonical term renamed, no type or file renamed.**

**Change inventory** — 20 modified, 1 deleted, 7 added by this sweep (the plan
and its companion findings doc were added by the earlier planning session):

| Area | Files |
| :--- | :--- |
| ROP-1 | `include/shs/core/result.hpp` (**deleted**), `docs/backlog/cpp26_refactoring_candidates.md`, `docs/education/README.md` |
| ROP-2 | `renderpath/execution/render_pass.hpp`, `renderpath/execution/pass_adapters.hpp`, `renderpath/execution/pluggable_pipeline.hpp`, `renderpath/planning/pass_contract_registry.hpp`, `tests/core_tests.cpp` |
| ROP-4 | `renderpath/renderpath.gateway.hpp`, `logic/logic.gateway.hpp`, `render/frame/frame.gateway.hpp`, `input/input.gateway.hpp`, `app/session_orchestrator.gateway.hpp`, snake `snake_level_01.hpp`, tetris `event_ids.hpp` |
| New gates | `tools/check_doc_paths.py` + `check_doc_paths_negative_test.sh`; `tools/check_pass_execution_shape.sh` + `check_pass_execution_shape_negative_test.sh`; `tools/check_claimed_primitives.py` + `check_claimed_primitives_negative_test.sh`; `tools/unclaimed_core_primitives.json` |
| Wiring | `CMakeLists.txt` (7 registrations), `docs/backlog/engine_header_inventory.json` (regenerated) |

**One meta-finding worth keeping:** the doc-path gate failed on *this document*
mid-sweep, because §0/§5 quote the old broken spellings as evidence. That is the
gate working, and the response was a documented escape hatch rather than a
weakened rule — see ROP-4.3. It is recorded here because the same trap will
catch any future close-out note that quotes a defect.




