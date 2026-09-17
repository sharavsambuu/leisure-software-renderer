# Constitution Enforcement Plan — Contract Guardrails + Kleisli Rails across every DVO

> Status: **active (2026-09-17)** — owner directive. Governs the post-migration
> phase: the domain-separation migration is complete (record archived at
> [`docs/outdated/engine_domain_separation_migration.md`](../outdated/engine_domain_separation_migration.md)),
> the Kleisli gateway ports are landed (Runs A–C closed in
> [`kdba_conformance_backlog.md`](kdba_conformance_backlog.md)), and the old
> pure-reducer pod era is retired. What remains is **enforcement**: make
> Rule 17 (contract guardrails) and §8 (Kleisli doctrine) mechanically verified
> on every pod, and sweep the last old-way sites. This plan is schedule, not
> law; precedence: Constitution II §2.2.

> Verification after every item: full build + CTest green +
> `check_kdba_boundaries.sh` green + `check_include_graph.py` green; a touched
> header regenerates the inventory in the same commit (`inventory_headers.py
> --write`, Rule 15).

## Verified starting state (2026-09-17 audit)

- `reduce_*` tokens in `include/`: **0**; `void *_gateway(` signatures: **0** —
  the reducer-era vocabulary is gone and the K6.1/K6.2 gates hold against
  regrowth.
- Remaining old-way sites: `renderpath.gateway.hpp` (3 closed-enum `switch`
  dispatches: `technique`/`reason`/`toggle`) — INFO-tracked by the kdba gate;
  decompose into named arrows where they guard transitions.
- `SHS_PRE`/`SHS_POST`/`SHS_CONTRACT_ASSERT` uses in `include/`: **0** — the
  bridge (C1) is done and tested, but no library header is annotated yet.
- Cold registries (`resources`, `gfx`) still use node containers — INFO-tracked
  by the kdba gate; migrate to flat open-addressing maps.
- Gates already live: kdba-boundary (pod purity, direction law, Kleisli shape),
  include-graph, header-migration, inventory staleness.

## W-A — Ratify + land P1: guardrail placement by file role — **DONE 2026-09-17**

Source: `dvo_semantics_enforcement_proposal.md` P1 (ratified — owner go-ahead
2026-09-17). Landed:

- [x] `SHS_CONTRACT_ASSERT` = value invariants: only `*.contract.hpp` /
  `*.command.hpp` / `*.event.hpp` and pure leaf value headers
  (checker allowlist; first entry: `renderpath/planning/render_path_compiler.hpp`,
  pure-tier compiler — hosts the C2.2 transition-table asserts).
- [x] `SHS_PRE` / `SHS_POST` = edge law: only `*.gateway.hpp` / `*.contract.hpp`.
- [x] Checker gate 8: `tools/check_contract_placement.sh`, wired as gate 8 into
  `check_kdba_boundaries.sh` (bridge definition header exempted — the law
  governs use sites, not the macro definitions). Green across 231 headers.
- [x] Negative fixtures (CTest `shs_renderer_contract_placement_negative_test`):
  `SHS_CONTRACT_ASSERT` in a math leaf FAILs; `SHS_PRE` in a non-seam header
  FAILs; the same macros in their legal homes PASS.
- [x] Constitution II Rule 17 placement sentence
  (`docs/spec/value_oriented_programming.md` §2.3 item 2) + Conventions §11
  restatement — one commit.

## W-B — C2 renderpath pilot: annotate the seam, railway-conformant

C2.1–C2.3 of the adoption todo, made P1-legal by W-A:

- [x] **C2.1 gateway batch-rim postconditions** — DONE 2026-09-17: `SHS_POST`
  on the committed plan in `try_swap_plan` (`renderpath.gateway.hpp`): pass
  chain non-empty; every pass entry references a registered technique (pure
  helper `renderpath_plan_pass_chain_registered`). Negative test: enforced
  twin installs the capture handler, commits a hand-broken empty-chain plan
  (compatibility rules relaxed), asserts kind/expression capture.
- [x] **C2.2 transition-table asserts** — DONE 2026-09-17, P1-legal: the
  technique-mode table moved to its pure-leaf home (`render_path_recipe.hpp`)
  and the compiler asserts the plan/recipe table row
  (`SHS_CONTRACT_ASSERT` in `render_path_compiler.hpp`, gate-8 allowlisted).
  The `reason`/`toggle` legs stay INFO-tracked dispatches; the W-C retro rules
  whether they decompose into named arrows or move to contract headers.
- [x] **C2.3 wait-free span preconditions** — DONE 2026-09-17: `SHS_PRE` at
  the `renderpath_gateway` rim (commands span vs events buffer never alias).
  Recorded deviation: the sizes-equal half of Rule 7.1 has no dst/src job
  entry in this pod yet — the C2.4 retro places it when one lands.
- CMake keeps both twins compiling: enforcement twin runs the checks; release
  twin proves the `[[assume]]` fold emits no check.

## W-C — Joint retro → owner rulings (gates the traversal) — **DONE 2026-09-17**

C2.4 of the adoption todo, folded with the proposal's deferred gates. All
four items closed in the joint retro (owner granted all rulings):

1. [x] **Findings retro** — DONE 2026-09-17: pilot caught a real tiering
   violation (transition table moved to its pure-leaf home), one masked
   negative-test hand-break (isolated), one pure helper needed
   (`renderpath_plan_pass_chain_registered`). Verdict recorded in the
   adoption proposal §4.
2. [x] **Ruling: sweep beyond renderpath** — GRANTED: W-D opens, port order
   unchanged (logic first).
3. [x] **Ruling: P2 expected-rail exclusivity** — ENACTED as blocking:
   `tools/check_gateway_rails.sh` (boundary gate 9) bans `throw` and
   bool/integral status rails on `*_gateway` rims. Pilot pod `frame` already
   conformant (empty §6.1 vocabulary, Step-valued rim); all 4 gateways green.
4. [x] **Ruling: P5** — ENACTED: `default:` swallow banned in
   `*.gateway.hpp` dispatch (comments stripped before the scan) and every
   `std::visit` must carry a trailing `static_assert` exhaustiveness tail —
   the audit found the real gap (if-constexpr chains with no tail silently
   ignore new alternatives); tails added to the renderpath + logic dispatches.
   Negative fixture: `shs_renderer_gateway_rails_negative_test` (CTest).


## W-D — The per-pod traversal (one slice per pod, one commit each)

Established port order: **logic → frame → geometry → lighting → sky → scene →
resources → gfx → input** (input last — biggest vocabulary). Per pod:

1. **Railway conformance audit** — gateway is assembly only; transition bodies
   in named `apply_*` arrows; error family present in `docs/pods/ERROR_FLOW.md`
   (drift gate); zero-signal-loss; no phantom flags; no `default:` swallow.
2. **P1-legal guardrail annotations** on the pod's invariants + edges, with a
   negative test.
3. **Old-way mop-up** — residual switch dispatches that guard transitions
   decompose into named arrows (renderpath's three sites are the pattern
   source per the W-C ruling).
4. **Gates**: build + full CTest + kdba-boundary + include-graph; inventory
   same-commit if a header changed.

### Slice 1: logic pod — **DONE 2026-09-17**

- [x] **Railway audit**: gateway is assembly-only (arrows in `fsm_detail`);
  `FsmStart`/`FsmSignal`/`FsmForce`/`FsmTick` dispatch exhaustively (P5 tail
  landed with W-C); zero-signal-loss is materialized facts (K3.2 house
  answer); error/rejection family present in `ERROR_FLOW.md` (drift gate
  green); no `continue;` swallow; rims Step-valued (P2 green).
- [x] **Guardrail annotations**: rim PRE — transition-table integrity
  (`desc.rules_reference_states()`, new pure validator on `FsmDesc`: every
  rule references registered states on both ends — `select_rule` trusts the
  table, so a broken row was previously discoverable only after the commit);
  rim POSTs — the FSM never rests in an unregistered state (defense in depth
  for caller-owned pre-started states) + zero-signal-loss accounting
  (applied+facts+rejected == commands.size()). Deviation recorded: no
  wait-free dst/src span pair in this pod, so Rule 7.1 has no entry here
  (same deviation family as C2.3's sizes-equal leg — lesson 9.8: annotate
  what exists).
- [x] **Negative tests**: `shs_renderer_logic_guardrail_tests` (enforced:
  hand-broken table row → kind=pre; unregistered resting state + empty batch
  → kind=post; valid batch silent) + release twin (assume path).
- [x] **Gates**: 39/39 CTest, boundary (incl. gates 8+9), include-graph,
  inventory same-commit.

### Slice 2: frame pod — **DONE 2026-09-17**

- [x] **Railway audit**: the pinned identity transition (K1.5). Gateway is
  assembly-only and the command vocabulary is EMPTY by law (§6.1) — planners
  rebuild `FrameParams` per frame instead of reducing it, so there are no
  transition bodies to audit. Dispatch exhaustiveness is stronger than the P5
  tail: the visit body's `static_assert(is_same_v<T, monostate>)` makes any
  new alternative ill-formed outright; `frame.event.hpp` pins the empty event
  vocabulary with a `static_assert` + count. Rim is Step-valued
  (`FrameStep`, no bool rail, no throw — P2 green, gate 9). Error family
  documented in `ERROR_FLOW.md` ("Infallible monostate identity; no mutations
  or multi-step flows") — drift gate green. No phantom flags, no swallow.
- [x] **Guardrail annotations**: one true runtime rim invariant existed —
  zero-signal-loss for the identity pod is
  `commands_observed == commands.size()` (applied = facts = rejected = 0);
  rim `SHS_POST` added. The negative leg is not reachable through the public
  seam (no input can carry a non-monostate command — the variant is closed),
  so the compile-time pins are the negative story: the twins compile against
  the real header, so vocabulary drift breaks the build. Lesson 9.8 applied:
  nothing else exists to annotate (no Rule 7.1 dst/src pair, no
  caller-owned state invariants — `FrameParams` is pure config the gateway
  ignores by identity law).
- [x] **Tests**: `shs_renderer_frame_guardrail_tests` + release twin —
  silence + completeness proof (batch of 3 monostate commands fully
  observed, no events, handler untouched in enforced build / assume path in
  release).
- [x] **Gates**: 41/41 CTest, boundary (incl. gates 8+9), include-graph,
  inventory same-commit (Rule 15).

### Slices 3–8: geometry, lighting, sky, scene, resources, gfx — **DONE 2026-09-17** (batched, owner instruction)

These six pods share one shape, so their audits were batched into one commit
(deviation from one-commit-per-pod, by owner instruction: the pods are
provably identical — empty closed vocabularies, no gateway seam):

- [x] **Railway audit (each pod)**: `XCommand`/`XEvent` are
  `std::variant<std::monostate>` — explicitly empty closed vocabularies
  (Constitution §6.1); NO gateway exists yet, so there is no rim to audit or
  annotate. Real intents arrive with the R5 edge migrations (each command
  header names its migration step). Error family documented in
  `ERROR_FLOW.md` ("Silent error rails" — monostate identity, no invented
  `None` enum) — drift gate green. Registry/IO failures live outside these
  gateways per the Run C audit table.
- [x] **Hardening (the one real gap found)**: unlike `frame.event.hpp`, none
  of the 12 command/event headers pinned their emptiness compile-time.
  `static_assert(std::variant_size_v<X> == 1, ...)` pins added to all 12 —
  the empty vocabulary is now provable, and vocabulary drift fails the build
  (lesson 9.8: nothing else exists to annotate — no rim, no Rule 7.1 pair,
  no state invariants).
- [x] **Tests**: none added — no runtime seam exists; the pins are verified
  by the build itself (every test target compiles these headers).
- [x] **Gates**: full CTest, boundary (incl. gates 8+9), include-graph,
  inventory same-commit (Rule 15).


## W-E — Standing / parallel throughout

- **Cold-registry container migration** (resources, gfx → flat maps);
  shared lib utilities in `shs/containers/`, no private copies (vop §7 rule 6).
- **C4.2 toolchain tracking** (GCC 16/17 + Clang contract support table);
  C4.3/P4 native switch runbook stands until a toolchain ruling — the
  replay-parity CTest is the release blocker when it fires.
- **P3 re-scope bookkeeping**: the 2026-09-17 forwarder-tree removal already
  moved every checker scan root off `domains/` onto the canonical owner tree;
  P3's prefix hazard is resolved by that amendment (manifest-derived roots are
  moot — the migration manifest is empty by design after the retirement).
  Record as done-by-amendment next time the proposal is touched.

## Sequencing

```
W-A (ruling + gate) -> W-B (pilot annotations) -> W-C (retro + rulings) -> W-D (traversal, pod order above)
                                                        W-E runs in parallel
```

- W-A before W-B: placement law defines where pilot annotations may live.
- W-C before W-D: the adoption todo (C2.4) and the proposal (P2/P5 sequencing)
  both gate the sweep on the retro ruling.
- One commit per item; shrink-only; `docs/outdated/`, FROZEN backlogs and
  `kdba_history/` are never edited by any item here.

## Housekeeping: where the finished plans went (2026-09-17)

Completed migration-era backlogs moved to `docs/outdated/` per the archive
retention policy (rows added to its README, banners added, references updated):

| File | Moved from | Disposition |
| :--- | :--- | :--- |
| `engine_domain_separation_migration.md` | `docs/backlog/` | All 7 phases COMPLETE (incl. forwarder removal `c9928d2`); close-out table preserved |
| `kdba_kleisli_migration_plan.md` | `docs/backlog/` | Run A–C closed; Step/gateway vocabulary is law in Constitution II §8 |
| `domain_pod_hardening_backlog.md` | `docs/backlog/` | FROZEN, superseded by `kdba_conformance_backlog.md`; open items rolled forward there |

Still-active planning: this file,
[`contract_guardrails_adoption_todo.md`](contract_guardrails_adoption_todo.md),
[`dvo_semantics_enforcement_proposal.md`](dvo_semantics_enforcement_proposal.md),
[`kdba_conformance_backlog.md`](kdba_conformance_backlog.md),
[`namespace_cutover_mapping.md`](namespace_cutover_mapping.md) (0.2.0 schedule),
[`adventure_demo_conformance_backlog.md`](adventure_demo_conformance_backlog.md),
[`optimization_backlog.md`](optimization_backlog.md).
