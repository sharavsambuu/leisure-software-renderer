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

## W-C — Joint retro → owner rulings (gates the traversal)

C2.4 of the adoption todo, folded with the proposal's deferred gates:

1. **Findings retro** — did the pilot annotations catch anything? Did any
   condition need a pure, allocation-free helper? Update proposal §4.
2. **Ruling: sweep beyond renderpath** (adoption todo C2.4 gate) — opens W-D.
3. **Ruling: P2** — expected-rail exclusivity at gateways (no `bool`/status
   codes/out-params/exceptions as failure rails); pilot pod `frame`.
4. **Ruling: P5** — closed-variant exhaustiveness (ban `default:` swallow in
   `*.gateway.hpp` command dispatch).

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
