# KDBA Kleisli Migration Plan — K1.1 (Run A pilot: renderpath)

> Status: active plan (2026-09-17, Run A of the consolidated run plan in
> `docs/backlog/kdba_conformance_backlog.md`). This is the K1.1 DoD artifact:
> the shared `Step`/gateway vocabulary and the per-pod port order, published
> before the remaining ports land, so they are mechanical copies of one proven
> shape. Law precedence: Constitution II §2.2 — where this plan and a numbered
> rule disagree, the rule wins.

## Reassessment verdicts (2026-09-17 banner compliance)

The backlog banner required K1.1–K1.5/K5.1/K6.1 findings to be re-evaluated
against real contracts and callers before any port. Renderpath verdicts:

- **K1.2 (renderpath) — CONFIRMED, ported.** `renderpath_gateway` was the
  retired writer shape (`void` + `pmr::vector<Event>&` out-param). The only
  live caller is `tests/renderpath_tests.cpp` (demo host parked), so the port
  breaks nothing.
- **Batch rim fallibility — REFUTED.** The audit's literal text ("returns
  `expected<RenderPathStep, RenderPathError>`") would require an invented
  batch-level error enum. Every real renderpath failure is a compile
  rejection, already absorbed by the per-command `expected` rail
  (`try_swap_plan`) and materialized as the `PATH_SWAP_REJECTED` fact with
  the previous plan kept. ERROR_FLOW's non-vacuity law ("an error enum with
  no real values is worse than none") bans the invented enum. The batch rim
  is therefore an honest **plain value**: `RenderPathStep`.
- **K1.1 (plan doc) — DONE** by this file. **K6.1 — gate landed**
  (Kleisli-shape gate in `check_kdba_boundaries.sh`; the original
  "gate `inline void reduce_*`" wording was obsolete after §6.6, so the gate
  targets the writer *signature* regrowth vector instead).

## The shared Step/gateway vocabulary (Constitution II house shape)

- **Pod state** — plain value, `operator==` defaulted (kit-provable): e.g.
  `RenderPathPodState{recipe, plan, plan_generation}`.
- **Step** — the batch outcome summary returned by value:
  `RenderPathStep{commands_applied, noops_observed, swaps_rejected,
  plan_generation}`. Countable, serializable by construction (feeds the P6.2
  replay story), never bundles the event log (A.7 divergence: events stay on
  the caller's PMR arena — the Writer log must survive both rails).
- **Gateway** — the single public entry point per pod: assembly only
  (`std::visit` + `if constexpr` over the closed command variant); transition
  bodies live in named per-intent arrows (`apply_*`, Rule 2 as amended).
- **Per-command fallibility** — `expected<T, ClosedEnumError>` inside the
  arrows; events emitted only in `transform`/`or_else` continuations (§8);
  failure keeps state + materializes a rejection fact.
- **Zero-signal-loss (K3.2 house answer, decided here, copied by Run B)**:
  a consumed command **never emits nothing** — accepted transitions emit
  their change fact, same-value commands emit `*Unchanged` facts (a no-op is
  not a failure; it never touches the error rail), rejected swaps emit the
  rejection fact. Renderpath's three silent no-op sites are gone.

## Spike decision: batch vs per-command (K1.2 A.7 note)

**Decision: batch-span gateway returning one `Step` summary; per-command
`expected` stays inside the arrows.** Evidence: renderpath commands are
low-frequency (menu/edge-driven), events per command ≤ 2, and per-command
`expected` at the public rim would force `vector<expected>` accumulation —
the §7.1 granularity-law smell. Chunk-level monad at the batch, flat flow
inside. Revisit only if a per-command consumer needs individual Step values;
the kit's replay machinery already covers cross-session needs via the event
log.

## Run B verdicts (2026-09-17, behavioral pods)

- **K1.3 logic — CONFIRMED, ported.** Writer shape + root namespace + silent
  drops all real; only caller is `logic_tests.cpp`. Full `Fsm*` vocabulary
  moved root `shs` → `shs::logic` (safe — zero external consumers). dt moved
  from `FsmTick` to `LogicContext` per the approved consolidated plan (one
  dt per batch, matching input; the tick is a pure advance marker).
  Zero-signal-loss: the audit's three silent sites PLUS a fourth the audit
  missed (force-while-unstarted emitted nothing) now emit facts;
  `FsmSignalRejected` split from no-rule into one-fact-per-type
  (`FsmSignalRejected` / `FsmSignalNoRule`) per the K4.2 lesson.
- **K2.1 input — CONFIRMED, decomposed.** The `RuntimeCommand{kind, payload}`
  envelope was the root cause of the switch monolith (double dispatch). The
  vocabulary is now a PURE closed variant
  (`variant<MoveLocalIntent, LookIntent, ToggleLightShaftsIntent,
  ToggleBotIntent, QuitIntent>`) — the kind enum had zero consumers outside
  the pod, so the envelope died with its switch. Gateway = `std::visit` +
  `if constexpr` over named `apply_*` arrows, returning `InputStep`.
- **K5.1 — CONFIRMED, retired.** `runtime_state_gateway` deleted; its lib
  consumers (`edge/command_processor.hpp`, `core_tests.cpp`) ported to the
  single public gateway; `value_commands.hpp` remains as the pure command
  emitter header with the retirement banner.
- **K1.4 camera — REFUTED as a violation; decision: neither absorb nor fold.**
  The stub's command/event vocabularies are `variant<monostate>` — no real
  command can ever arrive, so nothing is silently dropped; the identity
  gateway is §6.1-legal (frame-pod precedent) and its identity is pinned by
  kit tests. Its CONTRACT seam is live code (`CameraRig` builders feed
  `input_state.hpp` + `execution/app/camera_sync.hpp`), so fold-delete would
  break real consumers. The genuine coupling — camera math living in input's
  `apply_move_local`/`apply_look` over the rig inside input's own
  `RuntimeState` aggregate — is *intra-pod* (RuntimeState IS input state),
  not cross-pod mutation. Full absorption (camera pod owns rig state, input
  emits facts, edge routes them) needs the orchestrator/host that P6.1–P6.3
  also wait for. Run C's K1.5 sweep ports the identity shape mechanically.

## Run B semantics notes

- Logic `FsmTick` no longer carries dt (context-owned); a batch of N ticks
  advances N × context.dt.
- Logic same-state force/signal rules remain a documented silent no-op
  (legacy mirror, stated in the gateway header) — the one deliberate
  exception to zero-signal-loss, pending a Run C K3.3 decision.
- Input `RuntimeCommand` equality semantics unchanged (variant of ==-able
  intents); factory names/signatures unchanged, so edge ICommand subclasses
  needed no edits.


## Port order (Run B/C, from the consolidated plan)

logic (K1.3, copies the K3.2 unchanged-fact answer) → frame → geometry →
lighting → sky → scene → resources → gfx (K1.5 sweep, mechanical) → input
last (K2.1 monolith decompose + K1.4 camera decision + K5.1 dual-gateway
retirement, one touch). Each pod lands with kit-extended tests (replay +
empty-log) proving the signature swap is behavior-neutral; grandfather list
carried in `check_kdba_boundaries.sh`.

## Semantics preserved (behavior-neutral notes)

- `TechniqueSwitchedEvent` / `*CullingModeChangedEvent` are still emitted
  after an attempted swap **even when the swap was rejected** — pre-existing
  semantics, unchanged by Run A (the rejection fact records the outcome).
  A future Run C (K3.3) may revisit this as a double-fact question.
- `plan_generation` counts *successful installs* (0 = none, 1 after the
  first, +1 per swap); rejections and runtime toggles do not bump it.
