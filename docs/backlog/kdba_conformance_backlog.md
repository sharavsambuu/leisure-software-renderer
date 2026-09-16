# KDBA Conformance Backlog — shs-renderer-lib

> **Reassessment required (2026-09-17).** This audit predates the [governing clarification](../spec/dod_ecs_architecture.md). Findings below are candidates, not newly approved migration work. K1.1–K1.5, K5.1, and K6.1 must be re-evaluated against actual contracts and callers: a `void` signature, identity gateway, or wrapper alone does not prove a defect. Do not invent vacuous errors or run signature-only ports. Boundary, lost-fact, determinism, and lifetime findings still require evidence and tests. Blocked P6.x and standing P4.4 work remain tracked; nothing is completed or dropped by this notice. This notice supersedes conflicting status and run-plan language below.

> Status: active (2026-09-16). Source: full audit of `include/shs/domains/` (11 pods) against Constitution II (Kleisli Domain Boundary Architecture) + the 2026-09-16 hardening amendments (gateway-gateway terminology, Rule 4.1 drain-order law, ERROR_FLOW failure-rail catalog + drift gate).
> Trigger: user directive — current pods are the "old gateway based Domain PODs architecture"; audit the lib against KDBA laws and register every violation candidate.
> Law precedence: Constitution II S2.1 + Rules 2/10/11/12, canon S6.1-S6.2, precedence S2.2. Roadmap is schedule, not law.
> Provenance: supersedes `domain_pod_hardening_backlog.md` (FROZEN same day, see its banner). Blocked P6.x items and standing laws roll forward here; nothing was silently dropped.
> Verification after every item: `build/` ctest suite green + `check_kdba_boundaries.sh` green (now with the ERROR_FLOW drift gate + final-enforcement exit guard).

## Naming migration (2026-09-17) — vocabulary harmonization, no behavior change

Executed under Constitution II **§6.6 Pod Identifier Law** (new; full model, post-mortem and
decision procedure: [`pod_identifier_law.md`](../spec/pod_identifier_law.md)). Scope: cosmetics
only — zero behavioral change; verified by `ctest` 16/16 green including `shs_renderer_boundary_check`.

| Was | Is |
| :--- | :--- |
| `*.reducer.hpp` (11 pods) | `*.gateway.hpp` |
| `*.action.hpp` (11 pods) | `*.command.hpp` |
| `reduce_camera` / `reduce_frame` / `reduce_geometry` / `reduce_gfx` / `reduce_input` / `reduce_lighting` / `reduce_resources` / `reduce_scene` / `reduce_sky` | `<pod>_gateway(...)` |
| `reduce_render_path` | `renderpath_gateway` |
| `reduce_fsm` | `logic_gateway` |
| `reduce_runtime_state` / `reduce_runtime_input_latch` | `runtime_state_gateway` / `input_latch_gateway` |
| `<Pod>Action` variant (`CameraAction`, `FsmAction`, `GfxAction`, …) | `<Pod>Command` |
| `RuntimeAction` / `RuntimeActionType` / `RuntimeActionPayload` | `RuntimeCommand` / `RuntimeCommandKind` / `RuntimeCommandPayload` |
| `MoveLocalAction` / `LookAction` / `ToggleFlagAction` | `MoveLocalIntent` / `LookIntent` / `ToggleFlagIntent` |
| `<Pod>ReduceInputs` (11 pods) | `<Pod>Context` (`dt` lives here, never a bare parameter) |
| `value_actions.hpp` | `value_commands.hpp` |
| `check_vop_boundaries.sh` | `check_kdba_boundaries.sh` |
| `shs_renderer_vop_*` targets + `tests/vop_*_tests.cpp` | `shs_renderer_*` + `tests/*_tests.cpp` |
| `[vop-boundary]` / `[vop-tests]` log prefixes | `[kdba-boundary]` / `[kdba-tests]` |

**Why `Command` for the variant and `*Intent` for the alternatives:** the `edge/` layer already
owns `LookCommand`, `MoveCommand`, `ToggleLightShaftsCommand`, `ToggleBotCommand`, `QuitCommand`
as `ICommand` subclasses (executable edge objects). Pod vocabulary therefore uses `*Intent`
payloads inside a `<Pod>Command` variant — exactly the shape `renderpath` and the demo pods
(`SessionCommand` + `*Intent`) already used. The first rename attempt produced a real
`shs::LookCommand` redefinition, which is the evidence for this rule.

**Gate work (same commit — mandatory):** `check_kdba_boundaries.sh` locates pods *by filename
glob*, so a rename that lands without the glob update makes the glob resolve an empty file set —
and `grep -r` then re-scopes to the working directory and enforces against the **wrong tree**.
Observed live: immediately after `*.reducer.hpp -> *.gateway.hpp`, the monolith tracker began
reporting switch sites in `execution/pipeline/` and `rhi/drivers/`. New gates added:
1. **Non-vacuity** — every pod must carry `<pod>.{contract,command,event,gateway}.hpp`, and each
   per-role glob must match a non-zero file count. This makes the silent mis-scope impossible.
2. **Paradigm-token ban** — `reduce_*`, `reducer`, `*Action`, `*.reducer.hpp`, `*.action.hpp`
   anywhere under `domains/` are hard FAILs, so the old vocabulary cannot reappear.
3. **Gateway presence** — every `<pod>.gateway.hpp` must expose a `<pod>_gateway` entry point.

**Effect on W1:** K1.1's "shared `Step`/gateway vocabulary" is now *named* — law §6.6 plus this
migration give the word `gateway` and the file suffix real, greppable existence, which is what
makes the 11 ports mechanical. Still open in K1.1: the `Step` **type** in code — the identifier
`Step` currently exists only in the demo pods (`SessionStep`, `MissionStep`, …), not in the lib.

**Deliberately not migrated:** namespace ownership (`shs::Fsm*` and the input vocabulary still sit
at root `shs`; §6.6 Rule N4 records it as a migration item, with `RuntimeState` correctly staying
root-level as a cross-pod aggregate); the `docs/outdated/` and `docs/education/kdba_history/`
archives (never rewritten); and the ~11 `exps-gpu-renderer` demos that include the long-dead
`shs/input/...` path (pre-existing breakage, not caused by this migration).

## Audit summary (2026-09-16)

Checked against KDBA laws: Kleisli house signature (`expected<Step{NextState, Events}, ClosedEnumError>`), switch-monolith ban (Rule 2 as amended — gateway = gateway, never `switch(action.type)`), zero-signal-loss, closed error vocabularies + failure-rail catalog, phantom/validity-flag ban, named-field events (no positional bools), one public gateway per pod, purity/edge laws.

**Clean:** banned tokens (shared_ptr/dynamic_cast/mutex/function: 0 non-edge hits), SDL/fopen in domains (0), ambient entropy (0), unordered_* in gateways (0), cross-domain includes (only legal directions: camera->geometry/aabb, renderpath->frame/technique_mode — both sanctioned seams), event/enum catalog drift (both gates green).

**Violations found:** every one of the 11 pod gateways uses the retired writer shape; one literal `switch(action.type)` monolith (input); one phantom validity flag; one positional-bool event; silent signal drops in logic; a dual-gateway pod; a discard-all stub gateway; namespace/Inputs drift in logic. Details per workstream below.

## W1 — Kleisli gateway migration (flagship, phased per pod)

- [x] **K1.1 Migration law note + per-pod port plan** — DONE 2026-09-17 (Run A): [`kdba_kleisli_migration_plan.md`](kdba_kleisli_migration_plan.md) publishes the shared `Step`/gateway vocabulary, the port order, and the batch-vs-per-command spike decision; renderpath landed with kit-extended tests (replay + empty-log + value-equality `operator==` on state/events) proving the signature swap is behavior-neutral. — was: publish the port order and the shared `Step`/gateway vocabulary before touching any gateway, so the 11 migrations are mechanical copies of one proven shape, not 12 ad-hoc designs. Pilot = renderpath (its `try_swap_plan` is already an arrow chain: `and_then`/`or_else` over `expected`, only the wrapper is `void` — smallest delta to full house shape). Port order: renderpath -> logic -> frame -> geometry -> lighting -> sky -> scene -> resources -> gfx -> input (input last: biggest monolith, needs W2 first).
- [x] **K1.2 renderpath** — DONE 2026-09-17 (Run A): `renderpath_gateway` now returns `RenderPathStep{commands_applied, noops_observed, swaps_rejected, plan_generation}` by value; events stay on the caller's arena (A.7 divergence honored); transition bodies moved to named per-intent `apply_*` arrows (K2.2 renderpath half); `try_swap_plan` returns its outcome over the unchanged per-command `expected` rail. **Reassessment verdict (banner):** the audit's literal "`expected` at the batch rim" was REFUTED — every real failure is a compile rejection absorbed by the per-command rail and materialized as `PATH_SWAP_REJECTED` (previous plan kept), so a batch-level error enum would be invented/vacuous (ERROR_FLOW non-vacuity law); evidence + decision in the plan doc. Unlocks the L1 leftover from the frozen backlog as stated.
- [ ] **K1.3 logic** — `logic_gateway` (logic.gateway.hpp:86): writer shape + `FsmInputs` empty + dt lives on `FsmTick` action instead of Inputs (L125-129) while input pod takes dt from Inputs — inconsistent time placement across pods. Port unifies: time in Inputs, gateway returns `expected<FsmStep, FsmError>`; silent `continue` drops become observable (K3.2). Namespace `shs` -> `shs::logic` (L23; every other pod is `shs::<pod>`, FsmDesc/FsmState/FsmEvent are the only domain types at root namespace).
- [ ] **K1.4 camera** — `camera_gateway` (camera.gateway.hpp:32-42) discards ALL arguments (`(void)state; (void)actions; ...`): a gateway that silently eats every command, the trivial worst-case zero-signal-loss violation. Decide: real camera gateway absorbing the camera math that currently lives in input's MoveLocal/Look handling (input.gateway.hpp:45-68 reaches directly into `state.camera` — cross-vocabulary coupling), or fold camera vocabulary into the input pod and delete the stub. DoD: no discard-all gateway in the tree.
- [ ] **K1.5 The 8 silent pods (frame, geometry, gfx, lighting, sky, scene, resources, camera-vocab)** — all are identity shells with the writer signature (evidence: grep `inline void reduce_` = 12 hits, all `void ... pmr::vector<X>&`). Mechanical port to the Kleisli shape; their vacuous error channel is `void`-error or a single `None`-style closed enum until real failure modes are designed (failure-rail law: an error enum with no real values is worse than none — ERROR_FLOW.md documents this convention). DoD: grep `inline void reduce_` in domains = 0; all still kit-green; ERROR_FLOW.md updated per drift gate.

## W2 — Kill the switch monoliths (Rule 2 as amended)

- [ ] **K2.1 input** — `input_gateway` (input.gateway.hpp:30-93) is the textbook forbidden monolith the amended Rule 2 names: literal `switch (action.type)` with 5 arms + `std::get_if` payload fishing inside each arm (type+payload double dispatch, desync-prone). Decompose into per-intent gateway functions (`apply_move_local`, `apply_look`, ...) assembled by the pod's Kleisli gateway over the closed `RuntimeAction` variant (`std::visit` + `if constexpr`, renderpath style). DoD: `switch` on action discriminators in `*.gateway.hpp` = 0 (add the grep to K6.2's gate when the last pod lands).
- [ ] **K2.2 logic + renderpath dispatch audit** — both use variant ladders living inline in the public gateway (`std::get_if` chains in logic.gateway.hpp:96-133; `std::visit` + `if constexpr` in renderpath). Legal under Rule 2's letter (dispatch on the closed variant, not a parallel enum), but the transition logic belongs in named per-intent functions; the gateway is only the assembly point (amended §2.1 definition). DoD: each public gateway reads as the arrow-chain assembly, no inline transition bodies.

## W3 — Error-channel + zero-signal-loss conformance

- [ ] **K3.1 Failure-rail inventory per pod** — walk all 11 pods: for each transition, classify infallible / fallible-with-closed-error / currently-silent. Output: per-pod error enums (named `*Error`/`*Rejection`/`*Reason` per ERROR_FLOW.md law) + ERROR_FLOW.md rows. Seed data: renderpath already has the only real rail (`PathSwapRejectionReason`, 6 values, mapped 1:1 from the compiler enum at renderpath.gateway.hpp:55-67 — this stays the model). DoD: ERROR_FLOW.md covers every pod's error vocabulary; drift gate green.
- [ ] **K3.2 Kill silent signal drops in logic** — `continue` sites consume a command and emit NOTHING: `!state.started` on signal (logic.gateway.hpp:111), no-rule-match on signal (L113) and on tick (L131). Under zero-signal-loss these are invisible failures. Emit `FsmSignalRejected`/`FsmTickNoRule` facts or route through the gateway error channel — pick ONE house answer and mirror it in renderpath's silent no-op sites (renderpath.gateway.hpp:155, 166, 175: same-technique/same-mode commands return silently; document the decision as a constitution note, not folklore). **Renderpath half DONE 2026-09-17 (Run A):** the house answer is **FACTS** — the three silent no-op sites now emit `TechniqueUnchangedEvent` / `ViewCullingUnchangedEvent` / `ShadowCullingUnchangedEvent` (a no-op is not a failure; it never touches the error rail), pinned by `test_unchanged_facts`; the logic half (`FsmSignalRejected`/`FsmTickNoRule`) remains for Run B, copying this answer.
- [ ] **K3.3 Compensator sweep (Rule 12)** — only the saga spike test exercises compensation today. Once W1 lands, audit each multi-step transition for invertibility: every state mutation inside a gateway must have a recorded compensator fact or be provably idempotent. DoD: per-pod compensator note in ERROR_FLOW.md or an explicit "no multi-step flows" entry.

## W4 — State-shape hardening

- [x] **K4.1 `has_plan` phantom flag** — DONE 2026-09-17 (Run A): `RenderPathPodState::plan_generation` (`uint32_t`, 0 = none, bumped on every successful plan install; rejections and runtime toggles never bump it) replaces the validity bit; semantics pinned by `test_plan_generation_semantics`; `grep has_plan` in shs-renderer-lib = 0. Feeds P6.1's plan-hash executor rebuilds (precondition contributor). — was: `RenderPathPodState::has_plan` (renderpath.gateway.hpp:35) is a validity bit shadowing plan existence: state can lie (default plan + has_plan=false is indistinguishable from an empty-but-real plan). Replace with a generation counter (`uint32_t plan_generation = 0`, 0 = none) — value-honest, replay-friendly.
- [x] **K4.2 Positional-bool event** — DONE 2026-09-17 (Run A): split into `ViewCullingModeChangedEvent{previous, current}` / `ShadowCullingModeChangedEvent{previous, current}`; closed vocabulary stays closed (5 → 9 facts); EVENT_FLOW.md + `renderpath_event_name` table updated together; drift gates green. DoD met: zero unnamed-bool payload fields in renderpath events.

## W5 — Gateway uniqueness + legacy seams

- [ ] **K5.1 input dual-gateway retirement** — input pod exposes BOTH `input_gateway` (house signature) and `runtime_state_gateway` (legacy by-value signature, value_commands.hpp:35-43, consumed by edge/command_processor.hpp:53). One pod, one public Kleisli gateway. Port the edge consumer, delete the legacy wrapper. DoD: value_commands.hpp gone or applied to a deprecation banner; single grep-visible gateway per pod.
- [ ] **K5.2 Grandfathered-architecture sweep** — parked items whose unlock conditions this directive supersedes: legacy callback StateMachine beside the value FSM (logic — "zero consumers" per P3.10; verify then delete), PluggablePipeline/FrameGraph retention re-check (P5.2 kept them pending P6.1; re-audit if W1 changes the calculus), AssetRegistry-class stale forks (deleted in P5.1; confirm none regrew).

## W6 — Mechanical drift gates (checker)

- [x] **K6.1 Kleisli-shape gate** — DONE 2026-09-17 (Run A): gate landed in `check_kdba_boundaries.sh` §(4). **Reassessment verdict (banner):** the original wording — gate `inline void reduce_*` — was obsolete (§6.6 already bans those tokens outright), so the gate targets the real regrowth vector, the writer SIGNATURE: FAIL on `void <pod>_gateway(` in a Kleisli-migrated pod, FAIL on any pod missing from the migrated/grandfathered registers (migrated: renderpath; grandfathered: the 10 Run B/C pods — P1.1 facade-case mechanism). — was: once the FIRST pod lands its gateway (K1.2), FAIL on any NEW `inline void reduce_*` added outside the migrated-pod list (grandfather list carried in the script, shs/pipeline facade-case precedent — same mechanism as P1.1). Prevents writer-shape regrowth during the phased migration.
- [ ] **K6.2 switch-monolith gate** — FAIL on `switch` over action discriminators in `*.gateway.hpp`; lands with W2 completion (audit grep already proven).
- [ ] **K6.3 Silent-drop gate** — grep-gate the logic silent-drop pattern once K3.2 fixes it — ONLY if the team picks "events" over "error channel"; a gate on an undecided design choice is premature. Standing until K3.2 decides.

## Rolled forward from the frozen backlog (tracked, not dropped)

- [ ] **P6.1 PATH_COMPILED-driven executor rebuilds** — BLOCKED (needs a live demo host; unchanged).
- [ ] **P6.2 Replay harness (cross-session codec + CI replay)** — BLOCKED on demo host; K1.5 strengthens the pod-level replay story (Step values are serializable by construction).
- [ ] **P6.3 Rollback snapshots + time-travel overlay** — BLOCKED on windowed host; K4.1's generation counter is a precondition contributor.
- [ ] **P4.4 Seeded determinism contract (STANDING)** — still no stochastic pods; first stochastic pod authors the contract.

## Consolidated run plan (2026-09-17) — 3 runs

> User directive: regroup the migrations into as few runs as possible. Only the
> grouping changed — every DoD, constraint, and the 2026-09-17 reassessment
> banner are carried forward unchanged. Each reassessment-flagged item
> (K1.1–K1.5, K5.1, K6.1) is verified against real contracts and callers
> *inside its run*, before its port lands. Nothing is completed or dropped by
> this regrouping; the 5-run plan below is kept as provenance.
> Verification after every run: `build_vcpkg/` ctest 16/16 green +
> `check_kdba_boundaries.sh` green.

- **Run A — Pilot + state shape (renderpath only).**
  Preflight: reconfigure the stale `build/` tree (still registers the retired
  `shs_renderer_vop_*` target names; boundary check "Not Run"; P0.1 déjà vu) so
  both trees report 16/16 before any port is judged.
  Reassess K1.1/K1.2 (+ the K5.1/K6.1 touchpoints renderpath can see).
  Land, in one renderpath-only touch: **K1.1** (plan doc + shared `Step`
  vocabulary + the batch-vs-per-command spike decision, recorded as a
  constitution note), **K1.2** (renderpath port to the Kleisli house shape),
  **K4.1** (`has_plan` → `plan_generation` counter), **K4.2** (culling event
  split into named view/shadow events), **K3.2's renderpath half** (silent
  same-technique/same-mode no-op sites get the ONE house answer — the decision
  Run B's logic emission must copy), **K6.1** (Kleisli-shape gate activated;
  grandfather list seeded by the K1.2 exemplar).
  DoD: renderpath is the single conforming Kleisli pod; gate live; both test
  trees green.
  **Status: DONE 2026-09-17** — port + K4.1/K4.2 + K3.2 house answer + K6.1
  gate + K1.1 plan doc landed; verdicts in
  [`kdba_kleisli_migration_plan.md`](kdba_kleisli_migration_plan.md).

- **Run B — Behavioral pods (logic + input + camera).**
  Reassess K1.3/K1.4/K2.1/K5.1 against real callers first.
  Land as two touches: **logic touch** — K1.3 (port to Kleisli shape, dt into
  `<Pod>Context`, `shs::logic` namespace) + K3.2's logic half (silent
  `continue` sites emit `FsmSignalRejected`/`FsmTickNoRule` facts — house
  answer decided in Run A) + K2.2's logic half (inline `std::get_if` ladders →
  named per-intent arrows; gateway = assembly point only).
  **input touch** — K2.1 (switch monolith → `std::visit` + `if constexpr` over
  the closed variant, per-intent `apply_*` arrows) + K1.4 (camera
  absorb-or-fold decision: real camera gateway absorbing the MoveLocal/Look
  camera math, or fold camera vocabulary into input and delete the stub —
  no discard-all gateway survives either way) + K5.1 (dual-gateway retirement:
  port the edge consumer in `command_processor.hpp`, delete the
  `runtime_state_gateway` wrapper, retire `value_commands.hpp`).
  DoD: zero switch/action-discriminator dispatch in `*.gateway.hpp`; one
  public gateway per pod; `value_commands.hpp` gone or banner-deprecated.

- **Run C — Sweep + harden (mechanical + docs + final gates).**
  Land: **K1.5** (the 8 silent pods — mechanical copies of the Run A shape,
  vacuous-channel error convention per ERROR_FLOW.md), **K3.1** (failure-rail
  inventory across all 11 pods → per-pod error enums + ERROR_FLOW.md rows;
  renderpath's `PathSwapRejectionReason` stays the model), **K3.3**
  (compensator sweep — per-pod Rule 12 notes or explicit "no multi-step
  flows"), **K5.2** (grandfathered re-audits: callback FSM verify-then-delete,
  PluggablePipeline/FrameGraph re-check, AssetRegistry-fork check), **K6.2**
  (switch-monolith gate — activates when the last gateway dispatch is ported),
  **K6.3** (silent-drop gate — unblocked by the Run A/B house-answer
  decision; ONLY if the team picked "events" over "error channel").
  DoD: `grep 'void <pod>_gateway('` writer shape = 0 in `domains/`; ERROR_FLOW
  covers every pod's error vocabulary; all drift gates green; every item in
  W1–W6 ticked or explicitly closed.

Constraints (carried from the 5-run plan, regrouped): K1.2 lands before K6.1
activates (internal Run A order); K2.1 before any input Kleisli port (same
touch, Run B); the K3.2 house answer is decided in Run A so Run B copies it
instead of deciding; K6.3 waits on the K3.2 decision (Run C). Each run is one
independently committable unit with its own green ctest + linter evidence —
no big-bang 12-pod migration.

## Run plan (SUPERSEDED 2026-09-17 by the consolidated run plan above; kept for provenance)

- **Run 1:** K1.1 + K1.2 (renderpath pilot) + K4.1 + K4.2 — one pod proves the whole shape (gateway, generation counter, event split) and K6.1's gate gets its grandfather list.
- **Run 2:** K2.1 + K1.4 (input decomposition + camera decision) + K5.1 — the monolith dies and the dual-gateway retires in one touch; input's camera reach-in gets rehomed here.
- **Run 3:** K1.3 + K3.2 (logic port + the silent-drop law decision).
- **Run 4:** K1.5 sweep (8 silent pods, mechanical) + K3.1 inventory + K6.2.
- **Run 4 close-out:** K3.3 + K5.2 re-audits + K6.3 decision.
- Constraint: K1.2 lands before K6.1's gate activates (the gate needs its first conforming exemplar). K1.5 parallelizes across pods once K1.2 pins the pattern.

## Non-goals

- No demo/adventure restarts (unchanged from the frozen backlog; P6.x stay blocked until a host exists).
- No big-bang 12-pod migration in one commit — per-pod, kit-tested, independently committable.
- No error enums invented for pods with no real failure modes (K1.5 uses the documented vacuous-channel convention instead).
- No per-frame pod reductions — batch planners/edges own hot loops (S7.1, unchanged).