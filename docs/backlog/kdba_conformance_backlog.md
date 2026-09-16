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

- [ ] **K1.1 Migration law note + per-pod port plan** — publish the port order and the shared `Step`/gateway vocabulary before touching any gateway, so the 11 migrations are mechanical copies of one proven shape, not 12 ad-hoc designs. Pilot = renderpath (its `try_swap_plan` is already an arrow chain: `and_then`/`or_else` over `expected`, only the wrapper is `void` — smallest delta to full house shape). Port order: renderpath -> logic -> frame -> geometry -> lighting -> sky -> scene -> resources -> gfx -> input (input last: biggest monolith, needs W2 first). DoD: plan doc exists; each pod lands with kit-extended tests (replay + empty-log) proving signature swap is behavior-neutral.
- [ ] **K1.2 renderpath** — `renderpath_gateway` (renderpath.gateway.hpp:137) returns `void`, events via `pmr::vector&` out-param; inner `try_swap_plan` (L108-131) already computes `expected` but discards it into `or_else` event pushes. Port: public gateway returns `expected<RenderPathStep, RenderPathError>` per command (or per batch, per the A.7 bundled-signature divergence note — replay/event-count spike decides batch-vs-single; record the decision). `RenderPathPodState::has_plan` dies here too (K4.1). Evidence: renderpath.gateway.hpp:137-192, 108-131. Unlocks the L1 leftover from the frozen backlog (its "do not churn" unlock condition is hereby superseded by this backlog).
- [ ] **K1.3 logic** — `logic_gateway` (logic.gateway.hpp:86): writer shape + `FsmInputs` empty + dt lives on `FsmTick` action instead of Inputs (L125-129) while input pod takes dt from Inputs — inconsistent time placement across pods. Port unifies: time in Inputs, gateway returns `expected<FsmStep, FsmError>`; silent `continue` drops become observable (K3.2). Namespace `shs` -> `shs::logic` (L23; every other pod is `shs::<pod>`, FsmDesc/FsmState/FsmEvent are the only domain types at root namespace).
- [ ] **K1.4 camera** — `camera_gateway` (camera.gateway.hpp:32-42) discards ALL arguments (`(void)state; (void)actions; ...`): a gateway that silently eats every command, the trivial worst-case zero-signal-loss violation. Decide: real camera gateway absorbing the camera math that currently lives in input's MoveLocal/Look handling (input.gateway.hpp:45-68 reaches directly into `state.camera` — cross-vocabulary coupling), or fold camera vocabulary into the input pod and delete the stub. DoD: no discard-all gateway in the tree.
- [ ] **K1.5 The 8 silent pods (frame, geometry, gfx, lighting, sky, scene, resources, camera-vocab)** — all are identity shells with the writer signature (evidence: grep `inline void reduce_` = 12 hits, all `void ... pmr::vector<X>&`). Mechanical port to the Kleisli shape; their vacuous error channel is `void`-error or a single `None`-style closed enum until real failure modes are designed (failure-rail law: an error enum with no real values is worse than none — ERROR_FLOW.md documents this convention). DoD: grep `inline void reduce_` in domains = 0; all still kit-green; ERROR_FLOW.md updated per drift gate.

## W2 — Kill the switch monoliths (Rule 2 as amended)

- [ ] **K2.1 input** — `input_gateway` (input.gateway.hpp:30-93) is the textbook forbidden monolith the amended Rule 2 names: literal `switch (action.type)` with 5 arms + `std::get_if` payload fishing inside each arm (type+payload double dispatch, desync-prone). Decompose into per-intent gateway functions (`apply_move_local`, `apply_look`, ...) assembled by the pod's Kleisli gateway over the closed `RuntimeAction` variant (`std::visit` + `if constexpr`, renderpath style). DoD: `switch` on action discriminators in `*.gateway.hpp` = 0 (add the grep to K6.2's gate when the last pod lands).
- [ ] **K2.2 logic + renderpath dispatch audit** — both use variant ladders living inline in the public gateway (`std::get_if` chains in logic.gateway.hpp:96-133; `std::visit` + `if constexpr` in renderpath). Legal under Rule 2's letter (dispatch on the closed variant, not a parallel enum), but the transition logic belongs in named per-intent functions; the gateway is only the assembly point (amended §2.1 definition). DoD: each public gateway reads as the arrow-chain assembly, no inline transition bodies.

## W3 — Error-channel + zero-signal-loss conformance

- [ ] **K3.1 Failure-rail inventory per pod** — walk all 11 pods: for each transition, classify infallible / fallible-with-closed-error / currently-silent. Output: per-pod error enums (named `*Error`/`*Rejection`/`*Reason` per ERROR_FLOW.md law) + ERROR_FLOW.md rows. Seed data: renderpath already has the only real rail (`PathSwapRejectionReason`, 6 values, mapped 1:1 from the compiler enum at renderpath.gateway.hpp:55-67 — this stays the model). DoD: ERROR_FLOW.md covers every pod's error vocabulary; drift gate green.
- [ ] **K3.2 Kill silent signal drops in logic** — `continue` sites consume a command and emit NOTHING: `!state.started` on signal (logic.gateway.hpp:111), no-rule-match on signal (L113) and on tick (L131). Under zero-signal-loss these are invisible failures. Emit `FsmSignalRejected`/`FsmTickNoRule` facts or route through the gateway error channel — pick ONE house answer and mirror it in renderpath's silent no-op sites (renderpath.gateway.hpp:155, 166, 175: same-technique/same-mode commands return silently; document the decision as a constitution note, not folklore).
- [ ] **K3.3 Compensator sweep (Rule 12)** — only the saga spike test exercises compensation today. Once W1 lands, audit each multi-step transition for invertibility: every state mutation inside a gateway must have a recorded compensator fact or be provably idempotent. DoD: per-pod compensator note in ERROR_FLOW.md or an explicit "no multi-step flows" entry.

## W4 — State-shape hardening

- [ ] **K4.1 `has_plan` phantom flag** — `RenderPathPodState::has_plan` (renderpath.gateway.hpp:35) is a validity bit shadowing plan existence: state can lie (default plan + has_plan=false is indistinguishable from an empty-but-real plan). Replace with a generation counter (`uint32_t plan_generation = 0`, 0 = none) — value-honest, replay-friendly, and it feeds P6.1's plan-hash executor rebuilds when those unblock. DoD: grep `has_plan` = 0; renderpath tests pin generation semantics.
- [ ] **K4.2 Positional-bool event** — `CullingModeChangedEvent{ true, previous, mode }` (renderpath.gateway.hpp:170/179) packs view-vs-shadow into an unnamed leading bool — call-site blindness and one event carrying two facts. Split into `ViewCullingModeChangedEvent` / `ShadowCullingModeChangedEvent` (closed vocabulary stays closed; EVENT_FLOW.md + name tables updated together per the P4.5 machinery). DoD: zero unnamed-bool payload fields in events; drift gates green.

## W5 — Gateway uniqueness + legacy seams

- [ ] **K5.1 input dual-gateway retirement** — input pod exposes BOTH `input_gateway` (house signature) and `runtime_state_gateway` (legacy by-value signature, value_commands.hpp:35-43, consumed by edge/command_processor.hpp:53). One pod, one public Kleisli gateway. Port the edge consumer, delete the legacy wrapper. DoD: value_commands.hpp gone or applied to a deprecation banner; single grep-visible gateway per pod.
- [ ] **K5.2 Grandfathered-architecture sweep** — parked items whose unlock conditions this directive supersedes: legacy callback StateMachine beside the value FSM (logic — "zero consumers" per P3.10; verify then delete), PluggablePipeline/FrameGraph retention re-check (P5.2 kept them pending P6.1; re-audit if W1 changes the calculus), AssetRegistry-class stale forks (deleted in P5.1; confirm none regrew).

## W6 — Mechanical drift gates (checker)

- [ ] **K6.1 Kleisli-shape gate** — extend `check_kdba_boundaries.sh`: once the FIRST pod lands its gateway (K1.2), FAIL on any NEW `inline void reduce_*` added outside the migrated-pod list (grandfather list carried in the script, shs/pipeline facade-case precedent — same mechanism as P1.1). Prevents writer-shape regrowth during the phased migration.
- [ ] **K6.2 switch-monolith gate** — FAIL on `switch` over action discriminators in `*.gateway.hpp`; lands with W2 completion (audit grep already proven).
- [ ] **K6.3 Silent-drop gate** — grep-gate the logic silent-drop pattern once K3.2 fixes it — ONLY if the team picks "events" over "error channel"; a gate on an undecided design choice is premature. Standing until K3.2 decides.

## Rolled forward from the frozen backlog (tracked, not dropped)

- [ ] **P6.1 PATH_COMPILED-driven executor rebuilds** — BLOCKED (needs a live demo host; unchanged).
- [ ] **P6.2 Replay harness (cross-session codec + CI replay)** — BLOCKED on demo host; K1.5 strengthens the pod-level replay story (Step values are serializable by construction).
- [ ] **P6.3 Rollback snapshots + time-travel overlay** — BLOCKED on windowed host; K4.1's generation counter is a precondition contributor.
- [ ] **P4.4 Seeded determinism contract (STANDING)** — still no stochastic pods; first stochastic pod authors the contract.

## Run plan

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