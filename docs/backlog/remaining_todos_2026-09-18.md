# Remaining Todos — Consolidated Tracker (2026-09-18)

> Status: **active (2026-09-18)** — created during the 2026-09-18 docs sweep so
> all remaining work is scannable in one place. This file is schedule, not law
> (Constitution II §2.2) and owns no task canonically: each item links its owner
> backlog, which remains the single source of truth. An item may only be ticked
> here in the same commit its owner-doc entry closes, with the same DoD
> evidence. Verification after every code item: full `build/` CTest green +
> `check_kdba_boundaries.sh` green + `check_include_graph.py` green; a touched
> header regenerates the inventory in the same commit (`inventory_headers.py
> --write`, Rule 15). Provenance: compiled after the R-series rasterizer
> hot-path close-out (backlog archived at
> [`docs/outdated/rasterizer_hot_path_todo.md`](../outdated/rasterizer_hot_path_todo.md))
> and the governance-review close-out (archived same day); CTest baseline
> **50/50** at commit `1f8400f`.

## State snapshot (2026-09-18)

- Library migration stabilized (KDBA Runs A–C closed, `05bb646`).
- Rasterizer hot-path track closed & archived (R1–R5); governance review
  closed & archived (G1–G3); `rhi/drivers/vulkan/` forwarders retired (`8b6484b`).
- **Correction (2026-09-18, owner ruling — FINAL): the previous snapshot line
  ("only unblocked work = Bucket A") was wrong.** The owner ruled that the DVO
  architecture — domain-context-separated Domain Value Objects, monadic (Kleisli)
  gateway composition, contract guardrails — is final and binding. The library-side
  programs are *closed* (domain-separation migration archived; namespace cutover
  0 open; contract guardrails C0–C3 done with only C4.3 toolchain-blocked;
  enforcement W-A…W-E 0 open; KDBA A–C, G1–G4 closed), and the genuine library
  remainder is **A0** below. Bucket A's demo track is *not* the only startable work;
  it was believed to be because
  [`domain_pod_engine_rollout_roadmap.md`](../roadmap/domain_pod_engine_rollout_roadmap.md)
  was stale (its P0.5 tree record contradicted the live tree, its P3 boxes 1–3 were
  done-unticketed, and its parked backlog was superseded). That file now carries a
  reconciliation banner.

## Bucket A — Unblocked, ready to start

### A0. Library DVO-architecture remainder (verified 2026-09-18 — owner: [`domain_pod_engine_rollout_roadmap.md`](../roadmap/domain_pod_engine_rollout_roadmap.md) reconciliation banner)

- [ ] **P3 box 4 — monolith → thin DVO composition.** `demo_forward_classic_renderpath.cpp`
      is 9,800 lines against a <1.5k DoD; its extracted pieces already exist and are
      CTest-gated (`demo_input_actions.hpp`, `demo_renderpath_bridge.hpp`,
      `demo_frame_planner.hpp`), so the work is wiring + deleting inline copies.
      The largest remaining structural win, and it is the proof-by-consumer for the
      whole DVO spine. Library-adjacent, not library-internal.
- [x] **P3 box 6, first half — open pass IDs** (Constitution I §7). **CLOSED
      2026-09-18:** `PassId` range law (`planning/pass_id.hpp`) + content-addressed
      `PassIdRegistry` (`execution/pass_id_registry.hpp`); a consumer pass is
      first-class with zero core edits — verified plan participation, typed plan
      queries, factory/descriptor registration under a verified `(id, name)`
      pair, loud collision refusal, no cross-registry aliasing. Gate
      `shs_renderer_pass_id_open_tests` (12 GPU-free checks); full CTest **72/72**
      (was 71); evidence
      [`open_pass_id_registry_evidence_2026-09-18.md`](open_pass_id_registry_evidence_2026-09-18.md).
      Independently corroborated by the 2026-09-18 Antigravity library review.
- [x] **Slang-plan P1.5 — shader identity as data (manifest core).** **CLOSED
      2026-09-18:** one `ShaderId` names one authored shader and each backend
      resolves its own realization — software yields a `ShaderProgram`, Vulkan
      yields module + entry points, and a backend with no realization reports
      `BackendNotRealized` instead of falling back. `ShaderManifest` is
      caller-owned/copyable/comparable (no ambient state), registration is
      verified `(id, name)` pairing, and the software offscreen realization's
      entry-name law now resolves through it (its duplicated literals are gone).
      Gate `shs_renderer_shader_identity_tests` (83 assertions, GPU-free,
      verified against the authored `offscreen_pipeline.slang`); full CTest
      **73/73**; contract-placement coverage 226 → **228** headers; three
      mutation probes failed the gate and were reverted byte-identically.
      Evidence
      [`shader_identity_manifest_evidence_2026-09-18.md`](shader_identity_manifest_evidence_2026-09-18.md).
      Residuals stated there: no consumer/open shader ids yet; the Vulkan
      binding is descriptive truth, not yet the loader's input (P2); the value
      builtins are registered software-only on purpose, making the
      dual-realization gap a census rather than invisible debt. The OpenGL
      selection note in `backend_factory.hpp` was corrected in the same commit —
      the identity layer now refuses OpenGL by data, where before selecting it
      silently ran software.
- [ ] **P3 box 6, second half — open light/technique registries.**
      `RenderPathLightVolumeProvider` + the technique/light preset enums get the
      same treatment (rule of two: reuse `PassIdRegistry`'s shape). Unblocked,
      pure library API work.
- [ ] **P5 box 1 — role/suffix completion.** Today each DVO zone carries a single
      `.contract.hpp` (`render` carries two) and only 5 gateways exist (`renderpath`,
      `render/frame`, `input`, `logic`, `app/session_orchestrator`); there are **0**
      `.plan.hpp` / `.edge.hpp` splits (`renderpath` holds 40 headers against 1
      contract header). Extends the existing `check_contract_placement.sh` gate's
      coverage (225 headers today).
- [ ] **P5 box 4 — retire `frame_graph.hpp` / `pluggable_pipeline.hpp`.** Genuinely
      open but **PATH_COMPILED-gated** (P6): `frame_graph.hpp` (218 lines) has one
      internal consumer; `pluggable_pipeline.hpp` (1,050 lines) still has 3
      (monolith, `hello_rendering_paths`, `core_tests`). Its facade role is already
      thin — `PluggablePipeline::execute` delegates to
      `PipelineRuntimeExecutor` + `PipelineExecutionPlanner` inside the same file.
- [ ] **P3 box 7 — migrate or retire the `hello_*_vulkan` probes.**
- [ ] **P4 — demo-internal Core 4 debt** (tetris/snake vocabularies; regenerate
      `docs/pods/EVENT_FLOW.md`).
- [ ] **Enforcement actually runs.** The repo has **no CI at all** (no
      `.github/workflows`), so the 71 CTest gates + boundary/include-graph/purity
      gates are opt-in — they run only when someone remembers. VOP-first roadmap
      item 5 ("wire CTest gates into CI so regressions fail automatically") is
      therefore genuinely open. Platform choice is an owner decision.
- Host-blocked, not startable: **P6** replay/rollback (PATH_COMPILED-driven executor
  rebuilds) and **C4.3** (needs `__cpp_contracts` + `<contracts>`; GCC 13.3.0 lacks
  both — owner baseline ruling outstanding).

### A1. GPU execution path (owner: [`kdba_conformance_backlog.md`](kdba_conformance_backlog.md) §"GPU execution path — selected next work")

- [x] **K-G1 tail — explicit recording failures** — CLOSED 2026-09-18:
      command-buffer prerequisite branch proven for non-empty streams (headless
      recorder suite + real-device prerequisite suite, correct
      stage/index/command attribution); `recording_ready` prerequisite
      precedence proven over stream validation with zero sink calls;
      missing-buffer branch and order-validation/stage-diagnostics breadth
      already table-pinned. Full CTest 50/50. Non-goals stand: sink failures
      stay fail-fast; headless evidence only. Pass/pipeline realization and
      in-pass execution are G2; factory-facing execution is G3. Next in this
      track: K-G2.
- [x] **K-G2 Minimal offscreen graphics realization** — finish attachment
      setup / pipeline creation+binding / begin-end-pass recording breadth in
      the new driver (slice 1 partial: `b2d7d79`; realization now at
      `shs/rhi/vulkan/value/vk_offscreen.hpp` after the R5 sweep). Acceptance:
      one deterministic scene through value commands; explicit
      formats/layouts/features; failed creation unwinds; zero validation
      errors on an available backend. Depends on G1.
      CLOSED 2026-09-18: acceptance evidenced end-to-end — deterministic scene
      through value commands to known pixels, explicit supports() tables, unwind
      on failure (incl. injected Vulkan faults), validation-clean lavapipe runs,
      SW/Vulkan triangle parity; see the kdba G2 DONE entry. Follow-on work
      belongs to G3 (upload tail — landed, factory-facing execution) and G4.
- [x] **K-G3 remaining: factory-facing execution** — CLOSED 2026-09-18: the
      minimal scene now runs from `create_render_backend()` + `app::Context`
      through a new vendor-free `IOffscreenExecution` hook on `IRenderBackend`
      (no `dynamic_cast`, no Vulkan type in the consumer), pinned by the portable
      gate `shs_renderer_vk_factory_offscreen_tests` (known pixels derived from
      the authored scene, rejection + reset/re-prepare, SKIP 77 when the surface
      or device is unavailable). Full CTest 68/68; boundary + include-graph gates
      green; header inventory regenerated. Evidence:
      [`kdba_g3_factory_facing_evidence_2026-09-18.md`](kdba_g3_factory_facing_evidence_2026-09-18.md).
      Async retirement unclaimed. Next in this track: K-G4 (closed 2026-09-18 —
      see the K-G4 entry below).
- [x] **K-G4 Library SW/Vulkan equivalence** — same minimal scene/policy
      through actual library execution paths; documented per-output tolerances;
      independent known-answer checks; portable CTest gates. Adventure AD1/AD4
      are related, not substitute evidence. Depends on G3 and a verified
      software realization of the selected recipe.
      CLOSED 2026-09-18: the software side now realizes the generic contract
      instead of declining it (`SoftwareOffscreenExecution` — no device to open,
      descriptor-derived stable ids, entry-name-bound CPU realization), and both
      realizations share one vendor-free descriptor gate so acceptance cannot
      drift. New portable gate `shs_renderer_sw_vk_equivalence_tests` drives ONE
      consumer function (factory → `app::Context` → generic `IRenderBackend` →
      generic `IOffscreenExecution`) against both backends with the same command
      stream, checks each readback against the authored scene independently, then
      compares them under two documented tolerances (≤ 1/255 per channel where
      both cover; coverage budget 16 at 32×32, GPU may only add coverage).
      Measured `cpu_covered=113 gpu_covered=128 mismatches=15 (both=0 gpu_only=15
      cpu_only=0)`. The software half is always asserted; only the Vulkan half may
      skip (77, equivalence explicitly not claimed). Full CTest 69/69; boundary +
      include-graph + self-containment + package-consumer gates green; inventory
      regenerated (225 → 226). Evidence:
      [`kdba_g4_sw_vk_equivalence_evidence_2026-09-18.md`](kdba_g4_sw_vk_equivalence_evidence_2026-09-18.md).
      Not claimed: CPU realization bound to the authored recipe by entry name;
      software accepted set is a strict subset of Vulkan's; coverage tolerance
      calibrated for this fixture; gate not compiled in a GPU-free build.
      **The library G-track (G1–G4) is now complete.**

### A2. Adventure demo conformance (owner: [`adventure_demo_conformance_backlog.md`](adventure_demo_conformance_backlog.md) — 13 open checkboxes; AD0, AD1, AD2, AD3, AD4 closed 2026-09-18)

- [x] **AD0 Fresh reproducible baseline** — DONE 2026-09-18:
      evidence in [`adventure_demo_baseline_2026-09-18.md`](adventure_demo_baseline_2026-09-18.md).
      All six pairs rebuilt + parity-recorded (no backend unavailable);
      two envelope breaches (01, 02) proven pre-existing device drift
      (Mesa 25.2.8) via a pre-R1 (`983925c`) worktree cross-check and
      re-pinned with dated rationale (shrink-only from here); zero refactor
      regressions. Depends on: —
- [x] **AD1 Portable automated demo gates** — DONE 2026-09-18:
      suite fully de-machined (CMake-supplied dirs, isolated scratch
      outputs, capability-skip vs failure split); 6 SW smoke + 6 per-pair
      parity CTest entries (incl. tier1 08) + negative-probe test; probes
      caught and fixed a real `--tol 1` parsing bug in `t0_parity.py`.
      Full CTest 63/63. Depends on AD0.
- [x] **AD2 Shared semantic state + explicit draw inputs** — CLOSED 2026-09-18:
      one execution-neutral `PassPolicy` (`common/adventures_pass_policy.hpp`)
      replaces the parallel `SwState`/`VkPipelineSetup` copies; `SwState` is
      deleted and the policy is a per-draw argument, so the prerequisite
      `raster.state = ...` sequencing (and the leak it allowed) is gone; the
      Vulkan harness bakes pipeline state from the same policy and derives its
      dynamic scissor from it. Stencil mode is closed, retiring the
      silently-ignored "invert without test" pair. Gate `t0_policy_tests` (45
      checks, GPU-free, cannot skip) covers defaults, depth storage, order +
      analytic source-over, all four stencil modes, scissor bounds/clamp, and
      the policy-isolation property; three mutation probes failed their gate.
      Six-pair baseline numbers identical to AD0 — no pixel moved. Depends on AD0.
- [x] **AD3 Typed compositional orchestration pilot** (depth/blend pair) —
      CLOSED 2026-09-18: demo 03 is now pure preparation
      (`depth_blend_plan.hpp`: request → `expected<DepthBlendPlan,
      DepthBlendError>`, plan owns its geometry) → explicit execution/PNG edges
      (`depth_blend_edges.hpp`, plus the Vulkan twin's own executor) with one
      host-boundary diagnostic mapping and stage-derived exit codes
      (preparation 1 / execution 2 / output 3). Closed 14-member error
      vocabulary; both twins consume the same plan. Gate
      `t0_composition_tests` (91 checks, GPU-free) covers plan shape,
      short-circuiting with the original error preserved, execution's refusal of
      unvalidated plans, the output edge (descriptor count unchanged on failure),
      plan ownership, vocabulary totality, and policy propagation into the
      executed result. Evidence:
      [`adventure_demo_ad2_ad3_evidence_2026-09-18.md`](adventure_demo_ad2_ad3_evidence_2026-09-18.md).
      Full CTest 71/71. Depends on AD2.
- [x] **AD4 Independent known-answer tests** — DONE 2026-09-18: evidence in
      [`adventure_demo_ad4_evidence_2026-09-18.md`](adventure_demo_ad4_evidence_2026-09-18.md).
      Always-active checks that fail the process; demos 01/04/08 verified
      against analytic oracles over the real SW demo PNGs, 02/03/05 driven
      in-process with direct depth/stencil STORAGE inspection; every tolerance
      justified in the note. Validation found and fixed five oracle defects
      (28 mismatches → 0); prove-fail covers both oracle styles. Full CTest
      67/67 + `check_kdba_boundaries.sh` + `check_include_graph.py` green.
      Depends on AD0 (registered through AD1).
- [ ] **AD5 Single-source shared scene inputs.** Depends on AD0.
- [ ] **AD6 Execution-format adapters** (vertex/push-constant packing isolated
      at the Vulkan boundary). Depends on AD2.
- [ ] **AD7 Rollout to remaining five pairs + close-out with evidence.**
      Depends on AD1–AD6.

Guardrails (from the owner backlog): no per-pixel events, no dummy stateful
pods, no vacuous errors, no general-purpose demo engine for signature
uniformity; preserve lesson-specific kernels (incl. demo 01's from-scratch
barycentric lesson); tolerance envelopes preserved unless separately
investigated.

## Bucket B — Blocked on owner ruling

- [ ] **C4.3 / P4 C++26 native contracts switch** (owner:
      [`cpp26_native_switch_runbook.md`](cpp26_native_switch_runbook.md),
      DRAFT) — the in-house `SHS_PRE`/`SHS_POST`/`SHS_CONTRACT_ASSERT` bridge
      is the ratified contract mechanism (owner ruling 2026-09-17); the native
      switch is the sanctioned replacement. Runbook fully drafted (14 sites,
      all single-expression; keyword mapping, handler-move plan, acceptance
      gates, rollback). G3.1 threshold DONE 2026-09-18: fires on
      `__cpp_contracts` + matching `<contracts>` library (feature-test macro
      only, never compiler-name checks). Local baseline GCC 13.3.0 has
      neither; mixed-toolchain items stay gated until Clang ships it.
      **Execution requires only the owner baseline ruling**; the replay-parity
      CTest is the release blocker when the switch fires.

## Bucket C — Blocked on demo/windowed host (P6, owner: kdba backlog)

- [ ] **P6.1 PATH_COMPILED-driven executor rebuilds** — live plan switching,
      resize, safe GPU retirement (accepted plans rebuild; rejected plans
      preserve the working renderer; in-flight resources outlive submission).
      BLOCKED: needs a live demo host.
- [ ] **P6.2 Replay harness (cross-session codec + CI replay)** — explicit
      codecs, versions, reconstruction rules remain unbuilt (K1.5's pod-level
      replay tests are not portable serialization). Host integration BLOCKED;
      **headless codec/fixture preparation may proceed independently via S5**.
- [ ] **P6.3 Rollback snapshots + time-travel overlay** — BLOCKED on windowed
      host; K4.1's generation counter is a precondition contributor.
- [ ] **P4.4 Seeded determinism implementation (STANDING)** — baseline
      contract adopted in Constitution II §11.1; the first stochastic pod
      version-pins its algorithm and proves S1. No completion claimed.

## Bucket D — Trigger-gated (no speculative infrastructure)

Owner: [`kdba_conformance_backlog.md`](kdba_conformance_backlog.md) §
"Future-domain scalability preparedness" (Constitution II §11.1, Rule 12).
Each item carries its explicit trigger; none authorizes speculative work.

- [ ] **S1 Seeded determinism proof (P4.4)** — Trigger: first stochastic pod.
- [ ] **S2 Large-state headless spike** — Trigger: before the first
      high-volume mutable domain; available independently of a windowed host.
- [ ] **S3 Streaming edge contract** — Trigger: first asynchronous
      IO/streaming domain.
- [ ] **S4 Time policy integration** — Trigger: next real-time host
      integration (P6.1).
- [ ] **S5 Replay persistence and retention (P6.2/P6.3)** — Trigger: before
      shipping a persisted session format; headless preparation need not wait
      for GPU/window availability.
- [ ] **S6 Production saga proof (Rule 12)** — Trigger: first multi-domain
      transaction.

## Bucket E — Demand/measure-gated (optimization register)

Owner: [`optimization_backlog.md`](optimization_backlog.md). Not mirrored
checkbox-by-checkbox here: the owner doc carries its own checkboxes and its own
tick rule (a named measurement must show the change pays for itself before any
item closes). 18 boxes / **16 distinct tasks** (QW3≡#1, QW5≡#4), all gated on
consumer demand and measured evidence — no speculative scheduler, render graph,
meshlet pipeline, or LOD framework is authorized by listing them.

## Suggested sequencing

1. **Adventure demo track (A2)** — AD5 + AD6 next (single-source shared inputs,
   then execution-format adapters), then the AD7 roll-out and close-out of the
   remaining five pairs onto AD3's preparation/execution/error shape. This is
   now the only unblocked implementation track: the library G-track closed
   2026-09-18 (G4 was its last item) and AD0/AD1/AD2/AD3/AD4 are closed. It also
   gives consumer-contract feedback at demo granularity, which the library track
   no longer does.
2. **S5 headless prep** — banks host-blocked P6.2 work early, if desired.
3. C4.3 waits on the owner baseline ruling; P6.1/P6.3 wait on a windowed host;
   Bucket E stays demand/measure-gated.

## Related demowork

The adventure-demo backlog (A2) is the consumer-facing track;
[`kdba_conformance_backlog.md`](kdba_conformance_backlog.md) §"Related
demowork" keeps the scopes separate — demo evidence never substitutes for
library gates and vice versa.
