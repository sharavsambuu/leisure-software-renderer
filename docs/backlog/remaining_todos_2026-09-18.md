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
- Only unblocked, ready-to-start work: **Bucket A** below.

## Bucket A — Unblocked, ready to start

### A1. GPU execution path (owner: [`kdba_conformance_backlog.md`](kdba_conformance_backlog.md) §"GPU execution path — selected next work")

- [ ] **K-G1 tail — explicit recording failures** — dedicated command-buffer /
      missing-buffer branch coverage; broader ID checks; remaining
      recording-order validation; stage-diagnostics breadth. Mostly landed
      2026-09-17 (closed error vocabulary `MissingPipeline`/`MissingImage`/
      `MissingBinding`/`InvalidCommand`/`InvalidRecordingOrder`, whole-stream
      preflight, real-device prerequisite suite). Non-goals stand: sink
      failures stay fail-fast (no transactional rollback); headless evidence
      only. Depends on: —
- [ ] **K-G2 Minimal offscreen graphics realization** — finish attachment
      setup / pipeline creation+binding / begin-end-pass recording breadth in
      the new driver (slice 1 partial: `b2d7d79`; realization now at
      `shs/rhi/vulkan/value/vk_offscreen.hpp` after the R5 sweep). Acceptance:
      one deterministic scene through value commands; explicit
      formats/layouts/features; failed creation unwinds; zero validation
      errors on an available backend. Depends on G1.
- [ ] **K-G3 remaining: staging→device-local copy upload + factory-facing
      execution** — most of G3 is partial-complete (submission/readback,
      shutdown cache-invalidation, upload + failure-injection slices, triangle
      parity vs the software rasterizer). Depends on G2.
- [ ] **K-G4 Library SW/Vulkan equivalence** — same minimal scene/policy
      through actual library execution paths; documented per-output tolerances;
      independent known-answer checks; portable CTest gates. Adventure AD1/AD4
      are related, not substitute evidence. Depends on G3 and a verified
      software realization of the selected recipe.

### A2. Adventure demo conformance (owner: [`adventure_demo_conformance_backlog.md`](adventure_demo_conformance_backlog.md) — 36 open checkboxes, unstarted)

- [ ] **AD0 Fresh reproducible baseline** — record build config/CTest
      registration, compiler, Vulkan device/driver, Slang version; regenerate
      all six SW/Vulkan parity pairs; unavailable backends recorded as
      explicitly unavailable, never as passes. Depends on: —
- [ ] **AD1 Portable automated demo gates** — de-machine-specific
      `t0_parity_suite.py`; register SW smoke/known-answer tests and all
      available twin comparisons (incl. tier1 08) with CTest; documented
      capability skips only. Depends on AD0.
- [ ] **AD2 Shared semantic state + explicit draw inputs.** Depends on AD0.
- [ ] **AD3 Typed compositional orchestration pilot** (depth/blend pair).
      Depends on AD2.
- [ ] **AD4 Independent known-answer tests** (GPU-free CI, justified
      tolerances). Depends on AD0; register through AD1.
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

## Suggested sequencing

1. K-G1 tail → K-G2 (smallest unblocked step, same area as the R-series
   close-out); AD0 + AD1/AD4 in parallel (independent track).
2. S5 headless prep to bank host-blocked P6.2 work early, if desired.
3. C4.3 waits on the owner baseline ruling; P6.1/P6.3 wait on a windowed host.

## Related demowork

The adventure-demo backlog (A2) is the consumer-facing track;
[`kdba_conformance_backlog.md`](kdba_conformance_backlog.md) §"Related
demowork" keeps the scopes separate — demo evidence never substitutes for
library gates and vice versa.
