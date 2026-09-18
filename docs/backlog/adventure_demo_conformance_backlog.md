# Adventure Demo Domain Boundary & Composition Backlog

> Status: active, implementation not started (2026-09-17).
> Scope: six SW/Vulkan pairs under `cpp-folders/src/exps-rendering-adventures/` (tier0 01–05 and tier1 08), common helpers, and parity tooling. Parked experiment trees are excluded.
> Authority: [Constitution II](../spec/value_oriented_programming.md) and the [governing clarification](../spec/dod_ecs_architecture.md). This backlog schedules work; it does not add laws or reopen the completed library migration.
> Related: [library KDBA backlog](kdba_conformance_backlog.md).

## Review corrections and guardrails

- Demo 01's centroid probe prints a sample; it is neither an assertion nor a typed event today.
- A draw list is declarative data, not proof of monadic composition. Typed success/failure chaining needs implementation and tests.
- Shared geometry does not establish shared ownership of all semantics. Transforms, texture generation, pass policy, and shader constants still need review.
- There is no evidence that duplicated constants caused the normal-mapping tolerance envelope. Preserve measured envelopes; investigate differences rather than attributing them speculatively.
- Executor-local mutation and explicitly owned output buffers are legal. Remove hidden call-order dependencies, not all mutation. Backend layouts may differ; keep conversion at the execution boundary.
- Preserve demo 01's from-scratch barycentric lesson and other lesson-specific kernels. Do not introduce per-pixel events, dummy stateful pods, vacuous errors, or a general-purpose demo engine for signature uniformity.

## Task table

All implementation tasks are open. Finding numbers refer to the adventure-demo audit, not library P6 integration work.

| ID | Priority | Task | Review finding | Depends on |
| :--- | :--- | :--- | :--- | :--- |
| AD0 | First | Fresh reproducible baseline | Validation prerequisite | — |
| AD1 | High | Portable automated demo gates | P7 | AD0 |
| AD2 | High | Shared semantic state and explicit draw inputs | P1, P2 | AD0 |
| AD3 | High | Typed compositional orchestration pilot | P6 + composition gap | AD2 |
| AD4 | High | Independent known-answer tests | P3 | AD0; register through AD1 |
| AD5 | Medium | Single-source shared scene inputs | P5 | AD0 |
| AD6 | Medium | Isolate vertex/push-constant packing | P4 | AD2 |
| AD7 | Final | Roll out and close with evidence | All | AD1–AD6 |

## AD0 — Establish the baseline

- [x] Discover current build configuration and registered CTest tests; record commands, compiler, Vulkan device/driver, and Slang version where available. (2026-09-18, `adventure_demo_baseline_2026-09-18.md`)
- [x] Rebuild six SW demos and available Vulkan twins; regenerate outputs and record parity for all six pairs. Record unavailable backends explicitly, not as passes. (2026-09-18: all six twins available and exercised; fresh parity table in baseline note)
- [x] Separate existing defects from refactor regressions. Inspect enclosing CMake/CI wiring before claiming that the suite is entirely manual. (2026-09-18: two envelope breaches proven pre-existing device drift via 983925c pre-R1 worktree cross-check; demos were unregistered — wiring inspected, registration now lands via AD1)

**Acceptance:** reproducible commands and fresh results attached to close-out notes. Library test counts are not substituted for demo coverage.

## AD1 — Portable automated gates

- [x] Replace machine-specific paths in `t0_parity_suite.py` with explicit arguments supplied by CMake; use isolated output directories and retain subprocess diagnostics. (2026-09-18: --build-dir/--tier1-build-dir/--scratch-dir/--only; no baked-in paths; diagnostics retained)
- [x] Register SW smoke/known-answer tests independently of Vulkan. Register all available twin comparisons with CTest, including tier1 08. (2026-09-18: 6 SW smoke tests always registered; 6 per-pair parity tests when slang+Vulkan, incl. t1 08; full CTest 63/63)
- [x] Distinguish missing optional capability from failure: documented capability skips only; shader, upload, render, PNG, and comparison failures remain failures on an enabled backend. (2026-09-18: suite SKIP = missing *_vk binary; binary/tool/PNG failures are FAILs)
- [x] Test parity-tool argument handling (including documented `--tol 1` and `--tol=1` forms), malformed comparator output, subprocess failures, and envelope breaches. (2026-09-18: t0_gate_negative_probes.py + CTest t0_gate_negative_probes; probe found and fixed a real `--tol 1` parsing bug in t0_parity.py, plus malformed-output and subprocess-crash hardening in the suite)

**Acceptance:** GPU-free and Vulkan-enabled configurations report accurate pass/skip/fail results without checkout-specific paths. Negative probes fail reliably; envelopes are not loosened to pass a refactor.

## AD2 — Shared semantics, explicit draw inputs

- [ ] Define the smallest execution-neutral fixed-function policy needed by current demos (depth, blend, stencil, scissor). Keep shader paths and Vulkan handles out; document defaults and capability differences.
- [ ] Map this policy into SW and Vulkan adapters rather than maintaining unrelated `SwState`/`VkPipelineSetup` semantic copies.
- [ ] Pass policy explicitly per SW draw; remove prerequisite `raster.state = ...` sequencing. Keep depth/stencil/framebuffer mutation inside the executor.
- [ ] Test opaque/translucent order, stencil write/equal/invert, and scissor bounds. Resolve projection's unused local `SwState` without relying silently on defaults.

**Acceptance:** both twins consume the same intended pass policy; previous draw configuration cannot leak into the next draw. Lesson kernels stay visible and baseline comparisons hold.

## AD3 — Typed composition pilot

- [ ] Pilot demo 03: pure preparation/validation returns a backend-neutral plan; execution and PNG output stay at explicit edges.
- [ ] Inventory real failures and introduce a closed error vocabulary with stage diagnostics. Adapt bool/index failures without inventing failures for infallible math.
- [ ] Compose fallible stages using the project's C++23 `std::expected` conventions (`and_then`, `transform`, `transform_error`, or `or_else` as appropriate). Map errors to CLI diagnostics/exit codes once at the host boundary.
- [ ] Test success and failure at each stage: later stages do not execute after failure, original errors survive, and acquired resources are released. Short-circuiting does not imply rollback of external effects.

**Acceptance:** tests exercise a typed chain, not just renamed calls or a draw vector. No driver/file I/O enters pure preparation; no borrowed plan payload outlives its owner.

## AD4 — Independent known-answer tests

- [ ] Add always-active checks that fail the test process (not `assert` alone, which may disappear in release builds). Return diagnostics to the test/host edge rather than printing inside pure kernels.
- [ ] Demo 01: check interior barycentric color against analytically derived weights at the sampled pixel center, with an explicit quantization tolerance.
- [ ] Demo 02: check selected transformed coordinates and depth outcomes independently of twin image agreement.
- [ ] Demo 03: check opaque overlap and alpha composition; inspect depth storage to prove transparent draws do not write depth.
- [ ] Demo 04: check known nearest/bilinear samples, repeat wrapping, and unchanged pixels outside the scissor.
- [ ] Demo 05: check stencil storage and equal/inverted-mask behavior at interior/exterior sample points.
- [ ] Demo 08: check decoded normals, tangent-frame behavior for the supported input, and analytical flat/mapped shading samples. Do not assume one half is always brighter.
- [ ] Prove checks fail with deliberately wrong expected values or temporary mutations; a correlated SW/Vulkan mistake must not pass solely through parity.

**Acceptance:** every lesson has independent numerical assertions with justified tolerances; tests run in GPU-free CI. Document input restrictions separately from guarantees of a general rasterizer.

## AD5 — Single-source shared inputs

- [ ] Hoist duplicated checkerboard and bump-normal generation into execution-neutral helpers consumed by both C++ twins.
- [ ] Share projection transforms, normal-map settings, lighting/material constants, frame extent, and clear policy where they represent the same lesson intent.
- [ ] Supply shader parameters from shared semantic inputs through an explicit adapter when practical. If literals remain, add a drift check or known-answer test; comments alone are not enforcement.
- [ ] Keep SW math and Slang implementations separate where they teach distinct execution paths; share inputs rather than forcing one backend's layout on the other.

**Acceptance:** texture bytes and transforms match the pre-refactor baseline; each shared parameter has a clear owner and both backends consume it. Any intentional numerical change is separately explained and tested.

## AD6 — Execution-format adapters

- [ ] Keep `T1Vertex`'s normal vocabulary independent of `T0Vertex`'s color packing. Move the existing normal-to-COLOR0 conversion to a named execution adapter rather than exposing it as scene construction.
- [ ] Isolate push-constant matrix layout/transposition rules at the Vulkan boundary; reconcile contradictory layout comments against generated shader evidence.
- [ ] Test vertex offsets/stride, normal packing, push-block size, and a non-identity matrix upload. Document borrowed `VkDraw::push` lifetime or replace it with owned payloads if plans will be retained.
- [ ] Evaluate configurable Vulkan vertex layouts only when needed by a concrete lesson. A tested adapter is sufficient for the current pilot; no mandatory generalized layout framework.

**Acceptance:** pure scene construction does not depend on Vulkan headers, handles, shader byte layout, or normal-in-color conventions. Packing tests and fresh parity both pass.

## AD7 — Rollout and close-out

- [ ] After AD3 proves the shape, apply the minimal preparation/execution/error composition pattern to the remaining five pairs. Reuse helpers without hiding each lesson's algorithm.
- [ ] Document ownership and include direction for demo semantic values, pure preparation, backend adapters, and CLI/output edges. Use existing naming conventions for any real pods introduced; do not require a pod per kernel.
- [ ] Add narrowly scoped boundary checks for the new pure zone with negative probes. Do not ban Vulkan calls in Vulkan executors or output-buffer writes in kernels.
- [ ] Rebuild and run SW known-answer tests, available backend parity, composition failure tests, boundary checks, and existing library CTest coverage. Record exact commands/results and skips.
- [ ] Update this table with completion evidence, remaining limitations, and links to changes. Keep library P6 host integration and S1–S6 scalability tasks separate.

**Acceptance:** all six pairs demonstrate explicit semantic ownership and typed fallible orchestration; fresh tests verify behavior, failures, and boundaries. No implementation checkbox closes on documentation alone.

## Suggested delivery order

1. AD0, then AD1 + AD4: establish trustworthy feedback before refactoring.
2. AD2 + AD3: prove the architecture on the depth/blend pair.
3. AD5 + AD6: consolidate inputs and contain backend representations.
4. AD7: roll out, validate, and record closure.

Each delivery should be independently reviewable. Preserve tolerance envelopes unless a separately investigated rendering correction justifies a change. Current status: backlog authored only; no demo refactors or new tests have been implemented by this document.
