# Adventure Demo Domain Boundary & Composition Backlog

> Status: **active (2026-09-18)** — AD0, AD1, AD2, AD3, AD4 **closed** with
> recorded evidence ([`adventure_demo_baseline_2026-09-18.md`](adventure_demo_baseline_2026-09-18.md),
> [`adventure_demo_ad2_ad3_evidence_2026-09-18.md`](adventure_demo_ad2_ad3_evidence_2026-09-18.md),
> [`adventure_demo_ad4_evidence_2026-09-18.md`](adventure_demo_ad4_evidence_2026-09-18.md),
> plus the CTest gates). AD5, AD6, AD7 open (13 checkboxes); AD7 is the roll-out
> that applies AD3's shape to the remaining five pairs.
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

- [x] Define the smallest execution-neutral fixed-function policy needed by current demos (depth, blend, stencil, scissor). Keep shader paths and Vulkan handles out; document defaults and capability differences. (2026-09-18: `common/adventures_pass_policy.hpp` — `PassPolicy` + closed `StencilMode` + `ScissorRect`; defaults and the four capability differences documented in the header)
- [x] Map this policy into SW and Vulkan adapters rather than maintaining unrelated `SwState`/`VkPipelineSetup` semantic copies. (2026-09-18: `SwState` deleted; `SwRaster` holds only storage; `VkPipelineSetup` embeds `PassPolicy`, `stencil_state(StencilMode, ref)` and `clamp_scissor` are the Vulkan adapters; `render(draws, policy)` applies the dynamic scissor)
- [x] Pass policy explicitly per SW draw; remove prerequisite `raster.state = ...` sequencing. Keep depth/stencil/framebuffer mutation inside the executor. (2026-09-18: `draw_triangles(raster, policy, verts)`; no state member remains to leak; demos 01–05 twins, tier1 08 `*_vk` and the AD4 tool migrated)
- [x] Test opaque/translucent order, stencil write/equal/invert, and scissor bounds. Resolve projection's unused local `SwState` without relying silently on defaults. (2026-09-18: `t0_policy_tests`, 45 checks, GPU-free — order + analytic source-over, the four stencil modes against the stencil plane, scissor bounds/empty band/clamp, and the policy-isolation property; demo 02's dead local `SwState` replaced by an explicit stated policy)

**Acceptance:** both twins consume the same intended pass policy; previous draw configuration cannot leak into the next draw. Lesson kernels stay visible and baseline comparisons hold.

**CLOSED 2026-09-18.** Evidence: [`adventure_demo_ad2_ad3_evidence_2026-09-18.md`](adventure_demo_ad2_ad3_evidence_2026-09-18.md).
Gate `t0_policy_tests` (45 checks) plus the unchanged six-pair baseline table
(every differ/within-1/max number identical to AD0) and full CTest 71/71. Three
mutation probes (kernel depth test removed, plan `blend=false`, executor
ignoring `pass.policy`) each failed their gate and were reverted. Stencil mode is
now closed, which retires the silently-ignored "invert without test"
combination. Lesson kernels untouched.

## AD3 — Typed composition pilot

- [x] Pilot demo 03: pure preparation/validation returns a backend-neutral plan; execution and PNG output stay at explicit edges. (2026-09-18: `03_depth_test_alpha_blend/depth_blend_plan.hpp` (pure: request → `expected<DepthBlendPlan, DepthBlendError>`, plan owns its vertices) + `depth_blend_edges.hpp` (software execution + PNG edge); both twins consume the one plan)
- [x] Inventory real failures and introduce a closed error vocabulary with stage diagnostics. Adapt bool/index failures without inventing failures for infallible math. (2026-09-18: 14-member `DepthBlendError` + `DepthBlendStage`; six preparation, two software-execution, five Vulkan-execution values adapted from bools/indices, one output value; single `depth_blend_stage()` mapping, total message/name helpers)
- [x] Compose fallible stages using the project's C++23 `std::expected` conventions (`and_then`, `transform`, `transform_error`, or `or_else` as appropriate). Map errors to CLI diagnostics/exit codes once at the host boundary. (2026-09-18: software twin chains `prepare_depth_blend(...).and_then(execute_software).and_then(write_png)` and attaches its diagnostic via `or_else`; Vulkan twin has one `fail()` used by every stage; exit codes 1/2/3 derive from the stage)
- [x] Test success and failure at each stage: later stages do not execute after failure, original errors survive, and acquired resources are released. Short-circuiting does not imply rollback of external effects. (2026-09-18: `t0_composition_tests`, 91 checks, GPU-free — spy-counted short-circuit with zero later-stage runs, `or_else` keeps the original error, failed PNG write leaves the descriptor count unchanged, and execution refuses unvalidated hand-built plans)

**Acceptance:** tests exercise a typed chain, not just renamed calls or a draw vector. No driver/file I/O enters pure preparation; no borrowed plan payload outlives its owner.

**CLOSED 2026-09-18.** Evidence: [`adventure_demo_ad2_ad3_evidence_2026-09-18.md`](adventure_demo_ad2_ad3_evidence_2026-09-18.md).
Gate `t0_composition_tests` (91 checks) plus the unchanged six-pair baseline
table and full CTest 71/71. The pure zone is a separate header with no
framebuffer/rasterizer/PNG/Vulkan include, the plan moves the geometry into
itself, and a mutation probe (executor ignoring `pass.policy`) failed the gate,
which is why the gate asserts on the *executed* result. Demo 03 is the pilot;
the other five pairs convert under AD7.

## AD4 — Independent known-answer tests

- [x] Add always-active checks that fail the test process (not `assert` alone, which may disappear in release builds). Return diagnostics to the test/host edge rather than printing inside pure kernels.
- [x] Demo 01: check interior barycentric color against analytically derived weights at the sampled pixel center, with an explicit quantization tolerance.
- [x] Demo 02: check selected transformed coordinates and depth outcomes independently of twin image agreement.
- [x] Demo 03: check opaque overlap and alpha composition; inspect depth storage to prove transparent draws do not write depth.
- [x] Demo 04: check known nearest/bilinear samples, repeat wrapping, and unchanged pixels outside the scissor.
- [x] Demo 05: check stencil storage and equal/inverted-mask behavior at interior/exterior sample points.
- [x] Demo 08: check decoded normals, tangent-frame behavior for the supported input, and analytical flat/mapped shading samples. Do not assume one half is always brighter.
- [x] Prove checks fail with deliberately wrong expected values or temporary mutations; a correlated SW/Vulkan mistake must not pass solely through parity.

**Acceptance:** every lesson has independent numerical assertions with justified tolerances; tests run in GPU-free CI. Document input restrictions separately from guarantees of a general rasterizer.

**CLOSED 2026-09-18.** Evidence and input restrictions:
[`adventure_demo_ad4_evidence_2026-09-18.md`](adventure_demo_ad4_evidence_2026-09-18.md).
Artifacts: `tier0-.../tools/t0_known_answer_checks.cpp` (+ shared
`common/adventures_stb_load.cpp` PNG reader) and
`tier1-classic-shading/08_normal_mapping/normal_mapping_ka.cpp`; CTest entries
`t0_known_answer_checks`, `t0_known_answer_prove_fail`,
`t1_08_known_answer_checks`, `t1_08_known_answer_prove_fail`. Demos 01/04/08 are
checked against analytic oracles over the real software demo PNGs; 02/03/05 run
in-process through the shared tier0 `SwRaster` kernels so depth/stencil
**storage** is inspected directly. Validation found and fixed five oracle
defects (28 mismatches → 0); prove-fail now corrupts both oracle styles and
requires detection. Full `build/` CTest 67/67, `check_kdba_boundaries.sh` and
`check_include_graph.py` green. No parity agreement is accepted as evidence.

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

Each delivery should be independently reviewable. Preserve tolerance envelopes unless a separately investigated rendering correction justifies a change. Current status (2026-09-18): AD0, AD1, AD2, AD3, and AD4 are **closed** with
recorded evidence (`adventure_demo_baseline_2026-09-18.md`,
`adventure_demo_ad2_ad3_evidence_2026-09-18.md`,
`adventure_demo_ad4_evidence_2026-09-18.md` and the CTest gates); AD5, AD6, and
AD7 remain unimplemented. Next in this track: **AD5 + AD6** (single-source shared
inputs, then the execution-format adapters), leaving AD7 to roll out AD3's
preparation/execution/error shape to the remaining five pairs.
