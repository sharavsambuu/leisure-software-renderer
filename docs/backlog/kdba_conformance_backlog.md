# KDBA Conformance Backlog — shs-renderer-lib

> **Current status (2026-09-17): Run A–C complete; library migration stabilized.** The reassessment below was applied during those runs; it is not a new migration queue. Run C evidence and scalability contracts landed in commit `05bb646`. Remaining library work is P6.1–P6.3, standing P4.4, and trigger-scoped S1–S6. Adventure-demo AD0–AD7 is a separate consumer backlog. Historical audit findings and intermediate plans below do not override the close-out.

> **Historical reassessment directive (2026-09-17).** This audit predates the [governing clarification](../spec/dod_ecs_architecture.md). Findings below are candidates, not newly approved migration work. K1.1–K1.5, K5.1, and K6.1 must be re-evaluated against actual contracts and callers: a `void` signature, identity gateway, or wrapper alone does not prove a defect. Do not invent vacuous errors or run signature-only ports. Boundary, lost-fact, determinism, and lifetime findings still require evidence and tests. Blocked P6.x and standing P4.4 work remain tracked; nothing is completed or dropped by this notice. This notice supersedes conflicting status and run-plan language below.

> Status: active (2026-09-16). Source: full audit of `include/shs/domains/` (11 pods) against Constitution II (Kleisli Domain Boundary Architecture) + the 2026-09-16 hardening amendments (gateway-gateway terminology, Rule 4.1 drain-order law, ERROR_FLOW failure-rail catalog + drift gate).
> Trigger: user directive — current state is the old gateway-based Domain Value Object architecture (the retired term was used in the original directive); audit the lib against KDBA laws and register every violation candidate.
> Law precedence: Constitution II S2.1 + Rules 2/10/11/12, canon S6.1-S6.2, precedence S2.2. Roadmap is schedule, not law.
> Provenance: supersedes `../outdated/domain_pod_hardening_backlog.md` (FROZEN same day, see its banner; archived 2026-09-17). Blocked P6.x items and standing laws roll forward here; nothing was silently dropped.
> Verification after every item: `build/` ctest suite green + `check_kdba_boundaries.sh` green (now with the ERROR_FLOW drift gate + final-enforcement exit guard).

## Backlog ownership and historical dispositions (2026-09-17)

- This file owns the current status of library P4.4, P6.1–P6.3, and S1–S6.
  P4.4/S1 are one stochasticity workstream; P6.2/P6.3 and S5 share persistence
  work, not separate codec implementations. Headless S5 preparation can proceed
  while host integration remains blocked.
- The [hardening backlog](../outdated/domain_pod_hardening_backlog.md) remains frozen (archived 2026-09-17).
  Its P4.4/P6 checkboxes are historical references, not duplicate assignments.
  Its unchecked **L1 renderpath uniform-signature migration** is superseded by
  completed K1.2/Run A and the Run C reassessment; do not schedule another port.
  Frozen DoD checkboxes are not evidence of new work without a current audit.
- The [migration plan](../outdated/kdba_kleisli_migration_plan.md) records intermediate
  decisions. Current Constitution II, pod contracts, and this close-out take
  precedence over its historical signature descriptions.
- Roadmap entries require a current implementation/dependency check before
  scheduling. Checkbox totals are not unique-task totals; absence of checkboxes
  does not establish completion. Performance work requires measured evidence,
  and S1–S6 retain their explicit triggers rather than authorizing speculative
  infrastructure.

## Consumer-driven renderer feature delivery (2026-09-17)

Priority: finish a minimal value-plan-to-pixels execution slice before expanding
visual effects or introducing performance infrastructure. These execution tasks
do not reopen the completed domain migration. No new pod is required merely to
wrap Vulkan ownership. Fallible preparation uses typed results; GPU effects and
resource lifetimes remain at the execution edge.

### GPU execution path — selected next work

- [x] **G0 Stable graphics pipeline-ID lookup** — DONE 2026-09-17: explicit
  ID-to-hash index fixes `VulkanPipelineCache::find_graphics`. Regression failed
  before the fix and passed afterward; deduplication, distinct/unknown IDs and
  failed creation/retry are covered. Driver target rebuilt; registered CTest
  suite 16/16 passed including the boundary checker. This proves bookkeeping,
  not GPU pipeline binding; no live GPU rendering was exercised. Known
  asymmetry: `intern_compute` shares the same ID/hash distinction but exposes no
  `find_compute`; when compute ID lookup is needed, reuse the proven ID→hash
  index pattern instead of keying a hash map by ID.
- [x] **G1 Explicit recording failures** — DONE 2026-09-18: replace silent unsupported/no-device
  recording with a closed error vocabulary and command/stage diagnostics. Test
  unsupported commands, missing IDs and invalid recording order; preparation
  rejects invalid streams before GPU effects where possible. Do not claim GPU
  rollback or treat headless bookkeeping as successful rendering.
  Partial progress (2026-09-17): `record_frame_commands` now returns
  `std::expected<void, VulkanRecordingFailure>` with an error code and zero-based
  command index (`SIZE_MAX` for prerequisites). It rejects an unavailable device
  or missing command buffer and propagates sink failures. Headless tests cover
  empty/nonempty streams returning `DeviceUnavailable`, unsupported pass/pipeline/
  draw/dispatch rejection and translation stopping at the first rejected command.
  Buffer binds now reject missing/null handles instead of silently doing nothing.
  The command-buffer and missing-buffer branches still need dedicated coverage.
  Nested-pass preflight now scans the whole stream before any sink calls and
  reports `InvalidRecordingOrder` at the second begin-pass command. Its regression
  failed at runtime before the fix and passed afterward (including a leading
  barrier that must not reach the sink). Each stream starts outside a pass.
  Other whole-stream preflight, broader ID checks, remaining recording-order
  validation and stage diagnostics remain open. Sink failures are still fail-fast,
  not transactional: earlier calls are not rolled back. Spy/headless tests prove
  contracts, not GPU rendering.
  Batch progress (2026-09-17): whole-stream `validate_command_order` preflight
  rejects unmatched end-pass (`InvalidRecordingOrder`), unterminated passes
  (attributed to the opening begin-pass command), draws outside a pass and
  dispatches inside a pass; draws require pipeline + index bindings per call
  (`MissingBinding`, no cross-call/cross-pass leakage), dispatch requires a
  pipeline; binds reject zero IDs; index offsets are alignment-checked
  (`InvalidCommand`); in-pass barriers are rejected as unsupported; barrier
  stage/access enums are range-checked. `VulkanCommandRecorder::validate_command`
  is a pure resource/capability preflight: begin-pass targets resolve against
  the image pool (`MissingImage`), pipelines resolve through the pipeline cache
  by pass context (`MissingPipeline`, incl. graphics/compute kind mismatch),
  binds resolve buffer handles (`MissingBuffer`); unsupported operations still
  report `UnsupportedCommand`. Failures carry `stage`
  (Prerequisite/Validation/Recording), `command` kind and `resource_id`
  diagnostics. `record_commands` checks optional `recording_ready` and
  `validate_command` sink hooks before any recording call. New error codes:
  `MissingPipeline`, `MissingImage`, `MissingBinding`, `InvalidCommand`.
  `VulkanPipelineCache::find_compute` added (ID→hash index, same pattern as
  `find_graphics`). Tests: table-driven valid/invalid stream suite (exact codes,
  indices, stages, sink zero-call checks), per-position fail-fast sink table,
  resource-preflight table through a validating spy, and a real-device
  prerequisite suite (`shs_renderer_vk_recording_prerequisite_tests`, skips
  explicitly when Vulkan is unavailable; exercises device-unavailable,
  command-buffer-unavailable, null/missing buffer handles and whole-stream
  rejection before recording, with no submission). Documented limits: pass/pipeline
  realization and in-pass execution remain G2; `set_command_buffer` is a borrowed
  handle whose recording state is caller-managed. Fail-fast semantics unchanged;
  earlier calls are not rolled back. Spy/headless tests prove contracts, not GPU
  rendering.
  Tail closure (2026-09-18): the command-buffer prerequisite branch is now
  proven for non-empty streams in both suites — a stream that would fail order
  preflight still reports the prerequisite failure (correct stage, index
  `SIZE_MAX`, command `None`) before any stream inspection or sink call,
  headless (`DeviceUnavailable`) and on a real device
  (`CommandBufferUnavailable`). A spy with a failing `recording_ready` hook
  proves prerequisite precedence over whole-stream validation — including
  streams that would fail order preflight — with zero sink calls issued.
  Missing-buffer branch coverage (null/unknown IDs through direct calls and
  `record_commands`, exact index/stage/resource_id attribution) and
  order-validation breadth (nested/unmatched/terminated passes, binding
  leakage, index alignment, barrier stage/access ranges) were already
  table-pinned; the enum range checks are complete for the closed
  `RHIPipelineStage`/`RHIAccess` sets. Fail-fast non-transactional sink
  semantics stand; headless/spy evidence only; pass/pipeline realization and
  in-pass execution remain G2, factory-facing execution G3.
- [x] **G2 Minimal offscreen graphics realization** — implement actual attachment
  setup, pipeline creation/binding and begin/end-pass recording in the new driver.
  Acceptance: render one deterministic scene through value commands; supported
  formats/layouts/features are explicit, failures release acquired resources,
  and Vulkan validation reports no errors on an available backend. Depends on G1.
  — DONE 2026-09-18: all G2 acceptance criteria are evidenced end-to-end. One
  deterministic scene runs through value commands to known RGBA8 pixels:
  attachment realization (slice 1), pipeline realization + in-pass binding with
  cache identity (slice 2), non-indexed draw recording (slice 3), and backend
  submission/readback verifying independent known pixels. The support set is
  explicit (`supports()` tables on pass and pipeline), failed creation unwinds
  (including injected Vulkan allocation/pipeline/submit/map faults), and
  validation-enabled lavapipe runs report no errors. The same scene also drives
  the library SW/Vulkan triangle-parity evidence. No code gap remains against
  the stated acceptance; the staging→device-local upload tail (G3, landed
  2026-09-18 below) and factory-facing execution (G3) are the follow-on tracks.
  — PARTIAL 2026-09-17 (slice 1, `b2d7d79`): new
  `execution/rhi/drivers/vulkan/vk_offscreen.hpp` — explicit (non-lazy) owned
  (Path staleness recorded 2026-09-18: that realization now lives at
  `shs/rhi/vulkan/value/vk_offscreen.hpp`; the `execution/rhi/drivers/` spelling was
  retired with the R5 forwarder sweep — commit `8b6484b`.)
  realization of a narrowly-supported attachment config: RGBA8_UNorm 2D, 1
  mip/layer, ColorAttachment usage, LOAD_OP_CLEAR (transparent black), STORE,
  no depth, single subpass with external serialization dependency;
  `supports()` makes the accepted desc set explicit, failed creation unwinds
  acquired objects, re-prepare while live is refused, `reset()` destroys
  view/render pass/framebuffer (caller guarantees no pending references).
  `VulkanCommandRecorder` grew an optional `VulkanOffscreenPass*`: begin/end
  pass now record real `vkCmdBeginRenderPass/vkCmdEndRenderPass` (Vulkan 1.1
  render passes — no new device extensions beyond the existing bootstrap);
  begin validates the desc against the pass (`accepts`: matching color target
  id, no depth target, clear_color, no clear_depth), guards nesting
  (`InvalidRecordingOrder`), and everything else remains `UnsupportedCommand`
  (bind/draw/dispatch included — pipeline realization is the next slice).
  Test `shs_renderer_vk_offscreen_pass_tests` (real device, lavapipe locally):
  supports() negative table, view/pass/framebuffer creation on a real image,
  failed-prepare unwind, double-prepare refusal, accepts() table, stream
  rejection before any recording (validation stage, exact index), real
  begin/end recording twice through `record_commands`, nesting guards,
  explicit reset/recreate. NO submission, NO readback, NO pixel claim — G2
  remains open for pipeline realization, vertex upload and the deterministic
  scene; submission/readback evidence stays G3.
  — PARTIAL 2026-09-17 (slice 2): `VulkanOffscreenPipeline` explicitly owns
  a graphics pipeline/layout, creates and releases temporary shader modules,
  and unwinds Vulkan creation failures. Fixed ABI: triangle list, no vertex
  attributes/descriptors/push constants, RGBA8, no depth/blending, static
  viewport/scissor. Caller supplies trusted validated SPIR-V matching the ABI;
  `supports()` is a descriptor/header gate, not shader reflection/validation.
  The recorder now binds realized graphics pipelines inside a pass, requiring
  matching cache ID/hash, device and extent. Cache-only or retired realization
  remains UnsupportedCommand; missing IDs retain G1 MissingPipeline precedence.
  Cache identity does not include viewport extent: callers must explicitly
  re-realize after reset when changing extent, never treat cache hits as owners.
  Existing offscreen CTest also exercises real creation/binding, cache hits,
  negative descriptors, unsupported SPIR-V version, double-prepare refusal and
  reset/recreation. Slang fixture builds as SPIR-V 1.3 for Vulkan 1.1, using
  SV_VulkanVertexID to avoid the unenabled DrawParameters feature emitted by
  SV_VertexID. No bootstrap feature/extension changes. Without slangc the test
  explicitly reports attachment-only coverage; missing Vulkan returns skip 77.
  Local lavapipe evidence: pipeline creation and two bind streams, validation
  enabled with no reported errors, plus full CTest 18/18. No draw, submission,
  readback or pixel claim; G2 scene acceptance remains open. Factory-facing
  backend attachment/pipeline lifecycle integration is not yet wired. Creation
  failure unwind is implemented but Vulkan allocation failures are not injected.
  — PARTIAL 2026-09-17 (slice 3): added non-indexed `RHICmdDrawDesc`
  and value-command translation, preserving existing variant indices and sinks
  without draw support (explicit UnsupportedCommand). The Vulkan recorder requires
  an active pass and a bound, still-compatible realization. Pass boundaries clear
  binding state. The offscreen test records a deterministic procedural triangle;
  negative ordering/missing-binding streams fail preflight before recording.
  Full build and 18/18 CTest passed. Factory-facing attachment/pipeline lifecycle
  integration remained open at G1 close and no backend-owned execution API was
  claimed then; G3 closed both on 2026-09-18 (see the G3 entry below).
- [x] **G3 Upload, submit and readback proof** — complete the minimal scene's
  buffer upload, synchronization, submission and image readback. Check known
  pixels independently of parity; exercise failure and resource cleanup paths.
  Record unavailable Vulkan capability as a skip, never a pass. Depends on G2.
  — CLOSED 2026-09-18 (factory-facing execution, the last open slice). The gap
  was real: consumers obtained the backend from `create_render_backend()` and then
  `dynamic_cast`ed back to `VulkanRenderBackend` to open a device and submit.
  `IRenderBackend` now exposes an optional, vendor-free `IOffscreenExecution*`
  (`shs/rhi/core/offscreen_execution.hpp`; default `nullptr` = fall back or skip,
  never a pass) implemented by the value-tier Vulkan backend through a
  composition adapter, so the minimal scene runs from the factory plus
  `app::Context` with no downcast and no Vulkan type in the consumer. Portable
  gate `shs_renderer_vk_factory_offscreen_tests` asserts known pixels derived from
  the authored scene (not parity), rejection, and reset/re-prepare, and reports an
  unavailable surface or device as SKIP 77. Previously landed slices
  (submission/readback, shutdown cache-invalidation, vertex/index upload incl.
  staging→device-local, failure injection, triangle parity) are unchanged. Full
  CTest 68/68; boundary and include-graph gates green; header inventory
  regenerated. Evidence:
  [`kdba_g3_factory_facing_evidence_2026-09-18.md`](kdba_g3_factory_facing_evidence_2026-09-18.md).
  Asynchronous retirement remains unclaimed.
  — PARTIAL 2026-09-17: existing offscreen integration test now submits real
  commands, waits on a fence, transitions the RGBA8 attachment to transfer source,
  copies to host-visible staging memory with a host-read barrier, and verifies
  independent known pixels: triangle `(255,64,0,255)`, background `(0,0,0,0)`.
  Lavapipe run with validation enabled passed without reported validation errors.
  This is driver-level evidence, not factory-facing execution. The procedural
  scene does not upload vertex/index buffers; upload proof, reusable backend
  submission/readback, injected Vulkan failures, and software parity remain open.
  Backend shutdown cache invalidation also needs coverage before device recreation
  is exposed: existing registries currently retain retired resource records.
  — PARTIAL 2026-09-17 (backend integration): `VulkanRenderBackend` now exposes
  explicit `prepare_offscreen` / `execute_offscreen` / `reset_offscreen` for the
  fixed procedural ABI. The driver owns command pool/buffer and host-visible
  staging allocation; submission waits on a fence and readback invalidates the
  mapped allocation (including noncoherent memory). Tests compare the complete
  backend output against the independent driver fixture and known pixels, repeat
  execution, reject malformed streams/output sizes, recover after rejection,
  resize, and shutdown/recreate. Shutdown clears resource/pipeline lookups without
  recycling ID counters. An isolated real-device regression failed on the old
  shutdown implementation and passed with the fix; buffer/image IDs increase
  across recreation. Full build and validation-enabled lavapipe CTest: 18/18.
  This closes reusable backend submission/readback and shutdown-invalidation
  coverage for this synchronous slice, not all G2/G3 acceptance. Images remain
  registry-owned until shutdown; reset retires attachment/pipeline/transfer owners.
  Still open: vertex/index upload and consumed-geometry proof, injected Vulkan
  allocation/submission failures, broader preparation failure cleanup, and G4 —
  each subsequently closed (see the later PARTIAL entries and the G4 CLOSED
  entry above).
  The concrete Vulkan backend API is tested; generic factory-interface execution
  and asynchronous retirement are not claimed.
  — PARTIAL 2026-09-17 (upload + injection slice): explicit `RHIVertexLayout`
  (Procedural default, Position2F) in `RHIGraphicsPipelineDesc`; hashed into the
  pipeline cache key alongside the fragment entry name. `VulkanOffscreenPipeline`
  realizes location/binding-0 float2 vertex input when Position2F is selected.
  `VulkanRenderBackend::upload_buffer` writes full-size CPU-visible buffers
  (map/unmap; requires explicit CPUVisible memory class; retries after a failed
  map succeed). Backend execution preflights whole streams before recording:
  bind/draw-index ranges are checked against a CPU-side shadow of uploaded bytes
  (`InvalidCommand`/`MissingBinding` at the exact command index, output left
  untouched). Indexed draws record `vkCmdDrawIndexed` with pass-scoped index
  binding state. Regression evidence: uploaded triangle pixels equal the
  independent procedural fixture exactly; degenerate index lists consume both
  buffers and clear the image; moved vertices move the triangle; out-of-range
  indices and unuploaded buffers are rejected preflight with untouched output.
  Test-only `vk_failure_injection.hpp` interposes real Vulkan calls (memory
  allocation, pipeline creation, fence, queue submit, map) to prove typed
  failures and full cleanup/retry: failed pipeline creation unwinds the
  prepared target, failed upload leaves buffers usable, failed submission
  preserves the previous output and the transfer pool is reusable. Full build
  and validation-enabled lavapipe CTest: 18/18. Staging→device-local copy upload
  and factory-facing execution remain open G3 items.
  — PARTIAL 2026-09-17 (triangle parity): the same minimal triangle runs through
  the library software rasterizer (`rasterize_mesh`, cull None, clear {0,0,0,0})
  and the Vulkan backend: identical vertices, indices, flat fragment color
  {1,0.25,0,1} and clear policy. Away from triangle edges RGBA8 output is equal
  exactly; differences are coverage-only, within a fixed one-pixel edge band
  caused by the software rasterizer mapping NDC to (extent-1) while Vulkan uses
  extent (15 differences on the 32×32 fixture; per-pixel alpha-transition check,
  distance-to-triangle bound, and a cap on total differing pixels). Independent
  known-answer checks (software interior `(1,0.25,0,1)`, background transparent,
  raster stats) are asserted alongside the comparison. This was recipe-level
  evidence for the fixed offscreen ABI, not full G4: single triangle, no
  depth/motion paths, no portable CTest gate beyond the existing target. The G4
  CLOSED entry above supersedes it — the same CPU rasterizer is now reached
  through the generic library execution path, with the comparison wired as a
  portable gate whose tolerances are documented and whose 15-pixel boundary band
  this run independently reconfirms.
  — PARTIAL 2026-09-18 (staging→device-local upload): `upload_buffer` now accepts
  `RHIMemoryClass::GPUOnly` buffers carrying TransferDst usage: a transient
  host-visible staging buffer is created and bound, a one-time command buffer
  records a full-size `vkCmdCopyBuffer` plus a transfer→vertex-input buffer
  barrier, the copy is submitted behind a fence, and every transient object is
  released on success and on every failure path. Missing TransferDst usage,
  wrong size, empty bytes and unknown IDs are rejected before any copy; the
  CPU-side shadow updates only after a completed copy. Real-device evidence
  (lavapipe, validation enabled): a GPUOnly vertex/index pair renders the same
  known pixels as the CPU-visible fixture, and a re-upload moves the consumed
  triangle. `vulkan_buffer_upload_sync` lives in `vk_readback.hpp` (the
  synchronous-transfer owner). Factory-facing execution was the last open G3 item;
  closed 2026-09-18 — see the G3 CLOSED entry above.
- [x] **G4 Library SW/Vulkan equivalence** — run the same minimal scene/policy
  through actual library execution paths with documented per-output tolerances
  and independent known-answer checks. Wire portable CTest gates and retain
  backend diagnostics. Adventure AD1/AD4 are related, not substitute evidence.
  Depends on G3 and a verified software realization of the selected recipe.
  **CLOSED 2026-09-18.** The software side now *realizes* the generic contract
  instead of declining it (`SoftwareOffscreenExecution`: no device to open,
  descriptor-derived stable ids, entry-name-bound CPU realization) and both
  realizations share one vendor-free descriptor gate
  (`rhi_graphics_pipeline_desc_supported`) so acceptance cannot drift between
  them. The new portable gate `shs_renderer_sw_vk_equivalence_tests` runs ONE
  consumer function — `create_render_backend()` → `app::Context` → generic
  `IRenderBackend` → generic `IOffscreenExecution` — against both backends with
  the same command stream, checks each readback against the **authored** scene
  independently (interior `(255,64,0,255)`, clear `(0,0,0,0)`), and only then
  compares them under two documented tolerances (≤ 1/255 per channel where both
  cover; coverage budget 16 at 32×32, GPU may only add coverage). Measured:
  `cpu_covered=113 gpu_covered=128 mismatches=15 (both=0 gpu_only=15 cpu_only=0)`
  — the same 15-pixel boundary band the 2026-09-17 parity work found
  independently, now attributable to the `(W-1)/(H-1)`-vs-`w/h` screen-mapping
  convention gap. The software half is always asserted; only the Vulkan half may
  skip (77, with the equivalence claim explicitly *not* made). Two stale gate
  comments were corrected (an `edge budget 8` comment against a 16 assertion; a
  transposed known-answer pixel triangle). Full CTest 69/69; boundary,
  include-graph, self-containment and package-consumer gates green; header
  inventory regenerated (225 → 226). Evidence:
  [`kdba_g4_sw_vk_equivalence_evidence_2026-09-18.md`](kdba_g4_sw_vk_equivalence_evidence_2026-09-18.md).
  Not claimed: the CPU realization is bound to the authored recipe by entry name
  (not general SPIR-V portability); the software accepted set is a strict subset
  of Vulkan's (no generic buffer surface yet, so no bindings/indexed/instanced
  draws); the coverage tolerance is calibrated for this fixture; and the gate is
  not compiled in a GPU-free build.

### Follow-on features — existing ownership preserved

- **Live plan switching, resize and safe GPU retirement:** owned by P6.1;
  accepted plans rebuild execution resources, rejected plans preserve the working
  renderer, in-flight resources outlive submission. Offscreen G2/G3 do not wait
  for window presentation, and do not mark host integration complete.
- **Persisted capture/replay and time-travel debugging:** owned by P6.2/P6.3 + S5;
  versioned codecs, asset/config identity, handle reconstruction and snapshot/log
  resume precede an overlay. Headless preparation remains independently possible.
- **Library shader pipeline:** owned by the
  [Slang plan](../roadmap/slang_utilization_plan.md); verify existing phase status
  before scheduling compiler gates, shader identity, reflection-checked layouts
  and bindings. Adventure Slang usage alone does not complete this integration.
- **Optimization and additional effects:** deferred behind consumer demand and
  measured evidence; no speculative scheduler, graph or meshlet framework.

## Related demo work (2026-09-17)

The [Adventure Demo Domain Boundary & Composition Backlog](adventure_demo_conformance_backlog.md)
tracks AD0–AD7 for the six active adventure-demo pairs: shared semantic ownership,
typed composition, known-answer tests, execution adapters, and portable gates.

Status (2026-09-18): **AD0, AD1, AD4 closed** with recorded evidence; AD2, AD3,
AD5, AD6, AD7 open. This is separate consumer work, not a reopening of Run C or
completion of the library P6/S1–S6 integration and scalability tasks, and demo
evidence never substitutes for the G4 library gate (itself closed 2026-09-18 —
see the G4 CLOSED entry above).

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

## Run C close-out (2026-09-17)

- K1.5 reassessed: identity gateways are legal, not signature defects. Retain
  their existing plain Step summaries as explicitly tested per-batch counts;
  no universal return-type requirement or vacuous error enum is introduced.
  All eight identity suites pin counts, state/event preservation, replay, and
  empty batches. Input has real intents and its own Step/fact tests, not the
  monostate helper. The current 11-pod shape register is a drift guard only.
  **Amended 2026-09-17 (domain-separation migration step 4.5):** the seven
  pure-identity gateways (camera, geometry, gfx, lighting, resources, scene,
  sky) were retired as dead scaffolding — empty vocabulary, no applied state,
  no production callers — superseding the "retain" wording above for those
  pods; `frame` keeps its gateway (identity transition over real
  `FrameParams` state, the C1.4 replay-probe vehicle). Amendment record in
  [pod_identifier_law.md §2.7](../spec/pod_identifier_law.md); the boundary
  gate now fails on identity-gateway regrowth.
- K3.1/K3.3: per-pod transition/compensation inventory is in
  [ERROR_FLOW.md](../pods/ERROR_FLOW.md). Renderpath is the sole closed error
  family; logic rejects with typed facts. All three renderpath setters now
  emit change facts only on accepted swaps, with red-to-green rejection,
  prefix, replay, empty-batch, and retry regression coverage. Domain rejection
  safety is not an allocation-exception or general undo guarantee.
- K3.2: same-state logic signal/force/tick facts are emitted and tested.
- K5.2: source scan found no callback FSM consumers or AssetRegistry class
  forks; legacy FSM headers remain deleted. PluggablePipeline/FrameGraph are
  retained in execution: the former owns the latter and core tests still
  exercise the pipeline. Gateway value summaries do not replace executor
  ownership/lifetimes; P6.1 remains the retirement/rebuild prerequisite.
- K6.2/K6.3: temporary-copy probes for discriminator switches and silent
  `continue;` each exit 1 at the intended gate; an uncatalogued error-enum
  probe also exits 1. Baseline/restored copies exit 0. These regex gates
  recognize known syntax, not all semantic dispatch or lost-fact defects.
- Validation: clean build succeeded in `cpp-folders/build`; after the final
  renderpath fix, full rebuild and unfiltered CTest passed **16/16**, including
  the boundary checker. Run C and the scalability contracts were committed as
  `05bb646` (2026-09-17). P6.1–P6.3 host integration remains blocked;
  P4.4 remains standing. This reassessed close-out supersedes the historical
  signature-only/vacuous-error Run C plan below.

## Audit summary (2026-09-16)

Checked against KDBA laws: Kleisli house signature (`expected<Step{NextState, Events}, ClosedEnumError>`), switch-monolith ban (Rule 2 as amended — gateway = gateway, never `switch(action.type)`), zero-signal-loss, closed error vocabularies + failure-rail catalog, phantom/validity-flag ban, named-field events (no positional bools), one public gateway per pod, purity/edge laws.

**Clean:** banned tokens (shared_ptr/dynamic_cast/mutex/function: 0 non-edge hits), SDL/fopen in domains (0), ambient entropy (0), unordered_* in gateways (0), cross-domain includes (only legal directions: camera->geometry/aabb, renderpath->frame/technique_mode — both sanctioned seams), event/enum catalog drift (both gates green).

**Violations found:** every one of the 11 pod gateways uses the retired writer shape; one literal `switch(action.type)` monolith (input); one phantom validity flag; one positional-bool event; silent signal drops in logic; a dual-gateway pod; a discard-all stub gateway; namespace/Inputs drift in logic. Details per workstream below.

## W1 — Kleisli gateway migration (flagship, phased per pod)

- [x] **K1.1 Migration law note + per-pod port plan** — DONE 2026-09-17 (Run A): [`kdba_kleisli_migration_plan.md`](../outdated/kdba_kleisli_migration_plan.md) publishes the shared `Step`/gateway vocabulary, the port order, and the batch-vs-per-command spike decision; renderpath landed with kit-extended tests (replay + empty-log + value-equality `operator==` on state/events) proving the signature swap is behavior-neutral. — was: publish the port order and the shared `Step`/gateway vocabulary before touching any gateway, so the 11 migrations are mechanical copies of one proven shape, not 12 ad-hoc designs. Pilot = renderpath (its `try_swap_plan` is already an arrow chain: `and_then`/`or_else` over `expected`, only the wrapper is `void` — smallest delta to full house shape). Port order: renderpath -> logic -> frame -> geometry -> lighting -> sky -> scene -> resources -> gfx -> input (input last: biggest monolith, needs W2 first).
- [x] **K1.2 renderpath** — DONE 2026-09-17 (Run A): `renderpath_gateway` now returns `RenderPathStep{commands_applied, noops_observed, swaps_rejected, plan_generation}` by value; events stay on the caller's arena (A.7 divergence honored); transition bodies moved to named per-intent `apply_*` arrows (K2.2 renderpath half); `try_swap_plan` returns its outcome over the unchanged per-command `expected` rail. **Reassessment verdict (banner):** the audit's literal "`expected` at the batch rim" was REFUTED — every real failure is a compile rejection absorbed by the per-command rail and materialized as `PATH_SWAP_REJECTED` (previous plan kept), so a batch-level error enum would be invented/vacuous (ERROR_FLOW non-vacuity law); evidence + decision in the plan doc. Unlocks the L1 leftover from the frozen backlog as stated.
- [x] **K1.3 logic** — DONE 2026-09-17 (Run B): `logic_gateway` + the full `Fsm*` vocabulary moved to `shs::logic` (zero external consumers); dt moved from `FsmTick` to `LogicContext` (one dt per batch, input-parity); gateway returns `FsmStep{commands_applied, facts_observed, commands_rejected}` (batch rim infallible per the Run A decision — no invented error enum); dispatch is `std::visit` + `if constexpr` over named `apply_*` arrows; silent `continue` drops now emit facts (K3.2). — was: `logic_gateway` (logic.gateway.hpp:86): writer shape + `FsmInputs` empty + dt lives on `FsmTick` action instead of Inputs (L125-129) while input pod takes dt from Inputs — inconsistent time placement across pods. Port unifies: time in Inputs, gateway returns `expected<FsmStep, FsmError>`; silent `continue` drops become observable (K3.2). Namespace `shs` -> `shs::logic`.
- [x] **K1.4 camera** — RESOLVED BY REASSESSMENT 2026-09-17 (Run B; verdict in `kdba_kleisli_migration_plan.md`): the "discard-all gateway" charge is vacuous — the camera pod's command/event vocabularies are `variant<monostate>` (§6.1-legal empty vocabularies, frame-pod precedent; identity pinned by kit tests), so no real signal can ever be dropped. Its contract seam (`CameraRig`, builders) is live code, so fold-delete would break real consumers; the input pod's camera math over the rig inside its OWN `RuntimeState` aggregate is intra-pod, not cross-pod mutation. Decision: neither absorb nor fold — full absorption waits for an orchestrator host (same blocker family as P6.1-P6.3); Run C's K1.5 sweep ports the identity shape mechanically. DoD met in the reassessed form: no gateway with a non-empty vocabulary discards commands. — was: `camera_gateway` (camera.gateway.hpp:32-42) discards ALL arguments (`(void)state; (void)actions; ...`): a gateway that silently eats every command, the trivial worst-case zero-signal-loss violation. Decide: real camera gateway absorbing the camera math that currently lives in input's MoveLocal/Look handling (input.gateway.hpp:45-68 reaches directly into `state.camera` — cross-vocabulary coupling), or fold camera vocabulary into the input pod and delete the stub.
- [x] **K1.5 The 8 identity pods (frame, geometry, gfx, lighting, sky, scene, resources, camera)** — RESOLVED BY REASSESSMENT 2026-09-17. Identity is legal; retain plain Step batch counts with explicit tests in every suite (including replay and empty batches). No invented error enum and no universal signature rule. See Run C close-out above.

## W2 — Kill the switch monoliths (Rule 2 as amended)

- [x] **K2.1 input** — DONE 2026-09-17 (Run B): the textbook monolith is dead. Root cause fixed at the vocabulary: `RuntimeCommand{kind, payload}` (double dispatch, desync-prone) converted to a PURE closed variant `variant<MoveLocalIntent, LookIntent, ToggleLightShaftsIntent, ToggleBotIntent, QuitIntent>` (the kind enum had zero consumers outside the pod; `RuntimeCommandKind`/`RuntimeCommandPayload`/`ToggleFlagIntent` deleted). `input_gateway` is now `std::visit` + `if constexpr` over named `apply_*` arrows, returning `InputStep{commands_applied}`. DoD met: `switch` on action discriminators in `*.gateway.hpp` = 0 (K6.2's gate lands with Run C). — was: `input_gateway` (input.gateway.hpp:30-93) is the textbook forbidden monolith the amended Rule 2 names: literal `switch (action.type)` with 5 arms + `std::get_if` payload fishing inside each arm (type+payload double dispatch, desync-prone). Decompose into per-intent gateway functions (`apply_move_local`, `apply_look`, ...) assembled by the pod's Kleisli gateway over the closed `RuntimeAction` variant (`std::visit` + `if constexpr`, renderpath style).
- [x] **K2.2 logic + renderpath dispatch audit** — DONE 2026-09-17 (Run B + Run A): both gateways now read as arrow-chain assembly with zero inline transition bodies — renderpath's per-intent `apply_*` arrows landed in Run A, logic's in Run B; the `std::get_if` ladder is gone from logic (variant dispatch). — was: both use variant ladders living inline in the public gateway (`std::get_if` chains in logic.gateway.hpp:96-133; `std::visit` + `if constexpr` in renderpath). Legal under Rule 2's letter (dispatch on the closed variant, not a parallel enum), but the transition logic belongs in named per-intent functions; the gateway is only the assembly point (amended §2.1 definition).

## W3 — Error-channel + zero-signal-loss conformance

- [x] **K3.1 Failure-rail inventory per pod** — DONE 2026-09-17: all 11 pods classified in ERROR_FLOW.md; only real error vocabulary retained; drift gate and temporary-copy negative probe green. Historical scope: — walk all 11 pods: for each transition, classify infallible / fallible-with-closed-error / currently-silent. Output: per-pod error enums (named `*Error`/`*Rejection`/`*Reason` per ERROR_FLOW.md law) + ERROR_FLOW.md rows. Seed data: renderpath already has the only real rail (`PathSwapRejectionReason`, 6 values, mapped 1:1 from the compiler enum at renderpath.gateway.hpp:55-67 — this stays the model). DoD: ERROR_FLOW.md covers every pod's error vocabulary; drift gate green.
- [x] **K3.2 Kill silent signal drops in logic** — `continue` sites consume a command and emit NOTHING: `!state.started` on signal (logic.gateway.hpp:111), no-rule-match on signal (L113) and on tick (L131). Under zero-signal-loss these are invisible failures. Emit `FsmSignalRejected`/`FsmTickNoRule` facts or route through the gateway error channel — pick ONE house answer and mirror it in renderpath's silent no-op sites (renderpath.gateway.hpp:155, 166, 175: same-technique/same-mode commands return silently; document the decision as a constitution note, not folklore). **DONE 2026-09-17 (Runs A+B):** the house answer is **FACTS**. Renderpath half (Run A): the three silent no-op sites emit `TechniqueUnchangedEvent` / `ViewCullingUnchangedEvent` / `ShadowCullingUnchangedEvent`. Logic half (Run B): unstarted/no-rule consumptions emit `FsmSignalRejected` / `FsmSignalNoRule` / `FsmTickUnstarted` / `FsmTickNoRule` / `FsmForceUnstarted` (the force-while-unstarted drop the audit missed is closed too); pinned by `test_unchanged_facts` + `test_zero_signal_loss`. Run C closes the remaining same-state signal/force/tick exception with typed `*Unchanged` facts; `test_same_state_facts` pins payloads, counters, elapsed time, replay, and empty batches.
- [x] **K3.3 Compensator sweep (Rule 12)** — DONE 2026-09-17: per-pod audit and explicit scope limits in ERROR_FLOW.md; false renderpath change facts fixed and regression-tested; saga full rollback green. Historical scope: — only the saga spike test exercises compensation today. Once W1 lands, audit each multi-step transition for invertibility: every state mutation inside a gateway must have a recorded compensator fact or be provably idempotent. DoD: per-pod compensator note in ERROR_FLOW.md or an explicit "no multi-step flows" entry.

## W4 — State-shape hardening

- [x] **K4.1 `has_plan` phantom flag** — DONE 2026-09-17 (Run A): `RenderPathPodState::plan_generation` (`uint32_t`, 0 = none, bumped on every successful plan install; rejections and runtime toggles never bump it) replaces the validity bit; semantics pinned by `test_plan_generation_semantics`; `grep has_plan` in shs-renderer-lib = 0. Feeds P6.1's plan-hash executor rebuilds (precondition contributor). — was: `RenderPathPodState::has_plan` (renderpath.gateway.hpp:35) is a validity bit shadowing plan existence: state can lie (default plan + has_plan=false is indistinguishable from an empty-but-real plan). Replace with a generation counter (`uint32_t plan_generation = 0`, 0 = none) — value-honest, replay-friendly.
- [x] **K4.2 Positional-bool event** — DONE 2026-09-17 (Run A): split into `ViewCullingModeChangedEvent{previous, current}` / `ShadowCullingModeChangedEvent{previous, current}`; closed vocabulary stays closed (5 → 9 facts); EVENT_FLOW.md + `renderpath_event_name` table updated together; drift gates green. DoD met: zero unnamed-bool payload fields in renderpath events.

## W5 — Gateway uniqueness + legacy seams

- [x] **K5.1 input dual-gateway retirement** — DONE 2026-09-17 (Run B): `runtime_state_gateway` deleted from `value_commands.hpp` (now the pure command-emitter header with a retirement banner); lib consumers ported to the single public gateway (`edge/command_processor.hpp` drives `shs::input::input_gateway` directly; `core_tests.cpp` + `input_tests.cpp` updated). DoD met: single grep-visible gateway per pod. — was: input pod exposes BOTH `input_gateway` (house signature) and `runtime_state_gateway` (legacy by-value signature, value_commands.hpp:35-43, consumed by edge/command_processor.hpp:53). One pod, one public Kleisli gateway. Port the edge consumer, delete the legacy wrapper.
- [x] **K5.2 Grandfathered-architecture sweep** — DONE 2026-09-17: callback headers deleted with no source consumers, no AssetRegistry class forks, execution pipeline/graph retained with live consumers pending P6.1 (see close-out evidence). Historical scope: — parked items whose unlock conditions this directive supersedes: legacy callback StateMachine beside the value FSM (logic — "zero consumers" per P3.10; verify then delete), PluggablePipeline/FrameGraph retention re-check (P5.2 kept them pending P6.1; re-audit if W1 changes the calculus), AssetRegistry-class stale forks (deleted in P5.1; confirm none regrew).

## W6 — Mechanical drift gates (checker)

- [x] **K6.1 Kleisli-shape gate** — DONE 2026-09-17 (Run A): gate landed in `check_kdba_boundaries.sh` §(4). **Reassessment verdict (banner):** the original wording — gate `inline void reduce_*` — was obsolete (§6.6 already bans those tokens outright), so the gate targets the real regrowth vector, the writer SIGNATURE: FAIL on `void <pod>_gateway(` in a Kleisli-migrated pod, FAIL on any pod missing from the migrated/grandfathered registers (migrated: renderpath; grandfathered: the 10 Run B/C pods — P1.1 facade-case mechanism). — was: once the FIRST pod lands its gateway (K1.2), FAIL on any NEW `inline void reduce_*` added outside the migrated-pod list (grandfather list carried in the script, shs/pipeline facade-case precedent — same mechanism as P1.1). Prevents writer-shape regrowth during the phased migration.
- [x] **K6.2 switch-monolith gate** — DONE 2026-09-17; baseline and negative temporary-copy checks passed (see close-out). — FAIL on `switch` over action discriminators in `*.gateway.hpp`; lands with W2 completion (audit grep already proven).
- [x] **K6.3 Silent-drop gate** — DONE 2026-09-17; fact-based house answer and negative temporary-copy check verified (see close-out). — grep-gate the logic silent-drop pattern once K3.2 fixes it — ONLY if the team picks "events" over "error channel"; a gate on an undecided design choice is premature. Standing until K3.2 decides.

## Rolled forward from the frozen backlog (tracked, not dropped)

- [ ] **P6.1 PATH_COMPILED-driven executor rebuilds** — BLOCKED (needs a live demo host; unchanged).
- [ ] **P6.2 Replay harness (cross-session codec + CI replay)** — host integration remains BLOCKED on demo host; headless codec/fixture preparation can proceed independently (S5). K1.5 strengthens pod-level replay tests, not portable serialization: explicit codecs, versions and reconstruction rules remain unbuilt.
- [ ] **P6.3 Rollback snapshots + time-travel overlay** — BLOCKED on windowed host; K4.1's generation counter is a precondition contributor.
- [ ] **P4.4 Seeded determinism implementation (STANDING)** — baseline contract adopted in Constitution II §11.1; first stochastic pod selects/version-pins its algorithm and proves S1. No stochastic implementation is claimed complete.

## Future-domain scalability preparedness (2026-09-17)

Authority: [Constitution II §11.1](../spec/value_oriented_programming.md#111-future-domain-scalability-contracts-amendment-2026-09-17)
and Rule 12; event compatibility is mirrored in EVENT_FLOW.md. These are new
follow-ups, not reopened Run C migration items. Laws are adopted; the unchecked
implementation items below are not complete. Use existing PMR/SoA/span and test
facilities; do not build a general framework before a measured need exists.

- [ ] **S1 Seeded determinism proof (P4.4)** — Trigger: first stochastic pod.
  Pin algorithm/version, integer-to-sample mapping, seed/state/counter and
  stable stream assignment. DoD: known-answer vectors, snapshot/resume parity,
  replay and scheduling-order tests within a declared numeric/platform envelope.
  Existing P4.3 entropy gate remains; no duplicate gate. Add targeted positive
  and negative linter fixtures when refining its broad regex matches.
- [ ] **S2 Large-state headless spike** — Trigger: before the first high-volume
  mutable domain; available independently of a windowed host. Exercise 10k and
  100k entities in persistent SoA with preallocated chunk output/delta staging.
  Compare sparse/dense changes against a simple reference; measure allocations,
  bytes copied, latency and peak retained memory (report hardware/build settings,
  no invented universal time threshold). DoD: stable old snapshots, rejected
  batch preservation, stale-handle detection, arena reset/lifetime safety,
  capacity exhaustion and reclamation tests; deterministic results across chunk
  sizes/worker counts. Do not promote the disposable probe into shared machinery
  until the evidence justifies it.
- [ ] **S3 Streaming edge contract** — Trigger: first asynchronous IO/streaming
  domain. Specify bounded queue capacity, order/tick assignment, backpressure,
  loss reporting, duplicate handling and completion intents. DoD: burst/overflow,
  delayed/duplicate delivery and recorded-input replay tests; any coalescing
  proves semantic equivalence and preserves required facts.
- [ ] **S4 Time policy integration** — Trigger: next real-time host integration
  (P6.1). Default adopted: fixed simulation step, host accumulator, variable-rate
  presentation; domain-specific deviations require an explicit contract.
  DoD: identical authoritative state for the same logical inputs under differing
  presentation rates, pause/resume, catch-up limits and overload; record step
  configuration and input-to-tick assignment. Host rollout remains blocked.
- [ ] **S5 Replay persistence and retention (P6.2/P6.3)** — Trigger: before
  shipping a persisted session format. Headless preparation need not wait for
  GPU/window availability. Specify snapshot + command + external-input codec,
  stable IDs, version migration/rejection, asset/config identities, ordering,
  checkpoint cadence and bounded log/storage retention. DoD: round trips,
  historical fixtures, unknown versions, truncated/corrupt logs, handle remapping,
  and checkpoint/resume parity. Catalog completeness is not a codec test.
  Host replay and time-travel overlay remain blocked under P6.
- [ ] **S6 Production saga proof (Rule 12)** — Trigger: first multi-domain
  transaction. Preserve the existing in-memory saga regression; add failure at
  each stage, reverse-order compensation, repeated compensation and prefix
  preservation tests. For external effects, also specify/test idempotency,
  acknowledgement, retry, compensation failure and irreversible-effect recovery.
  DoD: persistent invariants survive each tested failure; do not describe
  external compensation as atomic rollback. Extract helpers only when reused.

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
  [`kdba_kleisli_migration_plan.md`](../outdated/kdba_kleisli_migration_plan.md).

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
  **Status: DONE 2026-09-17** — K1.3/K2.1/K2.2/K3.2/K5.1 landed; K1.4
  resolved by reassessment (verdict in
  [`kdba_kleisli_migration_plan.md`](../outdated/kdba_kleisli_migration_plan.md)).

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