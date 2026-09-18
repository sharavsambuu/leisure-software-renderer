# AD2 + AD3 — Shared pass policy and the typed-composition pilot (closed 2026-09-18)

Owner: `adventure_demo_conformance_backlog.md` AD2 and AD3. Evidence recorded per
their acceptance rules:

- AD2 — "both twins consume the same intended pass policy; previous draw
  configuration cannot leak into the next draw. Lesson kernels stay visible and
  baseline comparisons hold."
- AD3 — "tests exercise a typed chain, not just renamed calls or a draw vector.
  No driver/file I/O enters pure preparation; no borrowed plan payload outlives
  its owner."

## Reproduce

```sh
cd cpp-folders/build
cmake --build . -j"$(nproc)"
ctest -j"$(nproc)"                     # 71/71 (was 69/69 at the G4 close)
ctest -R 't0_policy_tests|t0_composition_tests' --output-on-failure
# fresh six-pair baseline comparison (GPU needed for the *_vk halves):
python3 ../src/exps-rendering-adventures/tier0-rasterization-foundations/tools/t0_parity_suite.py \
  --build-dir src/exps-rendering-adventures/tier0-rasterization-foundations \
  --tier1-build-dir src/exps-rendering-adventures/tier1-classic-shading \
  --scratch-dir /tmp/ad2_ad3_parity
```

## Artifacts

| File | Role |
|---|---|
| `common/adventures_pass_policy.hpp` | **new** execution-neutral `PassPolicy` + closed `StencilMode` + `ScissorRect` + `scissor_allows`/`clamp_scissor` |
| `common/adventures_sw_raster.hpp` | `SwState` **deleted**; `SwRaster` keeps only its storage; policy is a per-draw argument |
| `common/adventures_vk.hpp` / `.cpp` | `VkPipelineSetup` embeds `PassPolicy`; `stencil_state(StencilMode, ref)` and the policy-scissor adapters; `render(draws, policy)` |
| demo twins 01–05 (`*_sw` + `*_vk`), tier1 08 `*_vk` | migrated to the shared policy |
| `tools/t0_known_answer_checks.cpp` | AD4 oracles migrated (same draws, explicit policies) |
| `tools/t0_policy_tests.cpp` | **new** AD2 gate (GPU-free, cannot skip) |
| `03_depth_test_alpha_blend/depth_blend_plan.hpp` | **new** AD3 pure zone: request → `expected<DepthBlendPlan, DepthBlendError>` |
| `03_depth_test_alpha_blend/depth_blend_edges.hpp` | **new** AD3 edge zone: `execute_software_target`/`execute_software`, `write_png` |
| `03_depth_test_alpha_blend/depth_blend_sw.cpp` / `_vk.cpp` | both twins rewritten around the typed chain / the plan |
| `tools/t0_composition_tests.cpp` | **new** AD3 gate (GPU-free, cannot skip) |

## AD2 — what changed

One vocabulary, two adapters. `PassPolicy` holds exactly the fixed-function
state both twins need — depth test, depth write, straight-alpha blending, the
stencil mode + reference, and the scissor — and nothing else: shader paths,
vertex layout, Vulkan handles and the depth/stencil/framebuffer *storage* stay
with each executor. Before this, the same four concepts existed twice
(`SwState` and the depth/blend/stencil fields of `VkPipelineSetup`) and could
drift apart unobserved.

1. **`SwState` is gone; the policy is an argument.** `draw_triangles(raster,
   policy, verts)` cannot inherit a previous draw's depth/blend/stencil/scissor,
   because there is no "current state" member left to inherit from. This
   removes the prerequisite `raster.state = ...` sequencing the backlog called
   out, and with it the whole class of ordering bugs it invited.
2. **`VkPipelineSetup` consumes the same type.** `add_pipeline()` bakes the
   pipeline-level fields from `setup.policy` through `stencil_state(StencilMode,
   ref)`; `render(draws, policy)` applies the one dynamic-state field (scissor)
   through `clamp_scissor`. The demo twins now author *one* policy object and
   hand it to both calls instead of spelling the scissor twice.
3. **The stencil vocabulary is closed.** `StencilMode{Disabled, WriteRef,
   TestEqual, TestNotEqual}` replaces three independent bools. `invert` without
   `test` used to be representable in both backends and silently ignored by
   both; it is no longer expressible. `WriteRef` is the "compare ALWAYS +
   passOp REPLACE" pair, the two test modes are EQUAL/NOT_EQUAL with KEEP on
   failure.
4. **Defaults and capability differences are documented in the header, not
   implied**: depth test on with LESS, depth write on, blending off, stencil
   Disabled, no scissor; scissor is per-draw on the software side but one
   dynamic scissor per `render()` on the Vulkan side; Vulkan configures both
   faces identically while the rasterizer has no facing; culling is deliberately
   not part of the policy (curriculum demos draw back faces on purpose).
5. **The projection demo's dead local `SwState` is resolved.** Demo 02 built an
   `SwState` it never applied and silently relied on the rasterizer's default;
   it now states `const PassPolicy policy{}` and passes it explicitly.

## AD2 — evidence

`t0_policy_tests` (45 checks, GPU-free, registered unconditionally) drives the
real shared software kernel and inspects the z-buffer and stencil plane
directly. It cannot skip: a CPU rasterizer needs no device, so a refusal there
is a hard failure.

| Group | Asserted |
|---|---|
| defaults | documented defaults are asserted, so a silent default change cannot pass review |
| depth | rejected fragments write neither color nor depth; `depth_write=false` leaves the z-buffer at the clear value; `depth_test=false` lets a farther fragment win |
| order + blend | opaque-then-translucent equals the independently derived source-over value `(128,0,128,255)`, the translucent pass leaves depth at the opaque value, and the reversed order yields pure opaque red — the two orders provably differ |
| stencil | `WriteRef` stores ref 7 only inside its scissor, `TestEqual` keeps only ref pixels (mask boundary asserted), `TestNotEqual` keeps the complement, `Disabled` leaves the plane untouched |
| scissor | band rows drawn, rows outside untouched, an empty band draws nothing, and `clamp_scissor` clamps/keeps-empty as the Vulkan adapter needs |
| **policy isolation** | a hostile first draw (no depth write, stencil `WriteRef`, left-half scissor) followed by a second draw stating the defaults covers the whole target, writes depth everywhere, and does not rewrite the stencil — the acceptance property, measured rather than argued |

The Vulkan half of "both twins consume the same policy" is evidenced by the
unchanged six-pair parity table below (demo 05's stencil pipelines and demo 03's
blend/depth pipelines are now built from the same `PassPolicy` values the
software twin consumed) plus the `*_vk` smoke/parity gates. It is deliberately
*not* re-tested by a library-style device test; there is no new Vulkan gate in
this slice.

## AD3 — what changed

The pilot is demo 03, split into three explicit zones:

1. **Pure preparation** (`depth_blend_plan.hpp`, no framebuffer/rasterizer/PNG/
   Vulkan include): `prepare_depth_blend(DepthBlendRequest)` returns
   `std::expected<DepthBlendPlan, DepthBlendError>`. It validates the frame
   extent, a non-empty scene, that both draws are whole triangle lists, that the
   two ranges sum to the scene, and that the translucent draw follows the opaque
   one — then **moves the vertices into the plan** and stamps each pass with the
   `PassPolicy` the lesson requires (opaque: defaults; translucent: depth tested,
   depth write off, blended). The plan is backend-neutral: it contains layers,
   ranges, policy and owned geometry, and no pipeline index, shader or handle.
2. **Execution edges**: `depth_blend_edges.hpp` holds the software realization
   (`execute_software_target` → image + z-buffer + stencil, `execute_software` →
   image) and the PNG edge (`write_png`). Execution trusts nothing: a plan is
   public aggregate data, so an out-of-range range (`DrawRangeOutOfBounds`) or a
   partial triangle (`DrawNotTriangleList`) is rejected rather than read past or
   silently truncated. The Vulkan twin keeps its own executor (device, SPIR-V,
   pipelines, uploads) but reads the same plan: one pipeline per prepared pass
   built from `pass.policy`, one upload of `plan.vertices`, and the shared
   `write_png` for output.
3. **One diagnostic mapping at the host boundary**: the software twin composes
   `prepare_depth_blend(request).and_then(execute_software).and_then(write_png)`
   and attaches its diagnostic with `or_else`, which observes the failure
   without rewriting it; the Vulkan twin has one `fail(DepthBlendError)` used by
   every stage. Both return `depth_blend_exit_code(error)`, derived from the
   error's stage: preparation 1, execution 2, output 3.

The **closed vocabulary** has 14 members: six preparation failures, two
software-execution failures, five Vulkan-execution failures (`DeviceUnavailable`,
`ShaderModuleUnreadable`, `PipelineUnavailable`, `VertexUploadFailed`,
`FrameSubmitFailed` — adapted bools and indices, never invented for infallible
math), and the output failure. `depth_blend_stage()` is the single error→stage
mapping; `depth_blend_message()` and `depth_blend_stage_name()` are total. The
Vulkan-only members are documented as such: the software path cannot produce
them.

## AD3 — evidence

`t0_composition_tests` (91 checks, GPU-free, registered unconditionally):

| Group | Asserted |
|---|---|
| plan shape | extent, owned vertex count, two passes in lesson order, ranges and `find()` by layer, and each pass's policy (opaque = defaults; translucent = depth tested, not written, blended, stencil untouched) |
| preparation failures | eight single-field mutations of a valid request each produce their own error value — zero/negative extent, empty scene, partial or absent opaque list, partial translucent list, non-summing counts, translucent-before-opaque; the untouched request still prepares, so each case isolates one rule |
| short-circuit + error preservation | a failing preparation leaves a spy-counted execution stage and output stage at **0 runs** and keeps the original error; `or_else` observes the failure exactly once and returns the same value |
| execution refusals | out-of-range range, partial triangle and non-positive extent rejected; a pass-less plan renders an untouched frame instead of reading anything |
| output edge | success returns the written path and a non-empty PNG; an unwritable path returns `OutputUnwritable` and leaves the process's open-descriptor count unchanged |
| plan ownership | the plan still executes after the request (and its vertex vector) has been destroyed |
| policy propagation | two plans differing in ONE policy bit: `blend=false` changes the pixels, and `depth_write=true` makes the z-buffer strictly shallower everywhere it differs — so the prepared policy demonstrably reaches the rasterizer, and "blended fragments must not poison the z-buffer" holds for the *executed* plan rather than only in the plan struct |
| vocabulary | all 14 errors map to the documented stage and exit code, every error has a non-empty message, every stage has a name |

## Mutations (the gates earned their keep)

Each mutation was introduced, the gate run, then reverted (final run green):

| Mutation | Gate | Result |
|---|---|---|
| software kernel: depth test removed (`if (false) continue`) | `t0_policy_tests` | **3/45 checks FAILED**, exit 1 |
| plan: translucent `blend = false` | `t0_composition_tests` | **2/91 checks FAILED**, exit 1 (plan shape + propagation) |
| executor: `draw_triangles(raster, PassPolicy{}, draw)` — ignore `pass.policy` | `t0_composition_tests` | **2/91 checks FAILED**, exit 1 (both propagation checks) |

The third one is why `execute_software_target` exists: without a policy-sensitive
assertion on the *executed* result, an executor that ignored the plan's policy
would have passed every earlier check.

## Baseline comparison (no pixel changed)

Fresh six-pair parity after AD2+AD3, against the AD0 baseline table:

| Pair | differ | % | within-1 | max | Envelope | vs baseline |
|---|---|---|---|---|---|---|
| 01 barycentric | 4 | 0.00% | 4 | 1 | near_exact | identical |
| 02 projection | 37884 | 12.33% | 28419 | 217 | tol13 | identical |
| 03 depth_blend | 42193 | 13.73% | 42193 | 1 | tol1 | identical |
| 04 texture_sampling | 6751 | 2.20% | 6751 | 1 | tol1 | identical |
| 05 stencil | 0 | 0.00% | 0 | 0 | exact | identical |
| 08 normal_mapping | 80744 | 26.28% | 79400 | 2 | tol27 | identical |

Every number matches `adventure_demo_baseline_2026-09-18.md` exactly, so the
refactor moved no pixel in either realization and no envelope was touched. Both
demo 03 twins exit 0 and print their usual success lines; the smoke gates for
all six software demos stay green.

## Corrections and follow-ups recorded

- **Header inventory (G4 follow-up).** Re-running `inventory_headers.py --write`
  picked up 8 consumer edges that the G4 regeneration had missed (5 for
  `shs/rhi/software/sw_offscreen.hpp`, 3 for
  `tests/sw_vk_equivalence_tests.cpp`) — a late include edit landed after G4's
  last `--write`. Committed separately and before this slice; `header_count`
  stays 226 and no tracked header was added or removed. This stale file is why
  `shs_renderer_header_inventory_check` was red on entry to this slice.
- **AD4 tool migrated, not rewritten.** `tools/t0_known_answer_checks.cpp` now
  passes explicit `PassPolicy` values (same draw sequence, same oracles); its
  recorded results are unchanged (checks and prove-fail both green).
- **Stale comments corrected rather than left behind**: demo 03's "pipeline
  state must be set before drawing" sequencing comment, the software raster's
  `SwState` doc block, and demo 04's duplicated scissor pixels.

## Not claimed / limitations

- **AD3 pilots demo 03 only.** The other five pairs still use straight-line
  `main()`; converting them is AD7's job, with this shape as the reference.
- **No device-level test for the Vulkan failure mapping.** The five
  Vulkan-execution errors are adapted at the edge and are exercised by real demo
  runs plus the parity suite, but no CTest entry injects a device, SPIR-V or
  pipeline fault. That remains unclaimed.
- **The Vulkan twin's output-failure exit code moved from 1 to 3**
  (stage-derived). Tooling only distinguishes zero from non-zero, and the gates
  treat any non-zero exit as failure.
- **Allocation failure is not in the vocabulary.** `bad_alloc` during frame or
  plan construction stays a process-level failure, not an error value.
- **Demo 01 is untouched by AD2**: it hand-rolls its coverage loop because that
  *is* the lesson, so it has no `draw_triangles` call and no policy argument.
  Demo 04's software twin consumes the shared policy for its scissor test only.
- **Documented capability differences remain** (per-draw vs per-pass scissor,
  stencil facing, the closed stencil set, no culling in the policy). They are
  written down in `adventures_pass_policy.hpp`; they are not eliminated.
- **No library-side capability is added.** This is demo-consumer work; the
  library G-track is untouched by it.
