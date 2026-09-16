# Demo-to-Lib Ingestion & Domain POD Notes

> Status: methodology notes (2026-09-15), distilled from the Tier 0 milestone
> review. Records three policy decisions for the rendering-adventures curriculum:
> (1) the demo-first → lib-ingestion workflow, (2) when/why pure gateway-based
> Domain PODs apply to demo code, (3) how fixed-function operator semantics
> (depth/stencil/blend/scissor/sampling) graduate into `shs-renderer-lib`.
> Companion docs: `docs/education/tier0_rasterization_lessons.md` (Tier 0
> post-mortem), `docs/education/rendering_techniques_curriculum.md` (the ladder
> these policies govern), `docs/education/backend_capability_notes.md` (the
> tetris backend-switch exercise and SW capability story),
> `docs/roadmap/domain_pod_engine_rollout_roadmap.md`
> (the canon these notes reference), `docs/spec/value_oriented_programming.md`.

## 1. The demo-first ingestion workflow

The curriculum writes demos first, library generalization second. Demos are
proving grounds: logic is written inline (`*_sw.cpp` next to its `*_vk` twin)
where it is cheap to iterate and verifiable against the parity harness. Only
code that survives several demos gets promoted.

- **Promotion trigger (rule of three):**
  - used by one demo → stays in the demo;
  - used by two demos → per-tier `common/` (what tier0's
    `adventures_sw_raster.hpp` / `adventures_vk.*` are);
  - used by three demos, or is genuinely renderer-shaped → promote into
    `exps-rendering-adventures` cross-tier `shared/` or into `shs-renderer-lib`.
- **Layering rule:** curriculum-specific types (`T0Vertex`, scene generators,
  `OffscreenVulkan` harness, parity tooling) stay in the adventures tree. Only
  renderer-shaped things (rasterizers, resource upload, pipeline state, memory
  helpers) enter the lib. Long-run convergence is expected and desired: the
  curriculum rediscovers the lib's concepts from first principles, and repeated
  "this is just `RasterState` again" moments are the promotion signals.
- **Verification scales with ingestion:** once a helper becomes lib code, a
  change ripples across all demos. The 0.00% sw-vs-vk parity harness is what
  makes that affordable — keep it as the gate for every promoted piece.
- Do **not** ingest early: unsettled structure resists generalization, and
  premature abstraction taxes every subsequent demo.

## 2. Pure gateway-based Domain PODs in demos

The snake, tetris, and fps demos already enforce the pattern
(`domains/<name>/{contract,action,event,gateway}.hpp`), and the Domain Pod
canon docs declare it applicable to all demos alike. Policy for the
rendering-adventures tree:

- **Tier0-style static demos stay procedural.** One-shot render-to-PNG programs
  have no state timeline to reduce; forcing
  `contract/command/event/gateway` boilerplate onto them is ceremony without
  payoff and fights the demo-as-proving-ground philosophy.
- **Trigger point:** introduce Domain PODs at the first tier that has *time or
  input* (animation, camera, interaction). That is when state transitions exist
  worth auditing: `(SceneState, events, dt) -> SceneState'` as a pure gateway,
  then a pure `render(state) -> framebuffer`.
- **Synergy with parity:** record the event stream once, reduce once, feed the
  applied state to both backends, diff. Replayability and determinism fall out
  for free — the same auditable-transition property the canon cites.
- **Purity is structurally enforced, not compiler-enforced.** C++ has no effect system (even at the C++23 lib baseline) —
  the rules are: gateways take PODs by value/`const&` and return
  new PODs (or out-params), no globals, no RNG, no I/O below the edge layer,
  events as plain enums — then purity is *tested* via determinism/replay gates
  (same inputs → byte-identical outputs), as tetris already does.

## 3. Ingesting fixed-function operator semantics into the lib

Tier0 hand-rolled depth, stencil, blend, scissor, and sampling semantics with
**booleans** (`depth_test`, `stencil_invert`, `blend`, …). The lib's RHI layer
already has the ingestion slots: `RHIDepthStateDesc` / `RHIBlendStateDesc` in
`shs/execution/rhi/pipeline/pipeline_desc.hpp`, backend mappers
(`vk_depth_state` in `vk_pipelines.hpp`), a SW rasterizer
(`shs/execution/sw_render/rasterizer.hpp`), and pipeline-state hashing keyed on
these descs. The ingestion path for each operator:

1. **Generalize booleans → operator enums** (`CompareOp`, `StencilOp`,
   `BlendFactor`, sampler wrap/filter modes) in `pipeline_desc.hpp` — the
   closed algebra the GPU already exposes, not a copy of `SwState`.
2. **Implement the same enums in the SW rasterizer.** The tier0 software
   implementation is an *executable specification* of what `LESS`, `REPLACE`,
   `INVERT`, src-alpha blending mean at the pixel level.
3. **Map in the Vulkan driver** (extend the existing desc mappers).
4. **Pin with a parity test.** Render-both-diff-pixels is a pixel-exact
   conformance suite for the promoted operator — stronger than struct-mapping
   unit tests, because it validates semantics, not spelling.

- **Duplication smell to retire:** the `exps-gpu-renderer` demos hand-write
  near-identical `VkPipelineDepthStencilStateCreateInfo` stanzas per demo.
  Each tier that deepens the operator vocabulary multiplies this duplication;
  promoting the state descs absorbs it.
- **Do not ingest the harness.** `OffscreenVulkan` (headless instance, one-shot
  submit, PNG readback) and the parity tools are curriculum tooling. Ingest the
  *operators and their semantics*; replicate the harness *pattern* as lib tests.
- **Cadence:** continuous, per-tier ingestion — each rung that forces a richer
  mode (stencil ops, additive blending, depth bias, MSAA) deposits exactly one
  operator into the lib, rather than one large migration someday. Demos remain
  the lib's conformance test suite.
