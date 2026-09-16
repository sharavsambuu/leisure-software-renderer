# Backend Capability & Approximation Notes

> Status: strategy notes (2026-09-15), distilled from the post-Tier0 review
> discussions. Records the capability story of the software backend relative to
> Vulkan: (1) an existing demo (tetris) as the backend-switch exercise, (2) what
> software can and cannot offer, (3) parity maturity (exact → tolerance), (4)
> approximation as a first-class, measurable axis, (5) compute as the cleanest
> dual-implementation stage (C++ + Slang).
> Companion docs: `docs/education/demo_to_lib_ingestion_notes.md` (ingestion
> workflow + Domain POD policy), `docs/education/tier0_rasterization_lessons.md`
> (parity method), `docs/education/rendering_techniques_curriculum.md` (the
> ladder these capabilities serve), `docs/roadmap/slang_utilization_plan.md`.

## 1. The tetris backend-switch exercise

The tetris demo is the best-case candidate for proving the backend-agnostic
thesis on a real application, because of its existing seams:

- `domains/` (matrix, progression, mission, powerups, spatial_fx) is
  presentation-free pure gateway code — ports to any backend for free.
- `edges/rasterizer/` is the only pixel-touching code; all rendering policy
  (z-tested transparency overlay: depth-test on, depth-write off, blend on;
  canvas y-flip) lives there.
- snake's `plan_snake_scene` already demonstrates the missing layer: a *plan*
  step between domain and presentation.

The path to "simply switch to Vulkan":

1. Lift the rasterizer edge's implicit decisions into a backend-agnostic
   **draw list / scene plan** (draw triangles with alpha, z-test, blend flags;
   SoA particles → instanced draws).
2. Map the hand-rolled rules to pipeline state — the overlay rule is exactly
   tier0 demo 03's `depth_test = true, depth_write = false, blend = true`,
   expressible via `SwState` and `RHIDepthStateDesc`.
3. Plug into the existing backend selection (`ctx.set_primary_backend`) and
   render-path recipe machinery — no new mechanism needed.
4. Pin with the tier0 parity pattern: render the same frame through both
   backends, diff.

Known work items (not blockers): HUD/text and canvas pixel ops need draw-list
representation (SDF text or a dedicated pass); the custom overlay semantics
must be pinned by a parity test, not assumed.

End state: same game, same gateways, one flag flips the backend, harness proves
the pixels agree — the public demonstration of the backend-agnostic thesis.

## 2. What the software backend can offer vs Vulkan

Three buckets, honestly separated:

- **Techniques — yes, and software is sometimes better.** Everything Vulkan
  renders is math over pixels; software expresses all of it. Transparency is
  *stronger* in software: the GPU blend unit is a fixed-function constraint
  (one `src*f(src) + dst*f(dst)` equation), while software blending is a
  function we wrote — per-pixel linked lists, stochastic transparency, deep
  images, arbitrary blend operators are trivial. Software ray tracing is
  *easier* than the GPU version (no BVH/RT-core API constraints): the SW path
  can be a ground-truth path tracer.
- **Hardware features — implementable as emulation, not acceleration.** MSAA
  (N samples + resolve, ~Nx cost), mesh/task shaders (equivalent geometry
  stage), conservative rasterization, variable-rate shading: all have
  straightforward software equivalents at parity-harness resolutions. Async
  compute maps to threads (multithreaded best-practices doc), with
  "same observable behavior" as the contract — not same scheduling. Interactive
  framerates at scale are explicitly not the SW backend's job.
- **The real boundary — exact parity erodes with shader complexity.** GPU
  compilers contract to FMA, reorder, use fast-math; mobile mediump differs;
  interpolation/reduction order is unspecified. Divergence is sub-LSB per op
  but accumulates across long shaders. Tier0's 0.00% held because the shaders
  were small and conventions pinned (see lessons doc §6).

Strategic framing: the software backend is the **executable specification** of
the GPU paths — the readable, deterministic, constraint-free reference that (a)
validates every promoted feature via parity, (b) serves headless/CI/testing
where no GPU exists, (c) implements hardware-awkward techniques *better* than
hardware does.

## 3. Parity maturity: exact → tolerance

Expect the parity standard to grow a second mode as techniques deepen:

- **Exact mode** (tier0): "0.00% pixels differ significantly" — for
  convention-pinned, small-shader features. Remains the gate for every
  promoted fixed-function operator.
- **Tolerance mode** (complex shaders / stochastic techniques): "≤1 LSB on
  ≤0.01% of pixels" — the CTS-style equivalence testing used hardware-vs-
  hardware. Fast-math and unspecified reduction order make bit-exactness the
  wrong contract at this scale; the contract becomes "matches conventions,
  within bounded drift" (tier0 lessons: *match conventions, not bit-exact
  hardware*).

## 4. Approximation is a first-class, measurable axis

Real-time rendering *is* approximation of offline ground truth; the ladder
treats it as a designed axis, not a compromise:

- Most algorithms degrade by **sample count, not by algorithm change**: path
  tracing at 1 spp + temporal accumulation + denoising vs the same code at
  1024 spp; full raymarch → froxel grids; GI → probes/SSGI; particle counts
  downsize within the existing SoA structures.
- **Quality tiers are recipe presets over the same scene data** — the same
  mechanism as the mobile story ("same scene, simpler path"). One scene, a
  ladder of recipes, user picks a rung.
- **The SW path is the oracle that quantifies approximation error.** A GPU
  engine ships an approximation and hopes; here, the offline ground truth is
  renderable, and each fast approximation gets a tracked number ("denoiser
  within 2% of the 4096-spp reference"). The parity harness grows the
  comparison "fast recipe vs ground-truth recipe, error ≤ threshold" — same
  tooling, new comparison axis.
- **Stochastic sampling is seeded and replayable** (Domain POD purity rules),
  so approximations are deterministic run-to-run — a property temporal GPU
  denoisers cannot claim.

Non-approximable residue is small and benign: techniques whose result depends
on unsimulated global state simply have no low rung on the ladder.

## 5. Compute: the cleanest dual-implementation stage (C++ + Slang)

Compute carries none of the fixed-function complexity of graphics — buffers
in, dispatch, buffers out — so it is the easiest stage to implement on both
sides, and the parity contract is *stronger*: buffer-diff, byte-exact when
conventions are pinned (vs 0.00% pixels).

Existing substrate, already in the tree: the curriculum's
"Slang module + software `cpp_impl` handler" pattern; a `GpuCompute` path in
`demo_rendering_paths.cpp`; `shs::WaitGroup` parallel stages and xsimd
(`hello_flat_shading_xsimd.cpp` is hand-SIMD compute); the multithreaded
best-practices doc; the Virtual-SPU ("CPU as compute device, unified memory")
roadmap vision.

Mapping table:

| Vulkan compute concept | Software equivalent |
|---|-----------------------|
| dispatch (x, y, z)        | parallel_for over tiles          |
| workgroup / shared memory | per-tile arrays, cache-friendly  |
| `barrier()`               | `WaitGroup` / tile-boundary sync |
| SSBO / descriptor buffers | plain spans (unified memory)     |
| atomics                   | C++ atomics                      |
| subgroup ops              | xsimd / tile-level reductions    |

What disappears in software: descriptor management, pipeline layouts, binding
models — the plumbing tax. A CPU kernel is a function; game state *is* the
input (zero-copy).

Application note: most modern techniques are compute-first (OIT, denoising,
GI, light clustering, particle sim, culling). Implementing their cores as pass
contracts with `cpp_impl` + Slang compute gives each technique a deterministic
CPU oracle and a GPU production path feeding their respective rasterizers.
The existing culling domains (`culling_query`, `jolt_culling`) are already
software compute awaiting GPU twins.

Performance asymmetry is expected and accepted: the software path is the
deterministic, replayable, debuggable oracle — not the production frame-rate
path.

