# Tier 0 Rasterization — Lessons Learned

> Status: post-mortem of `cpp-folders/src/exps-rendering-adventures/tier0-rasterization-foundations/`
> (2026-09). Five software-vs-Vulkan/Slang demo pairs were driven to exact pixel
> parity (0.00% differing pixels on every pair). This doc records what the parity
> chase taught us, so Tier 1+ rungs don't re-learn it the hard way.
> Companion: `docs/education/rendering_techniques_curriculum.md` (Tier 0 section).

## The headline lesson

**Exact pixel parity is the best test oracle a renderer can have.** "Looks close"
hides sign errors, phase shifts, and silent culls; "0.00% pixels differ" proves
the whole pipeline — vertex transform, rasterization rules, interpolation,
blending, stencil, sampling — agrees with the hardware. Every bug below was
invisible until the parity harness turned "approximately right" into a number.

## Lessons

### 1. Edge-function orientation vs area sign (demo 04 — 3.72% parity)

If rasterizer edge functions are written as `e = cross(p − v_i, v_j − v_i)` while
the triangle area is `cross(v_j − v_i, v_k − v_i)`, the two have **opposite
signs**, and naive weights `w = e / area` produce *negated* barycentrics. The
image doesn't break — it *mirrors*: interpolated varyings (uv) come out negated,
and with `REPEAT` wrap the texture is sampled at a phase-shifted position. A uv
probe showed `uv = (−2.347, −1.486)` where the true value was `(+2.347, +1.486)`.

- Rule: **derive barycentric weights from one consistent orientation
  convention**, preferably a winding-agnostic formulation
  (`w_a = −e_bc / area` works for both windings in the convention above).
- Rule: with tiling samplers, a *sign* bug looks like an *offset* bug. Probe the
  actual interpolated varying before blaming the sampler.

### 2. Silent winding cull (demo 02 — 8.63% parity)

The software rasterizer only rasterized CCW (positive-area) triangles and
**silently dropped** CW triangles; the Vulkan twin ran with `cullMode=NONE`, so
every back-facing face of the projection cube rendered on VK and vanished in SW.

- Rule: **never discard geometry without a trace**. When the twin uses
  `cullMode=NONE`, the SW rasterizer must flip windings, not reject them (swap
  p1/p2 plus the varying pointers, negate the area — the top-left rule then
  biases the same geometric edges the GPU would).
- Corollary: with `cullMode=NONE`, depth alone decides visibility; winding is
  only a coverage question.

### 3. Same vertex stream, same triangles — check consumption first (demo 02)

The cube scene emitted 24 vertices (6 quads) while both twins walked the buffer
with stride 3 (12 non-indexed triangles, some degenerate). Because **both** sides
consumed the identical stride-3 list (`vkCmdDraw` is non-indexed; no index
buffer exists), the "misalignment" was consistent and parity-irrelevant. Hours
went into hypothesizing index-buffer handling that did not exist.

- Rule: **before hunting per-side bugs, prove the two sides consume identical
  inputs.** Diff vertex streams and draw counts first; only then compare
  rasterization behavior.
- Rule: parity work is mostly about *eliminating hypotheses cheaply* — a one-line
  grep for `vkCmdDraw`/index usage is cheaper than a probe program.

### 4. State defaults must mirror pipeline state (demo 05)

The SW state struct default-initialized `depth_test = true` while the VK twin's
pipeline disabled depth test/write for both passes; a coplanar quad was
depth-rejected against EQUAL-masked stencil fragments. Explicitly setting
`depth_test = false, depth_write = false` per pass fixed parity.

- Rule: **enumerate the pipeline state (blend, depth, stencil, cull, scissor,
  topology) as a checklist and set every item explicitly on both twins** — never
  rely on implicit defaults. Hidden state differences are the #1 source of
  "unexplainable" parity drift.

### 5. Shader transform conventions count as pixels (demo 05)

`stencil.slang` used column-vector form `mul(pc.mvp, float4(pos, 1.0))` while the
push-constant matrix was row-vector oriented; switching to `mul(float4(pos, 1.0),
pc.mvp)` with the `-matrix-layout-column-major` pin took parity from 80.78% →
14.19% in one change.

- Rule: **treat matrix layout, multiplication order, and NDC conventions
  (`GLM_FORCE_DEPTH_ZERO_TO_ONE`, y-flip) as part of the pixel contract** — not
  as "math details". One transposed multiply is a 66% parity error.

### 6. Replicating sampling rules is tractable — if uv is right (demo 04)

Once the barycentric sign was fixed, plain float `floor(uv·size − 0.5)` nearest
and standard bilinear matched the GPU samplers exactly at this resolution,
including `REPEAT` wrap and per-draw scissor bands.

- Rule: **match conventions, not bit-exact hardware**: texel centers at
  `(i + 0.5)/size`, `u·size − 0.5` offsets, `floor(+0.5)` rounding, `%`-wrap that
  also handles negative uv. Float arithmetic diverges from fixed-point hardware
  interpolation only at sub-LSB level, far below any sane parity threshold.
- Rule: rasterization edge cases (a pixel on a shared triangle edge being drawn
  by both triangles, a demo-local rasterizer with no top-left rule) are harmless
  when the shaded data is continuous, but become visible with discontinuous
  data (stencil masks, scissor boundaries). Know which rule you can skip.

### 7. Process lessons (tooling)

- **Diff-first debugging**: a row color-run dumper (`runrow`) and an ASCII
  difference map (`diffmap`) localized the projection bug to two triangles in
  minutes; the parity tool alone only said "8.63%".
- **Beware stale artifacts**: a PNG that looks regenerated but is a stale copy
  sends analysis down a rabbit hole (we briefly "found" projection-like wedges
  in the texture demo because the file under analysis was stale). Always
  regenerate inputs immediately before analysis; checksum when in doubt.
- **Validate the probe before trusting its output**: one early tool baked a
  *guessed* memory layout, another used the wrong stride — both produced
  confident nonsense. A probe is only evidence if its assumptions are verified.
- **Shell ergonomics in automation**: `grep -c` exits 1 on zero matches and
  breaks `&&` chains — use `;` or guard it, or multi-step verification scripts
  silently stop halfway.
- **Keep probes ephemeral**, named after what they test (`runrow`, `texprobe`),
  and remove scratch files from the source tree immediately after use.

## Reusable parity harness artifacts

All debug tooling from the run lives in `/tmp/t0final` (session-scoped):
`t0_parity` (difference-percentage report), `diffmap` (ASCII difference map),
`dump` (color-run dumper), `texprobe` (exact differing-pixel runs with RGBA),
`runrow`, plus one-off probes. Before Tier 1 work starts, promote `t0_parity` +
`diffmap` into the repo as the standard cross-backend validation gate (the
curriculum's "cross-backend equivalence test" for rungs).

## Summary checklist for the next tier

1. Checklist every pipeline state on both twins — no implicit defaults.
2. One matrix/NDC convention note per tier; cite it in every shader.
3. Prove identical input consumption (vertex streams, draw counts) before
   comparing outputs.
4. State sign/orientation conventions for every cross product in a rasterizer,
   in a comment next to the code.
5. Never silently drop geometry.
6. Diff-first tooling: ASCII difference maps beat percentages.
7. Regenerate artifacts immediately before any analysis.

8. Code style: vertical alignment is law (Constitution I §8 in
   `docs/spec/conventions.md`) — align declarations, struct tables, and trailing
   comment columns NASA/JPL-style.
