# AD4 — Independent known-answer tests (closed 2026-09-18)

Owner: `adventure_demo_conformance_backlog.md` AD4. Evidence recorded per its
acceptance rule ("every lesson has independent numerical assertions with
justified tolerances; tests run in GPU-free CI. Document input restrictions
separately from guarantees of a general rasterizer").

Independence claim: no check accepts parity agreement between the `*_sw` and
`*_vk` twins as evidence. Every expectation is derived here from first
principles (analytic geometry, the lesson's documented texture/mapping, direct
depth/stencil **storage** reads) or is an exact inversion of the lesson's own
encode step. The parity suite remains a separate gate.

## Reproduce

```sh
cd cpp-folders/build
cmake --build . --target t0_known_answer_checks t1_08_known_answer_checks
ctest -R 'known_answer' --output-on-failure   # 4/4: checks + prove-fail, both tiers
```

## Artifacts

| File | Covers |
|---|---|
| `tier0-rasterization-foundations/tools/t0_known_answer_checks.cpp` | demos 01–05 |
| `tier0-rasterization-foundations/common/adventures_stb_load.cpp` | one stb_image (PNG **read**) TU shared by both tiers |
| `tier1-classic-shading/08_normal_mapping/normal_mapping_ka.cpp` | demo 08 |

Both tiers register a `*_known_answer_checks` entry and a
`*_known_answer_prove_fail` entry (`--prove-wrong-expected`). Checks fail the
process (nonzero exit), never rely on `assert`, and run GPU-free: the two
PNG-based oracles execute the real software demo binaries, so they need no
Vulkan device and no window.

## Coverage and evidence style

| Lesson | Oracle style | Asserted |
|---|---|---|
| 01 | PNG of the real `t0_tri_barycentric_sw` | interior color vs an **analytic barycentric solve** (2×2 linear system — deliberately not the rasterizer's edge-function/area-ratio form), exterior stays clear, centroid weights equal 1/3 |
| 02 | in-process through the shared tier0 `SwRaster` kernels | hand-derived NDC (textbook trig + projection formulas, double) vs the demo's GLM chain for 8 cube corners × both projections; front-face depth equals the interpolated **plane** depth; front/back separation is measurable; nearest surface wins; front-face color |
| 03 | in-process, **depth storage inspected directly** | near-opaque depth stored (0.20), overlap color = near red, blended color over far blue, and transparent draws leave the z-buffer untouched (still 1.0) |
| 04 | PNG of the real `t0_texture_sampling_sw` | regenerated checkerboard + independent REPEAT-wrap/NEAREST/BILINEAR/uv model; scissor band boundaries; textured-vs-untouched pixels |
| 05 | in-process, **stencil storage inspected directly** | EQUAL mask interior/exterior, inverted (NOT_EQUAL) mask interior/exterior |
| 08 | PNG of the real `t1_normal_mapping_sw` | analytic Lambert flat half; full mapped chain (hand-computed uv, regenerated bump texels, 8-bit decode, `T = normalize(cross(up, N))`, `B = cross(N, T)`); decoded normal is unit length with positive z. No cross-half brightness assumption |

## Justified tolerances

| Tolerance | Value | Justification |
|---|---|---|
| 01 interior color | ±1.5 LSB | one 8-bit round-trip per channel plus a half-step for float accumulation |
| 02 NDC vs manual math | 1e-5 | pure double arithmetic; the two formulations agree to round-off |
| 02 stored depth | 1e-3 | the rasterizer interpolates in `float`; the oracle in `double` |
| 02 nearest-wins margin | 1e-3, guarded by a 5e-3 separation precondition | perspective depth is nonlinear here: the front/back gap is only ≈9e-3 at this cube scale (ortho ≈8e-2), so the margin must be a documented fraction of a *measured* gap, not a guessed constant |
| 03 depth storage | 1e-6 | exact stored plane constants (0.20 / 0.70) |
| 03/05 colors | ±1.0 LSB | 8-bit quantization of a float blend |
| 04 samples | ±2 LSB | 8-bit truncation plus the float mix chain |
| 05 clear-boundary colors | ±0.5 LSB | integer clear value in an 8-bit channel |
| 08 flat / mapped | ±1 / ±2 LSB | 8-bit quantization; the mapped chain adds a decode round-trip |

## Input restrictions (what these checks do NOT guarantee)

- **Fixed lesson inputs.** The oracles pin the frozen lesson scenes, the tier0
  640×480 frame, and the documented palettes/mappings. They assert those
  lessons, not the correctness of a general rasterizer.
- **PNG-based probes** read the software binaries' default-extent output only
  (no resize, no MSAA, no alternate formats).
- **In-process probes** drive the shared tier0 `SwRaster` kernels — the same
  code the `*_sw` demos use. They do **not** exercise the Vulkan twins; that
  remains the parity suite's job.
- **Oracle drift is a failure, by design.** Changing a lesson's palette,
  mapping, scissor band, or projection requires updating the oracle in the same
  change.
- No claim of bit-exact hardware sampler emulation; the contract is "matches
  conventions within bounded drift" (Tier 0 lessons §6).
- The tier-1 (08) harness still reports only the **first** mismatch, whereas the
  tier-0 harness reports up to 64. Known asymmetry, tracked below.

## Defects found while validating (the checks earned their keep)

The tool was authored but had never been compiled or run. Its first execution
reported **28 mismatches**. All five root causes were defects in the *oracle*,
not in the demos (the demos' own parity and SW-smoke gates were green):

1. **02 nearest-wins used a `-0.02` margin** — arithmetically impossible: the
   perspective front/back separation at this cube scale is only ≈9e-3.
   Replaced with a plane-depth comparison plus an explicit separation guard.
2. **03 probe `(320,300)` was covered by the translucent quad** (quad spans
   x 176–464, y 204–444). The measured `(137,143,66)` is exactly
   `0.55·green + 0.45·red`. Moved to `(320,180)` (inside both triangles, above
   the quad).
3. **04 probes sampled rows 400–470**, below anything drawn: the quad spans
   y 96–383 and the scissor band starts at 300, so rows **300–383** are the
   only textured rows. Probes moved inside.
4. **04 blue channel (6 checks)** — the oracle's even checkerboard block was
   `(230,230,220)`; the lesson palette is `(230,230,230)`. Only even-block blue
   diverged, which is exactly the observed failure signature.
5. **05 probe `(150,300)` was inside the silhouette** (at row 300 the triangle
   spans x 146.6–493.4), not exterior. Moved to `(80,300)`.

Supporting improvements in the same pass: the harness now reports up to 64
mismatches with probe coordinates and channel (previously first-mismatch only),
and `--prove-wrong-expected` now corrupts **both** oracle styles (barycentric 01
and sampler 04) and requires both to be caught.

Outcome: **28 → 0**. The failures above are themselves evidence that the
comparator is live — a vacuous gate could not have produced them.

## Prove-fail

| Target | Result |
|---|---|
| `t0_known_answer_prove_fail` | `PROVE OK: corrupted expectations caught (barycentric 3, sampler 8)` |
| `t1_08_known_answer_prove_fail` | `PROVE OK` |

Each prove entry exits 0 **only** when the deliberately corrupted expectation is
detected, so a correlated software/Vulkan mistake cannot pass through this tool
by image agreement alone.

## Verification (2026-09-18)

- `ctest` (full `cpp-folders/build`): **67/67 passed** (baseline 63 + these 4)
- `ctest -R known_answer`: 4/4
- `tools/check_kdba_boundaries.sh`: all checks passed
- `tools/check_include_graph.py`: OK

## Remaining limitations (not claimed as done)

- The tier-1 08 harness keeps first-mismatch-only diagnostics; aligning it with
  the tier-0 cap is a small follow-up.
- `--prove-wrong-expected` exercises the two PNG oracle styles. The in-process
  depth/stencil checks (02/03/05) share the same `near`/`require` comparators but
  carry no per-check mutation knob of their own, so their non-vacuity is
  inferred rather than directly proven. A per-check mutation hook is a possible
  follow-up.
- AD4 is demo-consumer evidence and does **not** substitute for the library
  SW/Vulkan equivalence gate (kdba backlog `G4`), which has a separate owner.
