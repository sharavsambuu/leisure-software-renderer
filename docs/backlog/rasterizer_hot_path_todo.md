# Renderer-Lib Review (2026-09-18) — hot-path & follow-up todo list

> Status: **active (2026-09-18)**. Source review:
> [`docs/review/2026-09-18_antigravity_shs_renderer_lib_review.md`](../review/2026-09-18_antigravity_shs_renderer_lib_review.md)
> (code-level audit of `shs-renderer-lib`; its §5 recommendations P1.1–P3.3 map
> to R-series below). Sibling backlog:
> [`governance_review_2026_09_18_todo.md`](governance_review_2026_09_18_todo.md)
> (G1–G3, closed 2026-09-18). Law precedence: Constitution II §2.2 (stricter
> rule prevails; cite, never re-legislate). Verification after every code item:
> full `build/` CTest green (48/48 as of 2026-09-18) +
> `check_kdba_boundaries.sh` green. Per the law-budget norm (Constitution II
> §2.2(5), adopted 2026-09-18): every code item below names its acceptance
> gate — items are ticked only with their DoD met. Review-finding staleness is
> recorded inline, never silently rewritten.

## Review-finding triage (2026-09-18, against current tree)

- The review's headline hot-path finding **P1.1 (per-triangle heap
  allocations in clipping) is already resolved**: `rasterizer.hpp:120-204` now
  clips via fixed-capacity `ClipPolygon` (`std::array<RasterVertex, 16>`,
  `reset`/`push`, double-buffered `poly`/`scratch` across all 6 frustum
  planes) — the exact remediation the review prescribed, landed with the G1.1
  allocator-interception work (governance backlog). No R-series item needed.
- The review's metrics are pre-G1.1/G3.2: test suite is now 48/48 (was 43/43)
  and the boundary gate suite has grown (pure-value library classification,
  G2.1). Dated review artifacts are not edited (Rule N5: stable paths, loud
  headers).
- The review's §2.1 coordinate-law claims (row law, Jolt `S·M·S` conjugation,
  quaternion involution, NDC mapping) are now mechanically pinned by
  `math_law_golden_tests.cpp` (G3.2) — prose findings became enforced law.

## R1 — Hot-path: fragment shader de-virtualization (review P1.2)

- [ ] **R1 Fragment-shader `std::function` removal** — `program.hpp:22-23`
  defines `VertexShaderFn`/`FragmentShaderFn` as `std::function`, and
  `rasterizer.hpp:457` invokes `program.fs(fin, uniforms)` per covered pixel:
  millions of indirect, non-inlinable calls per frame at 1080p; blocks SIMD.
  Action: parameterize `rasterize_mesh` on the shader functor type
  (`template <typename Program>`), enabling full inlining and
  auto-vectorization; keep `std::function` only at the app/host seam where
  type erasure is genuinely needed. **Gate (law-budget):** existing
  `sw_vk_parity_tests` + `math_law_golden_tests` must stay green with
  byte-identical output; full CTest green.

## R2 — Hot-path: incremental edge functions (review P1.3)

- [ ] **R2 Per-pixel barycentric recompute → incremental stepping** —
  `rasterizer.hpp:378` evaluates `barycentric_2d` (two 2D cross products + a
  divide) for every pixel in the triangle bounding box. Action: standard edge
  functions `E(x, y) = A·x + B·y + C` stepped incrementally (`Δx = A`,
  `Δy = B`), replacing per-pixel mul/div with adds. Land after R1 (same inner
  loop). **Gate (law-budget):** new golden test pinning one known triangle's
  pixel-coverage set (exact rows/columns) before/after the rewrite, plus the
  R1 gate set (parity + math golden + full CTest).

## R3 — Threading: screen-space tile binning (review P2.1)

- [ ] **R3 Tile-binned parallel rasterization** — `rasterizer.hpp:474` pushes
  per-large-triangle row slices through `parallel_for_1d` and blocks on
  `WaitGroup::wait()`: a hard sync barrier per large triangle; small triangles
  never parallelize. Action: coarse binning pass into 32×32 (or 64×64)
  screen-space tiles, then one job per tile rasterizing into disjoint memory —
  no per-triangle barriers. Land after R1+R2 (bins the simplified loop).
  **Gate (law-budget):** R1/R2 gate set green, plus a recorded benchmark of
  barrier/mutex contention vs. worker count — per the G2.2 governing
  clarification, `ThreadPoolJobSystem`'s mutex/condvar implementation is a
  tracked, benchmark-gated exception on the defined hot paths.

## R4 — Rejected-by-default: open PassId (review P2.2)

- [ ] **R4 Record rejection rationale for open `PassId`** — the review proposes
  widening the closed `PassId` enum to open 64-bit string hashes
  (Constitution I §7 pluggability). Do **not** implement: open pass
  vocabularies contradict the closed-vocabulary doctrine the codebase is built
  on (Gate 9 exhaustive-visit rails, closed command/event variants locked with
  `static_assert(std::variant_size_v == 1)`), and a hash-collided user pass
  cannot be semantically verified by any mechanical gate — failing the
  law-budget norm at adoption. Action: record this rationale here (done by
  this entry); revisit **only** if a concrete third-party-pass requirement
  appears, and then via a gateway-mediated registration design that keeps the
  internal ID space closed. No code change; item closes when the rationale is
  reviewed and the finding is marked rejected in the next docs sweep.

## R5 — Housekeeping: Vulkan forwarder residue (review P2.3)

- [ ] **R5 Retire `rhi/drivers/vulkan/` forwarder headers** — 8+ five-line
  forwarding headers redirect to `shs/rhi/vulkan/runtime/` (legacy migration
  residue, also tracked as migration step-7 forwarder retirement). Action:
  repoint remaining consumers, then delete the forwarders in the next
  path sweep. No renames beyond the already-scheduled retirement (Rule N5:
  stable paths, loud headers). **Gate (law-budget):**
  `check_backend_seam_symbols.sh` + header self-containment green; zero
  includes of the old paths repo-wide.
