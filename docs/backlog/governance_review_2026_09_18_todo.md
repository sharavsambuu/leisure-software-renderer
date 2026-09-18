# Governance Review (2026-09-18) — improvement todo list

> Status: **active (2026-09-17 started; review dated 2026-09-18)**. Source review:
> [`docs/review/2026-09-18_antigravity_constitutions_and_laws_review.md`](../review/2026-09-18_antigravity_constitutions_and_laws_review.md)
> (Constitutions I–III + annexes: tensions T1–T4, recommendations P1–P3, plus
> agent additions A1–A2). Law precedence: Constitution II §2.2 (stricter rule
> prevails; cite, never re-legislate). Verification after every item: full
> `build/` CTest green + `check_kdba_boundaries.sh` green +
> `check_include_graph.py` green. Header inventory is content-hashed — any
> header edit regenerates it in the same commit (`inventory_headers.py --write`,
> Rule 15). Items may only be ticked with their DoD met; disputes are recorded
> inline, never silently rewritten.

## G1 — Turn stated law into gates (P1)

- [x] **G1.1 Allocator-interception CTest (Tension 2 / P2)** — Override global
  `operator new`/`malloc` for the duration of `VerticalSliceHost::run_frame`;
  fail if any heap allocation escapes the designated frame arena. Then migrate
  the software-rasterizer clip buffers (`rasterizer.hpp`: per-triangle
  `std::vector<RasterVertex>`) to arena/fixed-capacity storage so the gate goes
  green. DoD: new CTest target red-before/green-after rasterizer migration;
  full CTest green; boundary + include-graph gates green.
  > **DONE 2026-09-18:** new CTest `shs_renderer_frame_allocator_interception_tests`
  > (#30; `tests/frame_allocator_interception_tests.cpp`): global
  > `operator new`/`delete` (incl. **aligned** variants — `RasterVertex`
  > embeds 16-byte `glm::vec4`, so clip-path vectors allocate through them)
  > replaced by counting versions; RAII `ArmedWindow` arms around
  > `run_frame`; caller passes a stack-buffer `monotonic_buffer_resource`
  > frame arena. Negative self-check proves the interception counts a
  > deliberate allocation (gate cannot pass vacuously). **RED before
  > migration:** `3 heap allocation(s) (~988 bytes) escaped the frame arena`
  > (rasterizer clip-path vectors + `to_render_items` value return).
  > **GREEN after:** `rasterizer.hpp` clip path migrated to fixed-capacity
  > `detail::ClipPolygon` (`std::array<RasterVertex, 16>`; triangle vs 6
  > half-spaces maxes at 9 vertices; ping-pong plane passes, zero heap);
  > `<vector>` include dropped from rasterizer.hpp. `SceneObjectSet::
  > to_render_items(std::vector<RenderItem>&)` fill-into-ref overload added
  > (capacity persists across frames; by-value overload kept);
  > `vertical_slice_host.hpp` uses it. Warm-up frames own the one-time
  > first-touch costs; steady-state frames allocate ZERO. Validated: full
  > build 0 errors; **CTest 46/46**; kdba boundary gate green; include-graph
  > gate green; `engine_header_inventory.json` regenerated in-tree
  > (`inventory_headers.py --write`).
- [x] **G1.2 Backend seam symbol-hygiene gate (agent A1; runbook §7.5)** — New
  `check_backend_seam_symbols.sh`: fail if the SDL2 anchor TU (dlopen dispatch)
  carries undefined `SDL_*` / `IMG_*` symbols (flat-namespace hijack class).
  Wire into CTest; document the zero-undef contract beside the runbook recipe.
  DoD: gate green on current tree; negative test proves it trips when an
  undefined `SDL_*` symbol is introduced.
  > **DONE 2026-09-18:** validated — `cmake --build cpp-folders/build --target
  > shs_renderer` green; CTest 45/45 (0 failed). New entries:
  > `shs_renderer_backend_seam_symbols_check` (#11, gate green on current tree —
  > SDL2 anchor carries zero undefined `SDL_*`/`IMG_*` symbols) and
  > `shs_renderer_backend_seam_symbols_negative_test` (#12, proves the gate
  > trips on a symbol-referencing stub TU and fails honestly with no anchor
  > object). SDL2 anchor policed; SDL3 anchor reported, not policed. Zero-undef
  > contract documented in `sdl3_cutover_runbook.md` §7.5.

## G2 — Constitution amendments (P2)

- [x] **G2.1 Stateless leaf classification (Tension 1 / P1)** — DONE 2026-09-18.
  Constitution II amendment in `value_oriented_programming.md` (§6.1 area):
  two module classes — *Stateful DVO Pods* (full Core 4) vs *Pure Domain Value
  Libraries* (contract + pure transforms; exempt from monostate
  command/event scaffolding; growth into state forces reclassification). Gate
  landed: `tools/check_pure_value_libraries.sh` (+ negative test), wired into
  `check_kdba_boundaries.sh` as the library-classification check; 3 pure
  domain value libraries verified conformant, all gate runs green.
- [x] **G2.2 Scope the wait-free guarantee honestly (Tension 4)** — DONE
  2026-09-18. Governing clarification in `multithreaded_coding_best_practices.md`
  (§5 area): wait-free guarantee scoped to the defined hot paths
  (simulation/recording loops, raster inner paths); `ThreadPoolJobSystem`'s
  mutex/condvar scheduling named as a tracked, benchmark-gated exception
  (evolution target: true wait-free job system). Cross-linked Constitution III.
- [x] **G2.3 "Law budget" adoption norm (agent A2)** — DONE 2026-09-18.
  Norm recorded as Constitution II §2.2 item 5 (`value_oriented_programming.md`):
  every newly adopted rule must, at adoption time, name its mechanical gate
  (CTest entry, boundary/script check, or inventory rule); gateless rules are
  born guidelines, not law. Referenced by the review README standards list
  (`docs/review/README.md` item 5).

## G3 — Scheduled / low-cost items

- [x] **G3.1 C++26 native contracts threshold (P3)** — DONE 2026-09-18.
  Threshold formalized in `cpp26_native_switch_runbook.md` (threshold banner):
  C4.3 trigger = first toolchain with GCC ≥ 16 **or** Clang ≥ 20 defining
  `__cpp_contracts` (feature-test macro only, never a compiler-name check) plus
  the matching `<contracts>` library. C4.2 standing tracking
  (`contract_guardrails_adoption_todo.md`) records the formalization and the
  probe/re-check procedure.
- [x] **G3.2 Golden-value tests for math laws** — DONE 2026-09-18. New CTest
  `shs_renderer_math_law_golden_tests`
  (`tests/math_law_golden_tests.cpp`, guarded on `TARGET Jolt`): pins the
  screen/canvas discrete row law (`row_canvas = (H−1) − row_screen` golden
  table H=8, involution across H ∈ {1,2,3,5,480,1080}), the continuous-vs-
  discrete domain distinction (`y_canvas = H − y_screen` maps the top edge to
  y=H, outside row indices; pixel-center floor bridges the two), the
  rasterizer NDC mapping `(ndc*0.5+0.5)*(D−1)` pinned against the row law
  (NDC y=+1 → screen row H−1 → canvas row 0, top; LH_NO y-up projection),
  the Jolt bridge conjugation `to_jph(m) == S·M·S` (hand-pinned golden
  matrix, exact float equality), and the roundtrip involutions
  `to_glm(to_jph(M)) == M`, `to_jph(to_glm(M)) == M`, quaternion x/y- and
  Vec3 z-negation roundtrips, plus the homomorphism property
  `f(A·B) == f(A)·f(B)`. Note: `to_glm(to_jph(M))` is the identity by
  construction (both directions negate the same slots) — the golden S·M·S
  comparison must target `to_jph(m)` itself. CTest 48/48 green.
- [x] **G3.3 Terminology duality onboarding note (Tension 3 / P4)** — DONE
  2026-09-18: added §1.5 "Terminology duality — 'DVO in prose, pod on disk'" to
  `docs/education/dvo_bounded_context_tutorial.md` (the new-contributor "start
  here" doc), covering all four rules: T1 (DVO in all new prose — retired term
  in live text is review-blocking), T5 + Part 6 ("pod" stays as the structural
  noun for file names/dirs/gate globs — never rename paths, Rule N5), T2
  (archives stay byte-identical), T6 (retired term is a recognized spoken
  alias). Pointer added to the `docs/education/README.md` "start here" bullet.
  No path renames (Rule N5: stable paths, loud headers). Note itself is
  grep-clean against the Rule T3 tracking command (68 pre-existing occurrences
  unchanged — all in archives, defining documents, and demo HISTORY frames).
  DoD met: onboarding doc updated.

---

## Order & constraints

- G1.1 and G1.2 are independent; both are "prose → gate" moves and land first
  (G1.2 first — cheapest, knowledge fresh from the SDL2 cutover; then G1.1).
- G2.1–G2.3 are documentation/gate amendments; batch the Constitution II edits
  (G2.1 + G2.3) into one pass, Constitution III clarification (G2.2) separate.
- G3 items land opportunistically after G1/G2; none block anything.
- Archive documents (`kdba_history/`, FROZEN backlogs, `docs/outdated/`) are
  never edited by any item here.
