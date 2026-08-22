# Tetris Domain-Pod Refactor — STATUS

**Date:** 2026-08-22 · **Branch state:** migration P0–P4 complete, all gates green

## §0 Docs relocated + consolidated (2026-08-22)

- Pass 1: this file and the as-built deep-dive moved from repo-root `docs/` into
  `tetris/docs/` (per-demo docs convention, mirroring snake; see
  `docs/dev/cpp_compilation_workflow.md` "per-demo canonical state").
- Pass 2 (same day): consolidated 5 files → 3:
  - `NOTES.md` (theory) + `DETAILS.md` (as-built) → **`ARCHITECTURE.md`**
    (Part I theory · Part II Lua philosophy · Part III as-built reference).
  - `REFACTOR_PROPOSAL.md` (completed migration record) → absorbed into the
    **History** section below.
  - Remaining set: `ARCHITECTURE.md` · `TODOS.md` (living tracker incl. the
    Part 4 level/mode campaign with GUI/FX variety) · `STATUS.md` (this file).
- As-built tree + frame dataflow are single-sourced in `ARCHITECTURE.md`
  Part III; `TODOS.md` links instead of duplicating.

## Build

| Item | Result |
|---|---|
| Configure | `cmake .` in `cpp-folders/build_vcpkg` — OK |
| Target | `cmake --build . --target Hello3DTetris -j22` — **0 errors** |
| Binary | `build_vcpkg/src/hello-3d-demos/tetris/Hello3DTetris` (~8.8 MB) |
| Flags | gcc `-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion -O3 -flto -march=native`; MSVC `/GL /LTCG /MP` ready |

## Headless verification (reproducible)

Run: `bash verify.sh` (project root) — uses `SDL_VIDEODRIVER=dummy`.

| Check | Command shape | Result |
|---|---|---|
| Determinism | two idle runs `--screenshot --frame=45`, byte-compare | **PASS** (identical BMPs) |
| Behavioral delta | idle vs `--autodrive-harddrop --frame=45` | **PASS** (frames differ) |
| Pod purity: platform refs under `domains/` | `grep -rl SDL domains/` | **NONE** |
| Pod purity: scoring refs under `domains/matrix/` | `grep -rn 'score\|combo' domains/matrix/` | **NONE** |
| Main edge size | `wc -l hello_3d_tetris.cpp` | **331 lines** (was 833) |

## Definition-of-done checklist

- [x] Only one definition of every type/function (root monolith headers deleted)
- [x] Zero SDL/shs includes under `domains/`
- [x] Zero scoring arithmetic under `domains/matrix/`
- [x] Main < ~350 lines, contains no synth/raster/font code
- [x] Root `tetris.*.hpp` deleted
- [x] Deterministic headless run byte-identical
- [x] Autodrive hard-drop frame differs from idle
- [x] gcc clean; MSVC Release flags wired

## History: domain-pod migration (completed 2026-08-22)

Absorbed from the former `REFACTOR_PROPOSAL.md`. Target architecture was
distilled from the fps demo refactor (verified pattern), the VOP/DOD
constitution (`ARCHITECTURE.md` Part I), and the canonical blueprint
(`TODOS.md` Part 1).

### Problems fixed (evidence-based)

1. **Two divergent header sets.** Live code = root `tetris.*.hpp` (flat
   `namespace tetris`, included by main). Dead code = `domains/**` namespaced
   variants that nothing included and which had drifted stale
   (`matrix.plan.hpp` even had a double-nested `namespace tetris::tetris` bug).
2. **God-snapshot.** `TetrisSnapshot` mixed three lifecycles: grid rulebook
   (grid/active/hold/queue/rng), scoring (score/high/lines/level/combo/target),
   and timing (gravity/danger pulse). One struct = one lifecycle per pod.
3. **Scoring leaks into the matrix reducer.** `reduce_tetris()` computed
   `base_scores[] * level + combo * 50 * level` inline and shipped
   `score_delta` inside `TetrisEvent`. Progression must *listen*, not be fused
   into the grid math (Constitution II Rule 8.1).
4. **Edges trapped in main.** ~833-line main contained the full audio synth +
   SPSC ring + callback, rasterizer helpers, the entire Mongolian-Cyrillic
   UTF-8 HUD engine, and inline SDL input polling.
5. **Lua edge orphaned.** `edges/lua.edge.hpp` was included nowhere; no
   `scripts/` directories existed.

### Migration phases (each ended green: builds + plays identically)

- **P0 — Delete dead weight.** Remove stale `domains/**` copies and root
  headers' duplicates-at-target; establishes single-source-of-truth baseline.
- **P1 — Extract edges from main** (mechanical, zero behavior change):
  audio → `edges/audio/`, rasterizer → `edges/rasterizer/`,
  HUD+font engine → `edges/ui/`, input polling → `edges/input/`.
- **P2 — Split the god-snapshot.** `config/rules.hpp` + `config/levels/`;
  carve `MatrixSnapshot` out of `TetrisSnapshot`; move scoring into
  `progression::reduce_progression` (same numbers, now event-fed); move
  particles/shake into `spatial_fx::FxState`.
- **P3 — Rewire main + conventions.** Rooted angle-bracket includes, quotes
  for siblings, one `-I` root, namespaces
  `tetris::{matrix,progression,spatial_fx,input,audio,raster,ui}`.
  Delete root `tetris.*.hpp`.
- **P4 — Verify like fps.** Headless hooks (`--screenshot`, `--frame=N`,
  `--autodrive-harddrop`), determinism double-run byte-compare, signature
  color checks, STATUS/DETAILS write-ups.
- **P5 (next) — New pods + Lua.** `powerups/` (bomb/laser/freeze,
  `scripts/*.lua` via lua.edge), `environment/` (diorama, mood lighting);
  add `find_package(Lua)`/vcpkg entry when wiring lua.edge.
  *(Now tracked concretely as TODOS.md Part 4 · L4/L5 and build order C/D.)*

Deliberately NOT created during the migration: `powerups/`, `environment/`
(Red Flag 2 — no live content existed; they become real pods in P5 when
bomb/laser blocks and the reactive diorama land, each with `scripts/` for Lua
rules).

## Notes / pitfalls encountered (for future migrations)

1. **CRLF defeated every `$`-anchored sed/awk extraction** from original files.
   Always `sed 's/\r$//'` (or strip `[ \t\r]*$`) before anchored matching.
2. **Ubuntu default awk is mawk**: `\b` word boundaries silently no-op. Use GNU
   `sed -E` for renames, or plain longest-first `gsub` chains.
3. **cmd.exe→wsl.exe inline `$var`s are unreliable**; prefer script files.
4. **Aggregator leaks sibling demo roots globally** (`fps` before `tetris`);
   fixed locally via target-scoped `INCLUDE_DIRECTORIES` on Hello3DTetris with
   the tetris root first. Consider scoping all four demos the same way later.
5. `rand()` debris velocities replaced by seeded xorshift inside `FxState` —
   required for determinism; visual behavior equivalent.