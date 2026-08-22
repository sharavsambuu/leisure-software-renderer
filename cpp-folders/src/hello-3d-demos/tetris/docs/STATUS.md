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
| Script economy override smoke | `--stage=2 --expect-target-score=20000` | **PASS** (Lua `get_config` override reaches `Rules`) |
| Blitz boots + determinism WITH scripting | two `--stage=2 --frame=45` runs, byte-compare | **PASS** (script active both runs) |
| Pod purity: platform refs under `domains/` | `grep -rl SDL domains/` | **NONE** |
| Pod purity: scoring refs under `domains/matrix/` | `grep -rn 'score\|combo' domains/matrix/` | **NONE** |
| Pod purity: raw Lua C-API outside `edges/lua/` | grep, comments stripped | **NONE** |
| Main edge size | `wc -l hello_3d_tetris.cpp` | **464 lines** (was 833 pre-pods; grew for campaign/script wiring) |

## §0b Build order A delivered: lua.edge wired + L2 Blitz 120 (2026-08-22)

- **Campaign manifest (M2 groundwork):** `config/campaign/main_campaign.hpp`
  registers stage 1 = MARATHON (pure C++) and stage 2 = BLITZ 120
  (`domains/progression/scripts/blitz_mode.lua`). `--stage=N` selects;
  `--script=<file>` overrides the manifest's script path.
- **Evaluator edge:** `edges/lua/lua.edge.hpp` owns a sandboxed
  `StatelessLuaEvaluator` (base/table/math only; `math.random*` and `print`
  stripped). Value-in/value-out only; the ONLY file including Lua headers.
- **Privilege seam:** pods stay Lua-free. `progression::ScriptHooks` carries
  plain function pointers (`line_clear_score`, `clock_rule`); null hooks ⇒
  native C++ rules. Main adapts evaluator calls to the hooks; blitz clock +
  time-up freeze live in the progression reducer behind those hooks.
- **Config-as-data:** `BlitzRules.get_config()` overrides plain `Rules` values
  at boot (target 12000 → 20000 asserted by the smoke gate).
- **L1/L2 presentation:** HudState banners/floaters/vignette/countdown/combo
  meter/RESULTS panel in the ui edge; shockwave rings on 30s ticks, hard-drop
  spark trails, tetris dolly punch + oversized gold-flecked bursts, victory
  golden burst, amber mood wire (`fx.mood_intensity`) in spatial_fx.
- **Build:** green WITH Lua (vcpkg `lua` 5.5, guarded `find_package(Lua QUIET)`
  + `TETRIS_LUA_ENABLED`) and without (empty header, native rules).

## §0c Playtest fixes: campaign advance + Mongolian text (2026-08-22)

User-reported after playing L1 → L2:

1. **Stale level-finished GUI over the next level.** Root cause: the session
   had no stage transition at all — finishing a stage left its victory modal
   up with nowhere to go. Fix = real campaign advance in main:
   - Victory on a non-final stage dwells 3s on the results modal, then
     auto-advances to the next manifest stage; R skips the dwell instantly.
   - `load_stage()` performs a FULL reset: fresh board + first piece, fresh
     `ScoreState` (session-high carried across stages), fresh `HudState`
     (banners/floaters cleared), fresh `FxState` (particles/rings/pulse/mood),
     fresh script sandbox (`apply_stage_script` re-runs per stage — no global
     leakage between stages), rules reloaded from the manifest factory,
     window title updated.
   - Manual restart also clears `HudState` now.
   - Blitz time-up freeze thawed for exactly one frame when R is pressed so
     the reset reaches the board too.
2. **Wrong Mongolian words.** Fixed in the ui edge label table:
   - `TXT_LEVEL`: bytes were `\xD0\xAE\xD0\x95` = "ЮЕ" (nonsense) → now
     `\xD2\xAE\xD0\x95` = "ҮЕ" (үе = level/stage).
   - `TXT_HURRY`: "ХУРДАА!" is not a word → "ХУРДАЦГАА!" (hurry up!).
   - `TXT_MAX_COMBO`: slangy "МАКС КОМБО" → "ДЭЭД КОМБО" (best combo;
     consistent with ДЭЭД on the score card; fits the RESULTS row width).
   - Combo floater now uses Cyrillic "КОМБО ×N" instead of Latin "COMBO".

Gates re-run after both fixes: DETERMINISM / DELTA / SMOKE_TARGET_SCORE / BLITZ_DETERMINISM PASS; purity NONE ×3.
## §0d M1 session pod + menus + RESULTS screen (2026-08-23)
New pod domains/session/ (contract/action/reducer): pure TITLE/LEVEL_SELECT/PLAYING/PAUSED/RESULTS state machine over SessionSnapshot (screen, cursors, unlocks, sound pref, last-run latch). Windowed boots to TITLE; headless verification still skips menus straight into PLAYING (gates unchanged).
- input edge emits session::SessionCommand intents alongside matrix commands (W/S/A/D nav, ENTER/SPACE confirm, ESC back-or-pause, P pause, M sound; key-repeat guarded on menus).
- audio edge gained menu move/confirm blips + set_enabled() master gate wired to the sound pref.
- ui edge gained pure menu projections: title attract (drifting tetromino silhouettes), level-select carousel (name/tier-tag/dots), pause overlay, RESULTS breakdown with contextual first row.
- main steps reduce_session() before gameplay pods; pods run only while PLAYING; STAGE_SELECTED/RUN_RESTART drive load_stage() FULL resets; QUIT_REQUESTED exits; run-end latch hands victory/time-up/game-over facts to RESULTS; session-high score survives restarts.
Gates re-run: DETERMINISM / DELTA / SMOKE_TARGET_SCORE(20000) / BLITZ_DETERMINISM PASS; purity NONE ×3 (session pod included).

## Definition-of-done checklist

- [x] Only one definition of every type/function (root monolith headers deleted)
- [x] Zero SDL/shs includes under `domains/`
- [x] Zero scoring arithmetic under `domains/matrix/`
- [x] Main < ~600 lines (campaign + session wiring added), contains no synth/raster/font code
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
6. **Lua headers must be included via `<lua.hpp>`**, not `<lua.h>`: the plain
   C header lacks `extern "C"`, so every `lua_*` reference got C++-mangled and
   failed to link against `liblua.a` despite correct link lines.
7. **A failed link leaves a zero-filled output file** that is NEWER than the
   objects — make then reports "up to date"/"Built target" without relinking.
   After any link failure, delete the output before rebuilding.
8. **`FxState` must be constructed with `std::pmr::get_default_resource()`**,
   never the per-frame arena: its particles/rings outlive frames, and the
   frame arena resets every tick (silent corruption otherwise).
9. **vcpkg classic mode picks the DEBUG `liblua.a` when `CMAKE_BUILD_TYPE`
   is empty** — harmless here (symbols identical), but pin a build type if
   release-only linking ever matters.
