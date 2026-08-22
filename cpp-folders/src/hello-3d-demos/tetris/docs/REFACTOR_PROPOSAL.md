# Tetris Domain-Pod Refactor Proposal

Target architecture distilled from the fps demo refactor (verified pattern),
the VOP/DOD constitution (`docs/NOTES.md`), and the canonical blueprint (`docs/TODOS.md`).

## Current problems (evidence-based)

1. **Two divergent header sets.** Live code = root `tetris.*.hpp` (flat
   `namespace tetris`, included by main). Dead code = `domains/**` namespaced
   variants that nothing includes and which have drifted stale
   (`matrix.plan.hpp` even has a double-nested `namespace tetris::tetris` bug).
2. **God-snapshot.** `TetrisSnapshot` mixes three lifecycles: grid rulebook
   (grid/active/hold/queue/rng), scoring (score/high/lines/level/combo/target),
   and timing (gravity/danger pulse). One struct = one lifecycle per pod.
3. **Scoring leaks into the matrix reducer.** `reduce_tetris()` computes
   `base_scores[] * level + combo * 50 * level` inline (lines ~270–276) and
   ships `score_delta` inside `TetrisEvent`. Progression must *listen*, not be
   fused into the grid math (Constitution II Rule 8.1).
4. **Edges trapped in main.** ~833-line main contains the full audio synth +
   SPSC ring + callback, rasterizer helpers, the entire Mongolian-Cyrillic
   UTF-8 HUD engine, and inline SDL input polling.
5. **Lua edge orphaned.** `edges/lua.edge.hpp` is included nowhere; no
   `scripts/` directories exist.

## Proposed tree

```text
hello-3d-demos/tetris/
├── CMakeLists.txt                     # ONE -I root (tetris/), MSVC /GL+/LTCG+/MP (already done)
├── hello_3d_tetris.cpp                # thin main edge: SDL lifecycle, PMR arena, loop wiring,
│                                      # headless hooks (--screenshot/--frame/--autodrive-*), event→sound map
├── config/
│   ├── rules.hpp                      # tetris::config::Rules — gravity curve, lock delay/max resets,
│   │                                  # DAS (future), base score table, combo bonus, level-up cadence,
│   │                                  # target score, bag size, rng seed
│   └── levels/
│       └── marathon_01.hpp            # tetris::config::Marathon01 — victory target, start level,
│                                      # board frame palette, stage layout constants
├── domains/
│   ├── matrix/                        # POD 1 — grid rulebook (pure, no SDL, no scoring)
│   │   ├── matrix.contract.hpp        #   GridTable, ActivePiece, HoldSlot, NextQueue (7-bag),
│   │   │                              #   MatrixSnapshot {grid, active, hold, hold_locked, queue,
│   │   │                              #   rng_state, gravity_timer} + TetrisStepResult
│   │   ├── matrix.action.hpp          #   8 intents + TetrisCommandFrame (verbatim move)
│   │   ├── matrix.event.hpp           #   RAW FACTS ONLY: PIECE_SPAWNED/MOVED/ROTATED, LOCK_IMPACT,
│   │   │                              #   HARD_DROP_SLAM{dropped_cells}, LINES_CLEARED{rows[4],count},
│   │   │                              #   HOLD_SWAPPED, GAME_OVER   (no score_delta field!)
│   │   └── matrix.reducer.hpp         #   get_piece_blocks / is_valid_position / get_ghost_y /
│   │                                  #   pull_next_piece (7-bag LCG) / reduce_matrix(prev, cmds,
│   │                                  #   rules, speed_level, dt, arena) — speed_level arrives as a
│   │                                  #   plain function arg wired by main from progression state
│   ├── progression/                   # POD 2 — scoring & objectives (event-driven, never touches grid)
│   │   ├── progression.contract.hpp   #   ScoreState {score, high_score, lines_cleared, level,
│   │   │                              #   combo_count, target_score, victory}
│   │   ├── progression.event.hpp      #   DERIVED: SCORE_CHANGED{delta}, COMBO_STREAK{combo},
│   │   │                              #   LEVEL_UP{new_level}, OBJECTIVE_COMPLETED, VICTORY
│   │   └── progression.reducer.hpp    #   reduce_progression(span<matrix events>, prev, rules)
│   │                                  #   → ScoreState + progression events (base score table lives HERE)
│   └── spatial_fx/                    # POD 3 — render vocabulary + fx state
│       ├── spatial_fx.contract.hpp    #   LowPolyTriangle, MeshBuilder, ProcessedTriangle,
│       │                              #   PipelineExecutionPlan, ShatterParticleSoA, FxState
│       │                              #   {particles, camera_shake}, piece palette (get_piece_color)
│       ├── spatial_fx.reducer.hpp     #   step_fx(prev FxState, span<matrix events>, dt) — particle
│       │                              #   SoA integration + shatter spawn + shake spring decay
│       └── spatial_fx.plan.hpp        #   MeshGen (board frame, locked blocks, active piece, ghost)
│                                      #   + plan_tetris_scene(MatrixSnapshot, FxState, …) → plan
│                                      #   (reads matrix contract read-only — same as fps)
├── edges/
│   ├── input/tetris.input.hpp         # poll_input() → InputState; reduce_input() → commands
│   ├── audio/tetris.audio.hpp         # verbatim port of current 12-voice synth + SPSC ring
│   │                                  # (fps lesson: port recipes exactly, no "improvements")
│   ├── rasterizer/tetris.rasterizer.hpp # clip_to_screen + tile job (extract from main)
│   ├── ui/tetris.hud.hpp              # Mongolian UTF-8 font engine + draw_hud + hold/next panels
│   └── lua/lua.edge.hpp               # existing StatelessLuaEvaluator — UNWIRED until Phase 2
└── docs/
    ├── STATUS.md                      # verification log (build, headless checks)
    └── DETAILS.md                     # this architecture deep-dive after migration
```

Deliberately **NOT created yet**: `powerups/`, `environment/` (Red Flag 2 —
no live content exists; they become real pods in Phase 2 when bomb/laser
blocks and the reactive diorama land, each with scripts/ for Lua rules).

## Frame data flow (after refactor)

```
input.edge ─► span<TetrisCommand>
                 │
                 ▼
matrix.reduce_matrix(prev_matrix, cmds, rules, speed_level◄─main─progression.level, dt, arena)
                 │  → {MatrixSnapshot, [raw matrix events]}
                 ├──────────────────────────────┐
                 ▼                              ▼
progression.reduce(events, prev, rules)   spatial_fx.step_fx(prev_fx, events, dt)
                 │  → {ScoreState, [prog events]}   │  → FxState {particles, shake}
                 │                              ▼
                 │              spatial_fx.plan_scene(matrix, fx, …) → plan
                 │                              ▼
                 │              raster tile jobs ×N (thread pool)
                 │                              ▼
                 └────────────► ui.draw_hud(canvas, matrix, score)
audio map (main): matrix+progression events → synth.play(SoundType) [SPSC ring]
```

Pod-to-pod contact happens ONLY through: (a) event spans, (b) main passing
plain values (speed_level), (c) planners reading foreign contracts read-only.

## Migration phases (each ends green: builds + plays identically)

- **P0 — Delete dead weight.** Remove stale `domains/**` copies and root
  headers' duplicates-at-target; establishes single-source-of-truth baseline.
- **P1 — Extract edges from main** (mechanical, zero behavior change):
  audio → `edges/audio/`, rasterizer → `edges/rasterizer/`,
  HUD+font engine → `edges/ui/`, input polling → `edges/input/`.
  Main shrinks to ~250 lines of wiring.
- **P2 — Split the god-snapshot.** `config/rules.hpp` + `config/levels/`;
  carve `MatrixSnapshot` out of `TetrisSnapshot`; move scoring out of
  `reduce_tetris` into `progression::reduce_progression` (same numbers,
  now event-fed); move particles/shake into `spatial_fx::FxState`.
- **P3 — Rewire main + conventions.** Rooted angle-bracket includes
  (`<domains/matrix/matrix.contract.hpp>`), quotes for siblings, one `-I`
  root, namespaces `tetris::{matrix,progression,spatial_fx,input,audio,raster,ui}`.
  Delete root `tetris.*.hpp`.
- **P4 — Verify like fps.** Headless hooks (`--screenshot`, `--frame=N`,
  `--autodrive-harddrop`), determinism double-run byte-compare, signature
  color checks, `docs/STATUS.md` + `docs/DETAILS.md`.
- **P5 (optional later) — New pods + Lua.** `powerups/` (bomb/laser/freeze,
  `scripts/*.lua` via lua.edge), `environment/` (diorama, mood lighting);
  add `find_package(Lua)`/vcpkg entry when wiring lua.edge.

## Verification checklist (definition of done)

- [x] Only one definition of every type/function in the tree
- [x] Zero SDL/shs includes under `domains/` (grep-enforced purity)
- [x] Zero scoring arithmetic under `domains/matrix/`
- [x] Main < ~300 lines, contains no synth/raster/font code
- [x] Root `tetris.*.hpp` deleted
- [x] Deterministic headless run: two runs byte-identical
- [x] Behavioral delta: autodrive hard-drop frame differs from idle
- [x] Build clean on gcc (-Wall… -Wconversion) and ready for MSVC Release