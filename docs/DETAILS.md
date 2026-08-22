# Tetris Domain-Pod Architecture — DETAILS

Companion to `REFACTOR_PROPOSAL.md` (the plan) and `STATUS.md` (verification).
This is the as-built deep-dive, including the **Lua integration design** for
the next task.

## 1. As-built tree

```text
tetris/
├── hello_3d_tetris.cpp        # main edge (331 lines): SDL lifecycle, arena, wiring,
│                              # event→sound map, headless hooks
├── verify.sh                  # reproducible headless verification battery
├── CMakeLists.txt             # single -I root; target-scoped include order
├── config/
│   ├── rules.hpp              # tetris::config::Rules — every gameplay number
│   └── levels/marathon_01.hpp # tetris::config::Marathon01 — level definition
├── domains/
│   ├── matrix/                # contract / action / event / reducer (pure grid rulebook)
│   ├── progression/           # contract / event / reducer (event-fed scoring)
│   └── spatial_fx/            # contract / reducer / plan (vocabulary + fx + planner)
├── edges/
│   ├── input/tetris.input.hpp     # SDL polling → intent tokens
│   ├── audio/tetris.audio.hpp     # verbatim synth port (12 voices, SPSC ring)
│   ├── rasterizer/tetris.rasterizer.hpp # screen-space helpers (tetris::raster::vop)
│   ├── ui/tetris.hud.hpp          # Mongolian UTF-8 font engine + draw_hud(m, sc)
│   └── lua/lua.edge.hpp           # StatelessLuaEvaluator — UNWIRED (see §4)
└── docs/                      # REFACTOR_PROPOSAL.md · STATUS.md · DETAILS.md
```

## 2. Frame dataflow (as built)

```
input::poll_input(arena) ─► InputState{quit, span<TetrisCommand>}
        │
        ▼
world.drop_interval = rules.gravity_for_level(score.level)   ← main-edge wiring
        │
matrix::reduce_matrix(world, commands, dt, arena)
        │  → MatrixStepResult{MatrixSnapshot, pmr::vector<MatrixEvent>}   RAW FACTS
        ├──────────────────────────────────────────┐
        ▼                                          ▼
progression::reduce_progression(events, …)   spatial_fx::step_fx(fx, events, dt)
        │  → ScoreState + ProgressionEvents         │  (particles, shake, fx clock)
        │                                           ▼
        │                          spatial_fx::plan_tetris_scene(world, fx, W, H, arena)
        │                                          │  → PipelineExecutionPlan
        │                                          ▼
        │                          tiled raster jobs ×N (raster::vop helpers)
        ▼                                          ▼
ui::draw_hud(canvas, world, score_state)    present / screenshot
audio map (main): MatrixEvent → audio::SND_* via SPSC ring
```

Contact rules honored: pods talk only through **event spans**, **plain values
wired by main** (`drop_interval`, restart preservation), and the planner's
**read-only** access to the matrix contract.

## 3. Ownership map (what moved where)

| Concern | Was | Now |
|---|---|---|
| Grid/piece/hold/bag/RNG state | `TetrisSnapshot` god-struct | `matrix::MatrixSnapshot` |
| Score/high/lines/level/combo/target/victory | same struct + inline math in `reduce_tetris` | `progression::ScoreState`; scoring derived from `MatrixEvent`s in `reduce_progression` |
| Base score table `{0,100,300,500,800}`, combo bonus, gravity curve, lock constants | hardcoded in reducer/main | `config::Rules` (+ `gravity_for_level`) |
| Gravity cadence updates on level-up | inside matrix reducer | main wires `rules.gravity_for_level(score.level)` each frame; reducer only reads `drop_interval` |
| Soft/hard drop points | silent `s.score += n` | raw facts: `SOFT_DROP{cells}` / `HARD_DROP_SLAM{cells}` → progression converts |
| Combo break on sterile lock | matrix field | progression watches `PIECE_LOCK_IMPACT` w/o clear in same frame |
| High-score preservation across reset | matrix snapshot copy | main edge captures/restores around restart |
| Shatter particles + camera shake + sway clock | main locals + plan args | `spatial_fx::FxState` (stepped by `step_fx`, read by planner) |
| Debris randomness | `rand()` (non-deterministic) | seeded xorshift in `FxState` |
| Piece palette | grid contract | `spatial_fx::get_piece_color` (render vocabulary) |
| Audio synth/ring/callback, raster helpers, HUD+font engine, input polling | ~600 lines inside main | `edges/{audio,rasterizer,ui,input}` |

## 4. Lua integration design (NEXT TASK)

The architecture already reserves the seams. Intended shape, aligned with
the tetris demo's `docs/TODOS.md` Part 2 and the fps lua lessons:

**4.1 Edge placement.** `edges/lua/lua.edge.hpp` stays the ONLY file that
includes Lua headers. Domains never see `lua_State*`. The evaluator remains
stateless-per-call: script text + plain-value inputs → plain-value outputs.

**4.2 First swap: the scoring rule.**
`progression::compute_line_clear_score(rules, lines, level, combo)` is a pure
value-in/value-out function isolated for exactly this. Phase 1 of Lua work:

- Add `config/rules.lua` (or `scripts/scoring.lua`) returning
  `{ base = {...}, combo_bonus = n, hard_drop_per_cell = n, ... }`.
- Main edge loads it once at boot through `lua.edge` into a `config::Rules`
  instance — config stays a plain struct; Lua is just an authoring format.
- Optional Phase 1b: route per-clear evaluation through a Lua function
  `compute_score(lines, level, combo) -> int`, called from
  `reduce_progression`. Keep the C++ path as fallback when no script is set.
  Determinism note: same script + same inputs must give identical outputs;
  no os/io/random in rule scripts (enforce by opening a sandboxed luaL).

**4.3 Future pods arrive WITH their scripts.**

- `domains/powerups/` (bomb/laser/freeze): pod contract holds powerup state;
  trigger/effect rules live in `scripts/powerups/*.lua` evaluated via lua.edge
  against `(matrix facts, powerup state) -> commands/events`. The pod's reducer
  applies returned intents; matrix stays untouched.
- `domains/environment/`: mood/difficulty curves authored in
  `scripts/environment/*.lua` (e.g., backdrop hue as f(score, danger)).
- Event bus extension: main may forward selected `ProgressionEvent`s to a
  `on_event(type, values)` Lua hook for achievements/objectives without any
  domain knowing about Lua.

**4.4 Build wiring (when activated).** vcpkg `"lua"` entry +
`find_package(Lua REQUIRED)` guarded so the demo still builds without it
(follow the SDL2_image optional pattern in the aggregator). Sources list gains
`scripts/*.lua` for IDE visibility only.

**4.5 Verification additions.** Extend `verify.sh` with:
`--script <file>` flag on main → loads a Lua rules file; determinism double-run
must stay byte-identical WITH scripting active; a `blitz_mode.lua` smoke test
(target score override) asserting `ScoreState.target_score` changes.

## 5. Conventions enforced here (fps-standard)

- Includes: root-relative angle brackets across pods (`<domains/matrix/…>`,
  `<config/rules.hpp>`); quotes ONLY for same-directory siblings
  (`"shs_renderer.hpp"`). No `../../` chains anywhere.
- Namespaces: one nested level per pod — `tetris::{matrix,progression,
  spatial_fx,input,audio,raster,ui,config}`; cross-pod visibility via explicit
  `using tetris::matrix::X;` declarations at the consumer (never global
  using-directives in headers outside `main`).
- Purity gates (CI-able greps): no `SDL` under `domains/`; no `score|combo`
  under `domains/matrix/`.
- Include-path hygiene: aggregator leaks sibling roots globally, so
  Hello3DTetris carries a target-scoped `INCLUDE_DIRECTORIES` with its own root
  FIRST (see STATUS.md pitfall 4). Other demos should adopt this later.