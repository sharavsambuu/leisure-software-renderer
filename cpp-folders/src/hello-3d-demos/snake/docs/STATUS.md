# snake/STATUS.md — Canonical Reference for Future Agents

> **Read this file first.** It is the single source of truth for the `snake` demo: what it is, how it is structured (VOP/DOD domain pods), what the current on-disk state actually is, and every known inconsistency. Do NOT trust `IMPLEMENTATION_PLAN.md` — it was written by an earlier agent and does not match reality.
> **Then read `DETAILS.md`** (same folder) — symptom→root-cause playbook for the coordinate-system/camera/rendering gotchas this demo has already paid for (GLM lookAtLH mirror quirk, winding coupling, depth-bias burial, idle-decay, pixel forensics).

---

## 0. Agent Log / Session Notes

### 2026-08-22 (latest) — DOCS RELOCATED + DUPLICATION PURGE ✅

Per user request:
1. **Docs moved into `snake/docs/`**: `STATUS.md`, `IMPLEMENTATION_PLAN.md` (git mv), and
   `LESSONS_LEARNED.md` renamed → **`docs/DETAILS.md`** (plain mv; was untracked). All in-repo
   references updated (this header, tree below, CMakeLists comment, docs/dev/cpp_compilation_workflow.md).
2. **Duplication / structure purge** — origin domain-pod layout is now STRICT (config/ at root;
   domains = matrix{contract,action,event,reducer} + progression{contract,reducer} +
   spatial_fx{contract,plan}; edges = input,audio,rasterizer). Deleted (recoverable from git history):
   - `domains/config/difficulty.hpp` — stale duplicate of root `config/difficulty.hpp` (lacked defaults)
   - `domains/matrix/snake.plan.hpp` — a PLAN does not belong in the matrix domain; wrong tetris-style copy
   - `domains/spatial_fx/snake.reducer.hpp` — a REDUCER does not belong in spatial_fx; broken/dead
   - `domains/environment/snake.contract.hpp` — entire extra pod absent from the origin structure (unwired MoodState)
   - `_dump.sh` — scratch script
   Verified pre-delete: ZERO includes referenced any of them. Kept: `edges/audio` + `edges/rasterizer`
   (origin-structure slots, deliberately unwired until assets/canonical-loop decision).
3. **Dev-artifact gitignore policy** — root `.gitignore` now ignores `cpp-folders/_diag_*/` plus
   `cpp-folders/**/*.bmp|log`: the pixel-forensics suites are dev-local tooling, NOT repo content.
   `snake/.gitignore` refreshed (stale `_dump.sh` entry removed; now ignores demo-local *.bmp/*.log).
   DETAILS.md §5 documents each script so they can be recreated if missing.

### 2026-08-22 — CONTROLS + MIRROR + FOOD: three root causes found & fixed ✅

User report: "controls feel like the camera is north behind the board facing south." Traced the full
input→command→direction→render chain and ran deterministic autodrive experiments. THREE real bugs:

1. **UP/DOWN inverted (control side)** — `matrix/snake.action.hpp` still had tetris-wall semantics
   (`UP → dy -= 1`) from before the floor remap. With floor mapping (x,y)→(x,0,-y), grid +y renders
   AWAY from the front camera = screen-UP, so ArrowUp must be `dy += 1`. Fixed + documented contract.
2. **HORIZONTAL MIRROR (render side)** — this GLM's `lookAtLH` builds the side vector as
   `s = cross(up, f)` → for our front camera s = (-1,0,0): the ENTIRE view rendered horizontally
   MIRRORED (world +X on screen-left). Proven empirically: snake gradient head (grid x=9) rendered
   LEFT of tail (x=7); autodrive ArrowRight moved the bar screen-LEFT. This is why controls felt
   "viewed from behind". FIX: hand-rolled view basis in the plan (`right = cross(fwd, up)` = (+1,0,0),
   rows [r;u;f] + translation column) + rasterizer front-face flipped to **Clockwise** (un-mirroring
   flips screen-space winding; CCW culled EVERYTHING — verified by an empty frame). Do NOT revert to
   glm::lookAtLH. Post-fix pixel positions match hand-computed projection EXACTLY (head 624 vs 624.6).
3. **Food invisible** — food box centered at y=0 (top 0.45) lost every depth test against its own
   tile because its depth_bias (+0.06) exceeded the tiny NDC gap to the tile top (y=0.25). Zero orange
   pixels in frame dumps. FIX: lift food center to y=0.45 (top 0.9 clears tiles) + bias 0.

Also fixed en route: **idle-decay bug** — with no input, delta=(0,0) fell through to the normal-move
path (new_head == head_pos), vacating the tail + duplicating the head EVERY FRAME: an idle snake
decayed to one segment in seconds (same for soft-wall holds). Reducer now returns snap unchanged for
no-intent ticks and soft-wall holds.

**Verification hooks added** (main): composable flags `--screenshot`, `--frame=N`, `<path>`,
`--autodrive-up`, `--autodrive-right` (inject ONE synthetic intent at frame 60). Diag scripts in
`cpp-folders/_diag_snake/`: `compare_runs.py` (idle-stability A==B + ArrowUp bbox-top rise),
`bbox_snake.py`, `snake_ends.py` (gradient direction = mirror detector), `find_food.py`.
Final suite ALL PASS: idle@58 ≡ idle@70 (no decay); ArrowUp bbox-top rose 16px; ArrowRight bar
shifted RIGHT one pitch; head-cyan RIGHT of tail; food visible at grid(3,3)→screen(379,506);
analyzer PASS (board centered 0.50).

### 2026-08-22 — CAMERA: fixed front-facing, tilted down ✅

User request: camera in FRONT of the board facing down at it (replaces the body-length-driven orbit).
`plan_snake_scene` now parks a deterministic camera on the +Z side of the floor (the grid's y=0 row):
`eye = arena_center + (0, 13, 17)`, `target = arena_center + (0, 0, -1)` (aimed a touch past center),
up = +Y. (NOTE: this entry's claim "screen-right = +X via lookAtLH" was WRONG — see the newer
CONTROLS + MIRROR + FOOD entry: lookAtLH mirrors X; the view basis is now hand-rolled.)
Grid y reads near→far (away from camera = up-screen). Analyzer PASS after rebuild:
bbox 940x420, centroid (0.50, 0.63), fill 1.00.
Camera is fully deterministic per frame — no orbit yaw, no drift.

### 2026-08-22 (later) — COORDINATE SYSTEM FIX: wall → floor ✅

User screenshot showed the board rendered EDGE-ON as a vertical wall. Root cause vs Constitution I
(`docs/spec/conventions.md`: LH space, +Y up, +Z forward): the plan mapped grid (x,y) → world
(x, y, 0) = a VERTICAL WALL, while the camera was a top-down orbit expecting a FLOOR. Fixed:

1. **Floor mapping** — plan now maps board (x,y) → world **(x, 0, -y)** (proper rotation about +X,
   det=+1). A reflection (x, 0, y) would flip triangle winding → backface culling/mirrored lighting.
   Tiles, snake segments, food all remapped; boxes extrude along +Y (heights unchanged).
2. **`level.arena_center`** → (9.5, 0, -9.5) — floor center; orbit camera targets it (unchanged form).
3. **Aspect ratio fixed** — was hardcoded 1.0 (horizontal squeeze); now canvas_w/canvas_h passed into
   `plan_snake_scene(..., int canvas_w, int canvas_h)` mirroring tetris.plan.hpp.
4. **Death shatter FX remapped** to the floor plane (pos y≈0.4, velocity pops +Y, gravity -Y).
5. **Verification hook added**: `./Hello3DSnake --screenshot [bmp] [frame]` renders N frames, saves a
   BMP, exits. Analyzer: `python3 cpp-folders/_diag_snake/analyze_frame.py <bmp>` — verifies the board
   reads as a wide centered top-down blob (PASS) not an edge-on band (FAIL). Gotcha encoded there:
   SDL_PIXELFORMAT_RGBA32 memory bytes are B,G,R,A on little-endian.
6. Rebuilt clean (0 errors / 0 snake-file warnings); analyzer RESULT: PASS (bbox 1028x504, aspect 2.04,
   centered, fill 1.00/1.00).

### 2026-08-22 — FULL BUILD PASSES ✅

`Hello3DSnake` compiles and links with **0 errors and 0 warnings from snake's own files** via:

```bash
wsl.exe -d Ubuntu-24.04 --cd /home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/build_vcpkg -- bash -c "cmake . > /tmp/cfg.log 2>&1 && cmake --build . --target Hello3DSnake -j$(nproc)"
# binary: cpp-folders/build_vcpkg/src/hello-3d-demos/snake/Hello3DSnake
```

(Only pre-existing `-Wshadow`/`-Wsign-conversion` warnings from the SHARED `shs_renderer.hpp` remain — they affect every demo equally and are out of scope here.)

**Fixes applied this session (Option B — minimal reconcile, chosen and completed):**

1. **Contract filename collision (root cause of most failures)** — bare `"snake.contract.hpp"` from
   `domains/spatial_fx/snake.plan.hpp` resolved SAME-DIRECTORY to the spatial_fx contract instead of the
   matrix one (same-dir quote-include beats `-I` order). Renamed
   `domains/spatial_fx/snake.contract.hpp` → `domains/spatial_fx/spatial_fx.contract.hpp` (git mv,
   matches tetris naming). Plan now includes both contracts via bare names resolved through `-I`.
2. **Sibling namespace lookup** — plan lives in `snake::spatial_fx`; added explicit
   `using snake::matrix::SnakeSnapshot; using snake::matrix::SnakeCommand;` (unqualified lookup never
   searches sibling namespaces).
3. **Matrix reducer rewritten** (`domains/matrix/snake.reducer.hpp`) — was broken: unquoted include
   (`#include snake.event.hpp`), missing difficulty include, wrong cell scale (0.5-unit cells vs 1-unit),
   no food spawn on eat, growth appended at the tail instead of the head. Now: grid==world coords
   (`vec3(x,y,0)`), deterministic food-table cycling, correct grow-at-head movement, soft/solid wall
   handling, self-collision excluding the vacating tail.
4. **Level data fixed** — `body_spawn` declared 3 slots for 2 segments; `std::array<glm::ivec2,N>`
   initializers converted to DOUBLE braces (single-brace elision fails on GCC: "too many initializers").
5. **Main entry wired** — includes matrix reducer + progression reducer; snapshot initialized from level
   spawn data; scoring moved into the pure `reduce_progression` pod (was hand-rolled in main); death
   shatter FX emitted once on the alive→dead transition with high-score capture.
6. **Plan corrected** — tiles/food drawn at grid==world coords (were offset by arena_center twice);
   removed invalid `snap.alive` particle block (snapshot has no alive flag); `add_quad` moved above
   `add_box` (used-before-declared); unused params marked `(void)`.
7. **CMakeLists** — added `<snake>/config` include dir (bare `"difficulty.hpp"`); listed renamed
   spatial_fx contract; pruned stale comment about dead dirs.
8. **Warning hygiene** — all `-Wconversion/-Wsign-conversion/-Wnarrowing` warnings in snake files fixed
   (size_t loops, `static_cast<uint8_t>` Color channels, SDL Uint32 casts, ptrdiff_t iterator math).
9. **SDL fix** — `SDL_QuitAudio()` does not exist → `SDL_QuitSubSystem(SDL_INIT_AUDIO)`.

### 2026-08-21 — Edges verified standalone (historical)

Input + rasterizer edges compiled in isolation; full build was blocked by VMA/vcpkg config and cross-pod
include issues (both since resolved — see above; builds now use `build_vcpkg`, where vcpkg deps resolve).

---

## 1. Project Overview

Semi-3D software-renderer snake demo built on top of the SHS Renderer (`shs_renderer.hpp`).
Scene: a FIXED front-facing camera (parked on the +Z side of the floor, elevated, tilted down at the
arena center, UNMIRRORED view) watching a low-poly snake glide across a checkerboard arena; food spawns
from a deterministic table; eating grows the tail (+10 score, speed ramp). Rendering mirrors the
**canonical tiled rasterization path** used by every other demo (see §4).

- Target executable: `Hello3DSnake`
- Entry point: `hello_3d_snake.cpp::main`
- Build/run:
  ```bash
  wsl.exe -d Ubuntu-24.04 --cd /home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/build_vcpkg -- bash -c "cmake . && cmake --build . --target Hello3DSnake -j$(nproc)"
  ./cpp-folders/build_vcpkg/src/hello-3d-demos/snake/Hello3DSnake
  ```
- Controls: arrows steer (grid-aligned turns), ESC quits. Soft walls by default (head holds position
  at the boundary); flip `difficulty.solid_walls` for lethal walls.

---

## 2. Architecture (VOP / DOD Domain Pods) — mirrors tetris

```
snake/
├── CMakeLists.txt                 # Hello3DSnake target; lists every compiled source
├── docs/
│   ├── STATUS.md                  # ✅ THIS FILE — canonical reference for agents
│   ├── DETAILS.md                 # symptom→root-cause playbook + verification recipes (read 2nd)
│   └── IMPLEMENTATION_PLAN.md     # historical intent only (out of date — trust STATUS.md)
├── hello_3d_snake.cpp             # main: SDL2 window + game loop (execution boundary)
├── config/
│   ├── difficulty.hpp             # snake::config::Difficulty (solid_walls) — on -I path
│   └── levels/snake_level_01.hpp  # SnakeLevel01: GRID 20x20, spawns, food_table[8] — on -I path
├── domains/
│   ├── matrix/
│   │   ├── snake.contract.hpp     # CORE TYPES: SnakeCommand(Type), BodySoA, FoodState, SnakeSnapshot, events, SnakeStepResult
│   │   ├── snake.action.hpp       # reduce_snake_commands(span<const SnakeCommand>) -> vec2 delta (pure)
│   │   ├── snake.event.hpp        # re-export of shared vocabulary
│   │   └── snake.reducer.hpp      # reduce_snake(snap, commands, difficulty, level) -> SnakeStepResult (pure)
│   ├── spatial_fx/
│   │   ├── spatial_fx.contract.hpp # PipelineExecutionPlan{triangles, view/proj/vp}, ProcessedTriangle, ShatterParticleSoA
│   │   └── snake.plan.hpp         # plan_snake_scene(snap, commands, difficulty, level, particles, canvas_w, canvas_h) -> plan (pure)
│   └── progression/
│       ├── snake.contract.hpp     # ScoreState{score, high_score, length, speed_mult} + fresh()
│       └── snake.reducer.hpp      # reduce_progression(events, prev) -> ScoreState (pure)
└── edges/
    ├── input/snake.input.hpp      # InputState + reduce_input(InputState&, arena) -> pmr::vector<SnakeCommand>
    ├── audio/snake.audio.hpp      # play_sfx(type, state) — DEAD until assets/snake/sfx_*.wav exist
    └── rasterizer/snake.rasterizer.hpp  # standalone edge (not wired into main; main has its own tiled loop)
```

**Frame data flow (per tick):**
SDL poll → `input::reduce_input` → `matrix::reduce_snake` (pure) → `progression::reduce_progression`
(pure, consumes events) → death FX emission (edge, one-shot) → particle update →
`spatial_fx::plan_snake_scene` (pure) → tiled parallel rasterization → present.

**Coordinate convention (Constitution I, fixed 2026-08-22):** one grid cell == one world unit in
BOARD space. `BodySoA` stores board coords as floats (exact by construction); `cell_to_world` is the
identity into board space. The PRESENTATION layer (spatial_fx plan + main's death FX) maps board
(x,y) → world **(x, 0, -y)** — a proper rotation about +X (det=+1), so the board is a FLOOR in the
XZ plane per Constitution I (+Y up, +Z forward). Do NOT use (x, 0, y): that reflection flips triangle
winding (backface culling / mirrored lighting). `level.arena_center` = (9.5, 0, -9.5) is the floor
center the FIXED front-facing camera targets (`eye = center + (0, 13, 17)` from the +Z side).
Projection aspect = canvas_w/canvas_h (was hardcoded 1.0 → squeeze).
**Screen-space control contract:** grid +x = screen-RIGHT, grid +y = screen-UP (away from camera).
The plan's view basis is HAND-ROLLED because this GLM's lookAtLH mirrors X (s = cross(up,f) =
(-1,0,0) here); main's rasterizer therefore culls **Clockwise** front faces. See spatial_fx/snake.plan.hpp.
**Visual verification:** `./Hello3DSnake --screenshot /tmp/snake_frame.bmp 60` renders 60 frames,
saves a BMP, exits; analyze with `python3 cpp-folders/_diag_snake/analyze_frame.py /tmp/snake_frame.bmp`
(checks: wide centered blob = floor; thin diagonal band = wall regression). Note: BMP bytes are
B,G,R,A order (SDL_PIXELFORMAT_RGBA32 = ARGB8888 packing on LE) — the analyzer handles this.

**Key VOP/DOD rules obeyed:** pure reducers/actions/plan (no SDL, deterministic); `vop::FrameMemoryResource`
pmr arena for per-frame command vectors; SoA particles; canonical renderer API only (§4).

---

## 3. Current On-Disk File Inventory

### Compiled (in CMakeLists.txt — all build clean)
- `hello_3d_snake.cpp` (~290 lines) — main loop; includes all pods by relative path.
- `domains/matrix/{snake.contract,snake.action,snake.event,snake.reducer}.hpp`
- `domains/spatial_fx/{spatial_fx.contract,snake.plan}.hpp`
- `edges/input/snake.input.hpp`
- `domains/progression/snake.contract.hpp`

### Stray-file purge (2026-08-22): COMPLETE — none remain
Deleted via git rm (recoverable from history): `domains/config/difficulty.hpp`,
`domains/matrix/snake.plan.hpp`, `domains/spatial_fx/snake.reducer.hpp`,
`domains/environment/snake.contract.hpp`, `_dump.sh`. Zero includes referenced them.

### Unwired but ORIGIN-STRUCTURE files (kept deliberately)
- `domains/progression/snake.reducer.hpp` — COMPILED (included by main); header-only so not listed in
  CMakeLists sources (harmless).
- `edges/audio/snake.audio.hpp` — origin edge slot; DEAD until `assets/snake/sfx_*.wav` exist.
- `edges/rasterizer/snake.rasterizer.hpp` — origin edge slot; compiles standalone; main uses its own
  tiled loop instead (kept for reference/experimentation).

---

## 4. Canonical Renderer API (shs_renderer.hpp) — DO NOT INVENT NEW TYPES

- `shs::Raster::FrustumClipPolygon` + `clip_triangle_to_frustum(c0, c1, c2)` — frustum clip.
- `ProcessedTriangle { glm::vec4 c0,c1,c2; shs::Color lit_color; float depth_bias; }` inside
  `PipelineExecutionPlan { std::pmr::vector<ProcessedTriangle> triangles; ... }` — THE render-ready format.
- Supporting types: `shs::Canvas` (ctor `(w,h,bg)`, `buffer().clear(Color)`, `draw_pixel_screen_space`,
  static `copy_to_SDLSurface(SDL_Surface*, Canvas*)`), `shs::ZBuffer(w,h,zn,zf)` + `clear()` +
  `test_and_set_depth_screen_space`, `shs::Color{r,g,b,a}`, `shs::rgb01_to_color(vec3)`,
  `shs::Job::ThreadedPriorityJobSystem` / `WaitGroup`. GLM left-handed (`lookAtLH`, `perspectiveLH_NO`).
- NOTE: canvas size/tile constants are NOT in the renderer header — each demo defines its own
  (`CANVAS_WIDTH/HEIGHT`, `TILE_SIZE_X/Y` in main).

---

## 5. Canonical Reference: tetris

`cpp-folders/src/hello-3d-demos/tetris/` remains the clean reference for pod layout, root-level shared
contract naming, `spatial_fx.contract.hpp` naming, plan shape, and main-loop structure. Snake now follows
it closely; remaining deltas are cosmetic (tetris keeps shared vocab at demo root, snake keeps it in
`domains/matrix/` — both work because of how `-I` dirs are ordered).

---

## 6. Known Bugs / Remaining Work

1. **No game-over reset/restart** — after death the snake freezes in place (particles play out);
   there is no R-to-restart or auto-reset. Reducer returns the unchanged snapshot when dead.
2. **Audio dead** — `edges/audio/snake.audio.hpp` needs real `assets/snake/sfx_*.wav` files plus a
   main-loop hook (tetris synthesizes audio procedurally — mirror that instead of wav assets).
3. **Progression speed ramp unused** — `score_state.speed_mult` is computed but the tick rate is fixed;
   wire it into the movement accumulator when adding time-based stepping.
4. **`docs/IMPLEMENTATION_PLAN.md` out of date** — describes an architecture that never existed;
   rewrite or delete it (moved to docs/, kept as historical record for now).
5. **Shared renderer warnings** — pre-existing `-Wshadow`/`-Wsign-conversion` inside `shs_renderer.hpp`;
   needs a dedicated renderer-hygiene pass (affects all demos).

---

## 7. Design Decisions Worth Remembering

- **Option B (minimal reconcile) was executed** on 2026-08-22 rather than the full tetris-mirror
  refactor (Option A). The multi-file pod layout was kept; only broken pieces were fixed.
- **Contract naming rule**: within one demo, a bare include name must resolve to EXACTLY ONE file
  across all same-dir + `-I` locations. Two `snake.contract.hpp` files broke this silently — hence the
  `spatial_fx.contract.hpp` rename. See `docs/dev/cpp_compilation_workflow.md` §C++ pitfalls.
- **Death FX ownership**: emitted by the main entry edge (one-shot on alive→dead transition), NOT by
  the plan — `SnakeSnapshot` intentionally carries no `alive` flag; liveness travels via
  `SnakeStepResult.alive`.

---

## 8. Quick Reference: Key Types & Signatures (current state)

| Type | Location | Notes |
|------|----------|-------|
| `SnakeCommandType` | matrix/snake.contract.hpp | LEFT/RIGHT/UP/DOWN |
| `SnakeSnapshot` | matrix/snake.contract.hpp | head_pos(ivec2), head_dir(vec2), food(FoodState), body(BodySoA) |
| `SnakeStepResult` | matrix/snake.contract.hpp | next_state + events(pmr::vector) + alive |
| `reduce_snake_commands` | matrix/snake.action.hpp | span<const SnakeCommand> -> vec2 delta |
| `reduce_snake` | matrix/snake.reducer.hpp | (snap, commands, difficulty, level) -> SnakeStepResult |
| `cell_to_world` | matrix/snake.reducer.hpp | (level, x, y) -> vec3(x, y, 0) — identity into BOARD space; floor mapping (x,0,-y) lives in the plan + main FX |
| `advance_food` | matrix/snake.reducer.hpp | cycles level.food_table deterministically |
| `SnakeLevel01` | config/levels/snake_level_01.hpp | GRID 20x20, food_table[8], arena bounds, rng seed |
| `Difficulty` | config/difficulty.hpp | solid_walls (sole copy — duplicate purged 2026-08-22) |
| `ScoreState` / `reduce_progression` | progression/{contract,reducer}.hpp | event-driven scoring + speed ramp |
| `InputState` + `reduce_input` | edges/input/snake.input.hpp | SDL boundary -> pmr::vector<SnakeCommand> |
| `ShatterParticleSoA` | spatial_fx/spatial_fx.contract.hpp | 4-vector SoA: pos, vel, color, life |
| `PipelineExecutionPlan` | spatial_fx/spatial_fx.contract.hpp | triangles + view/proj/vp matrices (canonical) |
| `plan_snake_scene` | spatial_fx/snake.plan.hpp | (snap, commands, difficulty, level, particles, canvas_w, canvas_h) -> plan |
