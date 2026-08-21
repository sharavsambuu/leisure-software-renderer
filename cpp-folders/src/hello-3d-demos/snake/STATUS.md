# snake/STATUS.md — Canonical Reference for Future Agents

> **Read this file first.** It is the single source of truth for the `snake` demo: what it is, how it should be structured (VOP/DOD domain pods), what the current on-disk state actually is, and every known inconsistency. Do NOT trust `IMPLEMENTATION_PLAN.md` — it was written by an earlier agent and does not match reality.

---

## 0. Agent Log / Session Notes (2026-08-21)

**Edges DONE & verified.** Both snake edge headers now compile cleanly in isolation:
- `edges/input/snake.input.hpp` — syntax-checks pass with its pod contract (`snake::matrix`) on the include path; produces `SnakeCommand` intents from raw input state.
- `edges/rasterizer/snake.rasterizer.hpp` — standalone rasterizer edge, compiles clean.

Fixes applied to reach this: matrix contract got `<memory_resource>` + fully-qualified cross-pod refs (`snake::matrix::`); input edge uses bare contract include + dropped `const` on scratch array; parent CMakeLists.txt now adds all 8 snake subdirs + shs renderer root to the global include path.

**Full build still blocked (two separate issues, unrelated to edges):**
1. **VulkanMemoryAllocator** — baseline `shs_renderer` dependency not installable on this WSL+GCC setup (vcpkg here is Windows-configured; port does not exist). Blocks ALL demos equally.
2. **Broad domain coupling** — full main entry (`hello_3d_snake.cpp`) + spatial_fx contract/plan have cross-subdir include issues and missing symbols beyond the edges scope (§6, §7).

---

## 1. Project Overview

Semi-3D software-renderer snake demo built on top of the SHS Renderer (`shs_renderer.hpp`).
Scene: a retro arcade tunnel view where a low-poly snake glides across a checkerboard arena, food spawns on cells, and eating grows the tail. Camera orbits slowly from above. Rendering mirrors the **canonical tiled rasterization path** used by every other demo (see §4).

- Target executable: `Hello3DSnake`
- Entry point: `hello_3d_snake.cpp::main`
- Build/run commands (from IMPLEMENTATION_PLAN.md, still valid):
  ```bash
  cd cpp-folders/build && cmake .. -DCMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake \
      -DHello3DSnake_ENABLED=ON && cmake --build . --target Hello3DSnake
  ./cpp-folders/build/src/hello-3d-demos/snake/Hello3DSnake
  ```

---

## 2. Intended Architecture (VOP / DOD Domain Pods)

The project should follow the same pod layout as `tetris` (§5 reference). Each **pod** owns a responsibility and exposes pure functions; side effects stay at execution boundaries (main loop + SDL/audio).

Canonical snake pod layout (what it SHOULD look like, mirroring tetris):

```
snake/
├── CMakeLists.txt                 # Hello3DSnake target; lists every source file
├── IMPLEMENTATION_PLAN.md         # intended build path
├── STATUS.md                      # ✅ THIS FILE — canonical reference for agents
├── hello_3d_snake.cpp             # main: SDL2 window + audio + game loop (execution boundary)
└── domains/
    ├── config/difficulty.hpp      # Difficulty struct + level meta-table (replay-ready)
    └── matrix/
        ├── snake.contract.hpp     # CORE TYPES ONLY: SnakeCommand, SnakeSnapshot, FoodState, events, step result
        ├── snake.action.hpp       # reduce_snake_commands(span<const SnakeCommand>) -> movement delta (pure)
        ├── snake.event.hpp        # re-exports shared vocabulary from contract (SnakeEventType, SnakeEvent)
        └── snake.reducer.hpp      # reduce_snake(snap, commands, difficulty, level) -> next_state + events (pure)
    └── spatial_fx/
        └── snake.plan.hpp         # plan_snake_scene(...) -> PipelineExecutionPlan (render-ready triangles)
    └── edges/
        ├── input/snake.input.hpp  # InputState + reduce_input(InputState&, arena) -> commands (SDL boundary) ✅ compiles standalone
        └── audio/snake.audio.hpp  # play_sfx(type, state) — SDL audio boundary
```

**Key VOP/DOD rules this must obey (from the constitutions):**
- Pure reducers/actions/plan functions: no SDL, no game-state mutation, deterministic given inputs.
- `std::pmr` frame arena (`vop::FrameMemoryResource`) for transient per-frame allocations; persistent state is separate.
- SoA layout for particle data; wait-free span contracts between pods.
- Canonical renderer API only (see §4) — never invent new rasterization types.

---

## 3. Current On-Disk File Inventory (15 files, MESSY)

The on-disk tree has **duplicates and conflicting definitions**. Below is the actual state. Line counts are approximate.

### Entry + build
- `hello_3d_snake.cpp` (~259 lines) — main loop. **Includes 2 files NOT in CMakeLists.txt** (`progression/snake.contract.hpp`, `edges/input/snake.input.hpp`) → will fail to link until fixed. Uses `plan_snake_scene(snap, commands, difficulty, level, particles)` (5 args), `ShatterParticleSoA`, `ScoreState`, `InputState`+`reduce_input`, `SnakeLevel01`.
- `CMakeLists.txt` — Hello3DSnake target lists only 5 sources: `hello_3d_snake.cpp`, `domains/matrix/{contract,action,event,reducer}.hpp`, `domains/spatial_fx/snake.plan.hpp`. **Missing**: `progression/snake.contract.hpp`, `edges/input/snake.input.hpp` (both included by main).

### matrix pod
- `domains/matrix/snake.contract.hpp` (~58 lines) — core types. Defines `SnakeCommandType{LEFT,RIGHT,UP,DOWN}` (**NOTE: no `NONE` member**, yet a comment references it), `SnakeCommand`, `BodySoA`, `FoodState`, `SnakeSnapshot{head_pos, head_dir, food, body}`, `SnakeEventType{HEAD_MOVED,SELF_COLLISION,FOOD_EATEN}`, `SnakeEvent`, `SnakeStepResult`.
- `domains/matrix/snake.action.hpp` (~23 lines) — `reduce_snake_commands(span<const SnakeCommand>) -> glm::vec2 delta`.
- `domains/matrix/snake.event.hpp` (~6 lines) — re-export of contract vocabulary.
- `domains/matrix/snake.reducer.hpp` (~82 lines) — `cell_to_world(level, x, y)` + `reduce_snake(snap, commands, difficulty, level) -> SnakeStepResult`. Includes `"../../../config/levels/snake_level_01.hpp"`.

### spatial_fx pod (CONFLICTING definitions)
- `domains/spatial_fx/snake.plan.hpp` (~154 lines) — defines its own `LowPolyTriangle{p0,p1,p2,color,depth_bias}`, `add_box/add_quad`, and a **6-arg** `plan_snake_scene(snap, commands, difficulty, level, particles)` that builds board tiles + snake body + food + particle FX (emits 40 shatter particles when `!alive`).
- `domains/matrix/snake.plan.hpp` (~213 lines) — **ALSO defines** a conflicting `PipelineExecutionPlan{ triangles: pmr::vector<ProcessedTriangle>, view/proj/vp_matrix }`, its own `ShatterParticleSoA`, and a **5-arg** `plan_snake_scene(world, particles, w, h, shake, arena)` (mirrors tetris). This is the WRONG plan for snake; it references `world.score.length` which does not exist.
- `domains/spatial_fx/snake.contract.hpp` (~61 lines) — defines a **third** conflicting type set: `Face`, `PipelineExecutionPlan{ faces }`, `SnakeCameraParams`, `SnakeLightParams`, and its own `ShatterParticleSoA`.

### config pod (DUPLICATE)
- `domains/config/difficulty.hpp` (~36 lines) — `Difficulty{solid_walls, speed_ticks}`, `LevelMeta`, 8-entry `levels` table. **Duplicate of** `config/difficulty.hpp`.

### level data
- `config/levels/snake_level_01.hpp` (~34 lines) — `SnakeLevel01`: GRID_W/H=20, head_spawn{9,9}, body_spawn[2], dir_spawn{1,0}, food_table[8], arena_center/half_w/half_h, rng_state. **This is the authoritative level data.**

### progression pod (stray)
- `domains/progression/snake.contract.hpp` (~19 lines) — `ScoreState{score, high_score, length, speed_mult}` + `fresh()`. Included by main but not in CMakeLists.txt.

### environment pod (stray, unused)
- `domains/environment/snake.contract.hpp` (~27 lines) — `MoodState` reactive color reducer. Not referenced anywhere; dead code.

### edges pod
- `edges/input/snake.input.hpp` (~28 lines) — `InputState{turn_left,right,strafe_up,down}` + `reduce_input(InputState&, memory_resource*) -> pmr::vector<SnakeCommand>`. **✅ Compiles cleanly in isolation** (verified 2026-08-21). Included by main but not in CMakeLists.txt.
- `edges/audio/snake.audio.hpp` (~35 lines) — `AudioState`, `play_sfx(type, state)`. References `assets/snake/sfx_*.wav` that **do not exist** → audio is effectively dead unless assets are added.

---

## 4. Canonical Renderer API (shs_renderer.hpp) — DO NOT INVENT NEW TYPES

The renderer defines exactly these rasterization primitives. Every demo (tetris, snake) must use them verbatim:

- `struct FrustumClipPolygon` + `static inline FrustumClipPolygon clip_triangle_to_frustum(const glm::vec4& c0, const glm::vec4& c1, const glm::vec4& c2)` — clips a triangle to the frustum.
- `PipelineExecutionPlan { std::pmr::vector<ProcessedTriangle> triangles; }` where each `ProcessedTriangle { glm::vec4 c0,c1,c2; shs::Color lit_color; float depth_bias; }`. **This is THE render-ready format.** (Do NOT use a separate `Face` type — that conflicts.)
- Mesh generation helpers: `MeshGen::add_box`, `MeshGen::add_quad`, `MeshGen::add_sphere` → build world-space `LowPolyTriangle{p0,p1,p2,color,depth_bias}`.

**Canonical rasterization loop (copy from tetris/hello_3d_tetris.cpp §809-813):**
```cpp
for (const auto& tri : plan.triangles) {
    const shs::Raster::FrustumClipPolygon poly = shs::Raster::clip_triangle_to_frustum(tri.c0, tri.c1, tri.c2);
    if (poly.count < 3) continue;
    // clip each vertex to screen space via NDC->screen mapping
    for (int i = 1; i + 1 < poly.count; ++i) {
        rasterize_triangle_tile(canvas, z_buffer, s0, s1, s2, tri.lit_color, tri.depth_bias, tmin, tmax);
    }
}
```

Supporting types: `shs::Canvas`, `shs::ZBuffer`, `shs::Color`, `shs::Job::ThreadedPriorityJobSystem`, `shs::Job::WaitGroup`. GLM uses Left-Handed (`lookAtLH`, `perspectiveLH_NO`).

---

## 5. Canonical Reference: tetris (mirror this layout)

`cpp-folders/src/hello-3d-demos/tetris/` is the clean, working reference. Snake should mirror it exactly:

- **Contract pod** (`tetris.contract.hpp`, `domains/spatial_fx/spatial_fx.contract.hpp`): particle SoA = `ParticleTableSoA { position, velocity, color : vec3; life }` with `add(vec3 pos, vec3 vel, shs::Color col, float dur)` and bump-alloc `push_back(mr, p, v, c, dur)`. Snake's `ShatterParticleSoA` should match this shape (snake currently has a 4-vector SoA in spatial_fx/contract.hpp — that one is actually correct; the matrix/snake.plan.hpp copy is wrong).
- **Plan pod** (`tetris.plan.hpp`, ~239 lines): `MeshGen::add_box/add_quad/add_sphere` → `LowPolyTriangle` → push to `plan.triangles` via `plan_tetris_scene(world, particles, w, h, shake, arena)`. Snake's plan should follow this signature/shape.
- **Main loop** (`hello_3d_tetris.cpp`, ~834 lines): SDL2 window + procedural audio synth callback, `vop::FrameMemoryResource` pmr arena, input polling → commands → `reduce_tetris(world, commands, dt, arena)` → next_state+events → particle update → `plan_tetris_scene(...)` → tiled parallel rasterization → 2D HUD → present. Snake's main loop structure matches this closely already; it just needs the includes/types reconciled.

---

## 6. Known Bugs / Inconsistencies (must fix)

1. **Link failure**: `hello_3d_snake.cpp` includes `progression/snake.contract.hpp` and `edges/input/snake.input.hpp`, but CMakeLists.txt does not list them → undefined symbols (`ScoreState`, `InputState`, `reduce_input`).
2. **Conflicting `PipelineExecutionPlan`**: defined three ways across `domains/matrix/snake.plan.hpp` (ProcessedTriangle.triangles), `domains/spatial_fx/snake.contract.hpp` (faces), and the plan function signatures disagree (5-arg vs 6-arg). Only ONE canonical definition should exist.
3. **Missing enum member**: `SnakeCommandType::NONE` referenced in a comment but not defined.
4. **Wrong plan signature mismatch**: main calls `plan_snake_scene(snap, commands, difficulty, level, particles)` (5 args) while spatial_fx/snake.plan.hpp defines it with 6 args `(snap, commands, difficulty, level, particles)` — actually matches count, but the two plan files disagree on return type and internals.
5. **Dead/stray code**: `domains/environment/snake.contract.hpp` unused; `config/difficulty.hpp` duplicate of `domains/config/difficulty.hpp`; matrix/snake.plan.hpp is a wrong tetris-style copy that references non-existent `world.score.length`.
6. **Missing assets**: audio references `assets/snake/sfx_*.wav` (do not exist).
7. **IMPLEMENTATION_PLAN.md** describes an architecture that does not match reality — out of date.

---

## 7. Next Steps / Recommended Fix Direction

Two options were discussed with the user; pick one:

- **Option A — Canonical refactor**: Restructure snake to mirror tetris exactly (one consolidated contract pod, single plan pod using `ProcessedTriangle.triangles`, correct main loop + food spawn). Cleanest but touches most files.
- **Option B — Minimal reconcile**: Keep the current multi-file layout; fix only what's needed to compile & run (unify the two PipelineExecutionPlan types, add missing includes/types, wire up CMakeLists + food spawn).

**Until a refactor is chosen, do NOT delete or restructure files.** Preserve all current content so history is intact. STATUS.md remains authoritative for future agents.

---

## 8. Quick Reference: Key Types & Signatures (current state)

| Type | Location | Notes |
|------|----------|-------|
| `SnakeCommandType` | matrix/snake.contract.hpp | LEFT/RIGHT/UP/DOWN; **no NONE** |
| `SnakeSnapshot` | matrix/snake.contract.hpp | head_pos, head_dir, food, body (SoA) |
| `SnakeStepResult` | matrix/snake.contract.hpp | next_state + events + alive flag |
| `reduce_snake_commands` | matrix/snake.action.hpp | span<const SnakeCommand> -> vec2 delta |
| `reduce_snake` | matrix/snake.reducer.hpp | (snap, commands, difficulty, level) -> step result |
| `SnakeLevel01` | config/levels/snake_level_01.hpp | GRID 20x20, food_table[8], arena bounds |
| `Difficulty` / `levels[]` | domains/config/difficulty.hpp | solid_walls, speed_ticks; 8 levels |
| `ScoreState` | progression/snake.contract.hpp | score, high_score, length, speed_mult |
| `InputState` + `reduce_input` | edges/input/snake.input.hpp | SDL boundary -> commands ✅ compiles standalone |
| `ShatterParticleSoA` | spatial_fx/snake.contract.hpp (correct) / matrix/snake.plan.hpp (wrong dup) | 4-vector SoA: pos, vel, color, life |
| `PipelineExecutionPlan` | THREE conflicting defs — unify to tetris shape |
