# Hello 3D Snake Demo — Implementation Plan (REVISED)

> **STATUS:** This plan was previously inaccurate. The on-disk tree does NOT match the architecture described below originally. Read `STATUS.md` FIRST for the authoritative, file-by-file inventory and every known inconsistency. This document now describes the intended build path and current reality.
>
> **Do not trust the original "Status" section** — it claimed all headers were in place; they are not (see §5).

## Goals

Build a semi-3D software-renderer snake game using the `shs_renderer.hpp` pipeline. Scene: a retro arcade arena where a low-poly snake glides across a checkerboard grid, food spawns on cells, eating grows the tail, and an orbiting camera views from above. Rendering mirrors the canonical tiled rasterization path used by every other demo (see §3).

## Domain Architecture (VOP / DOD pods)

The project follows the same pod layout as `tetris`. Each **pod** owns a responsibility; pure functions take spans/inputs and return values — side effects stay at execution boundaries (main loop + SDL/audio).

```
snake/
├── CMakeLists.txt                 # Hello3DSnake target; MUST list every source file
├── IMPLEMENTATION_PLAN.md         # this file (intended build path)
├── STATUS.md                      # ✅ CANONICAL — full inventory + known bugs. Read first.
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
        ├── input/snake.input.hpp  # InputState + reduce_input(InputState&, arena) -> commands (SDL boundary)
        └── audio/snake.audio.hpp  # play_sfx(type, state) — SDL audio boundary
```

**Key VOP/DOD rules:** pure reducers/actions/plan functions never touch SDL or game state; `std::pmr` frame arena (`vop::FrameMemoryResource`) for transient per-frame allocations; SoA layout for particle data; canonical renderer API only (never invent new rasterization types — see §3).

## Key Design Decisions

- **Domain folder = responsibility:** `domains/matrix/*` holds game logic; `domains/spatial_fx/*` holds FX/rendering setup.
- **Replay-ready difficulty table** in `config/difficulty.hpp` lets you swap levels mid-run without recompiling, so the reducer stays pure and testable.
- **Relative includes resolve via CMake include paths:** `include_directories("${CMAKE_CURRENT_LIST_DIR}/../")` adds the parent of CMakeLists.txt to the search path, so `"snake.contract.hpp"` from any file under `domains/` resolves to `../matrix/snake.contract.hpp`.

## Intent & Rationale (why it's structured this way)

The snake demo is a *learning exercise* in the SHS Renderer pipeline, not a shipped game. Key "why"s:
- **Orbiting top-down arena view** — keeps the scene readable while still exercising 3D batching (boxes for body segments, sphere for food). Deliberately simple so the rendering path stays tractable.
- **SoA particle data (`ShatterParticleSoA`)** — cache-friendly updates for shatter FX bursts; mirrors tetris's `ParticleTableSoA`.
- **Replay-ready difficulty table** in `config/difficulty.hpp` — lets you swap levels mid-run without recompiling, so the reducer stays pure and testable.
- **Relative includes via CMake (`../`)** — one include path covers all pods; `"snake.contract.hpp"` from any file under `domains/` resolves to `../matrix/snake.contract.hpp`.

## Build & Run

```bash
cd cpp-folders/build && cmake .. -DCMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake \
    -DHello3DSnake_ENABLED=ON \
    && cmake --build . --target Hello3DSnake
./cpp-folders/build/src/hello-3d-demos/snake/Hello3DSnake
```

## Current State — MUST FIX BEFORE BUILD (§5)

**Verified 2026-08-21:** The snake **edges** now compile cleanly in isolation (both edge headers syntax-check pass with their pod contracts on the include path). The full CMake build is blocked by two separate issues:

1. **VulkanMemoryAllocator (environmental blocker):** baseline `shs_renderer` dependency not installable on this WSL+GCC setup — vcpkg here is Windows-configured and the port does not exist. Blocks ALL demos equally, unrelated to snake edges.
2. **Broad domain coupling:** full main entry (`hello_3d_snake.cpp`) + spatial_fx contract/plan have cross-subdir include issues and missing symbols (`reduce_snake`, `ProcessedTriangle`, etc.) beyond the edges scope.

Known blockers for a full build (full detail in `STATUS.md` §6):

1. **Link failure:** `hello_3d_snake.cpp` includes `progression/snake.contract.hpp` and `edges/input/snake.input.hpp`, but CMakeLists.txt does not list them → undefined symbols (`ScoreState`, `InputState`, `reduce_input`).
2. **Conflicting `PipelineExecutionPlan`:** defined three ways across the tree (`.triangles`/`ProcessedTriangle` in matrix, `.faces` in spatial_fx contract). Only ONE canonical definition should exist — matching main's call signature and iteration (`plan.triangles`).
3. **Missing enum member:** `SnakeCommandType::NONE` referenced but not defined.
4. **Dead/stray code:** `domains/environment/snake.contract.hpp` unused; `config/difficulty.hpp` duplicates `domains/config/difficulty.hpp`; a stray tetris-style plan references non-existent `world.score.length`.
5. **Missing assets:** audio references `assets/snake/sfx_*.wav` (do not exist).

## Reference Demos

Three sibling demos under `cpp-folders/src/hello-3d-demos/`. They are **not** interchangeable references — pick each for a specific purpose:

| Demo | What it is | Use it for |
|------|-----------|------------|
| **tetris** (`tetris/`) | Full pod architecture (domains + edges), pure reducers/actions, `NOTES.md`, `TODOS.md`. The cleanest, most complete demo. | Mirror this for **structure**: contract-pod shape, particle SoA (`ParticleTableSoA`), plan signature (`plan_tetris_scene(world, particles, w, h, shake, arena)` → `PipelineExecutionPlan{triangles}`), and the main-loop pattern (SDL2 window + procedural audio synth callback + pmr frame arena + tiled parallel rasterization). |
| **fps** (`fps/`) | Minimal baseline: a single `hello_fps_demo.cpp` that includes only `"shs_renderer.hpp"` directly, no pods. | Reference for **raw renderer-API usage**: how to create the window/renderer, iterate triangles through `clip_triangle_to_frustum`, and rasterize via `rasterize_triangle_tile`. Not a pod-architecture reference. |
| **plane** (`plane/`) | Minimal baseline: single `hello_plane_demo.cpp` including only `"shs_renderer.hpp"`, uses quaternions for orientation. | Reference for the **simplest possible scene**: a flat oriented plane, showing the absolute minimum to drive the renderer. Not a pod-architecture reference. |

**Rule of thumb:** copy *structure* from tetris; copy *renderer-API mechanics* (window/rasterization) from fps/plane. Never mix them up — an agent mirroring "tetris" for structure should not expect fps-style single-file layout to be the target shape.
