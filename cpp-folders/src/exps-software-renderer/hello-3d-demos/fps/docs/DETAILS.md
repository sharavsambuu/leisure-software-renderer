# HelloFPSDemo — DETAILS

Architecture deep-dive for the FPS demo. See `STATUS.md` for verification state.

## Frame data flow (one iteration of the main loop)

```
SDL events ──> input::poll_input ──> InputState
                                        │ reduce_input(diff, dt, arena)
                                        v
                              pmr::vector<UserCommand>
                                        │ matrix::reduce_world(prev_world, cmds, diff, level, dt)
                                        v
                              WorldStepResult { next_world, events }
                                 │               │
              progression::reduce_progression    │ audio map (main edge)
                        │                        v
                   ScoreState               synth.play(SoundType)  [lock-free ring]
                                                 │
        spatial_fx::plan_fps_scene(next_world, meshes, ...) <── PipelineExecutionPlan
                                                 │
        raster::execute_tile_raster_job × N tiles (thread pool, disjoint regions)
                                                 │
        ui::draw_tracers / draw_enemy_health_bars / draw_fps_hud
                                                 │
                                          SDL present / BMP save
```

Everything between the SDL boundaries is a pure function; the only mutable
cross-frame state is `WorldSnapshot`, `ScoreState`, and two UI timers.

## Domain pods

### domains/matrix — simulation core
- **contract**: `PlayerSnapshot` (AoS, single entity), `BotTableSoA` /
  `ProjectileTableSoA` (structure-of-arrays hot tables), `WorldSnapshot`
  owning PMR vectors + deterministic LCG `rng_state`.
- **action**: `UserCommand` variant (`MoveIntent`, `LookIntent`, `FireIntent`,
  `JumpIntent`, `ResetIntent`) reduced to one `PlayerCommandFrame` per frame.
- **event**: `CombatEvent` stream (`PLAYER_FIRED`, `BOT_HIT`, `BOT_KILLED`,
  `BOT_FIRED`, `PLAYER_DAMAGED`, `PLAYER_JUMPED`) — the only channel to audio
  and scoring.
- **reducer**: `reduce_world` = step_player → step_hitscan → step_bot_ai →
  step_projectiles → step_tracers. All randomness via the snapshot's LCG;
  no `rand()`, no globals.

### domains/spatial_fx — render vocabulary
- **contract**: `LowPolyTriangle` (world-space authoring primitive with
  per-triangle `depth_bias`), `MeshBuilder` (quad/box/cylinder), and the
  render-ready `PipelineExecutionPlan` (clip-space verts + lit color).
- **meshes**: startup-built content — checkered arena from level data, bot
  (normal + hit-flash variants), viewmodel gun, muzzle-flash star, bolt.
- **plan**: `plan_fps_scene` builds LH view/proj, transforms + lambert-lights
  every batch into `ProcessedTriangle`s inside the frame PMR arena.
  - View-matrix note: `glm::lookAtLH` is kept deliberately; with the base
    heading (+Z forward at yaw=0) the side vector matches screen-right, so the
    mirror quirk seen in snake's front-facing camera does not apply. Re-verify
    via autodrive screenshot if camera conventions change.

### domains/progression — scoring
- `ScoreState { score, kills }`; reducer consumes only `CombatEvent`s.
  Combat code never touches score directly (snake lesson applied).

## Edges

- **input**: sole owner of SDL event/keyboard/mouse APIs. Sensitivity and
  key-look rates come from `Difficulty`; output is pure `UserCommand`s.
- **audio**: procedural synthesis only (no binary assets) — a faithful port of
  the original demo's synth: phase-accumulator oscillators with exponential
  pitch sweeps, per-sound noise bursts (xorshift32), oldest-same-type voice
  stealing across 16 voices, sqrt(count) voice normalization, and soft
  clipping (`x / (1 + 0.8|x|)`). Main thread pushes `SoundType`s into a
  lock-free SPSC ring; the SDL callback drains and mixes. Device-open failure
  degrades to silence.
- **rasterizer**: screen split into 64×64 tiles; each tile job clips every
  triangle to the frustum, projects, then barycentric-rasterizes with a
  Z-buffer test restricted to its own rectangle — wait-free, no locks.
- **ui**: Bresenham lines, filled/bordered rects, 7-segment digits; world-
  anchored enemy health bars projected through the same VP matrix; tracers,
  crosshair/hitmarker, HP bar, score panel, damage vignette.

## Include & namespace convention

- ONE project root on the include path (`fps/` itself, via CMake).
  Cross-pod headers are included by **root-relative path in angle brackets**,
  so each include line states its location in the architecture:
  `#include <domains/matrix/fps.contract.hpp>`,
  `#include <config/difficulty.hpp>`, `#include <edges/ui/fps.hud.hpp>`.
  No `../../` relative include chains.
- Angle brackets = cross-pod / rooted path (architecture-visible dependency);
  quote includes = same-directory siblings only (`"fps.meshes.hpp"`).
- Namespace identity lives in the code (`fps::matrix`, `fps::spatial_fx`,
  `fps::progression`, `fps::input/audio/raster/ui`); include-path identity
  lives in CMake. The two are independent on purpose.

## Memory discipline

- One 8 MB linear PMR arena per frame (`FrameMemoryResource` in main),
  O(1) reset. All per-frame allocations (commands, events, plan triangles)
  come from it. Long-lived world tables use the default resource.
- No allocation happens inside the tile jobs or HUD drawing.

## Determinism & headless testing

- Fixed `dt = 1/60` in screenshot mode; single LCG seeded from config.
- CLI: `--screenshot <path.bmp>` (implies headless, no window/audio),
  `--frame=N` (default 60), `--autodrive-fire` (injects FireIntent at frame 30).
- Diagnostics live in `cpp-folders/_diag_fps/` (frame diff + signature checks).

## Controls

WASD move · mouse look (relative mode) · F/LMB/Ctrl fire · Space jump ·
R reset · Esc quit. Arrow keys also steer look for keyboard-only play.