# HelloFPSDemo — STATUS

**Status:** COMPLETE (build + headless verification passed)

A first-person arena shooter demonstrating the VOP/DOD domain-pod architecture
on top of the shared software renderer (`shs_renderer.hpp`). Pure simulation,
pure render planning, SDL confined to edges.

## Verification (headless, `SDL_VIDEODRIVER=dummy`)

| Check | Result |
|---|---|
| Build (gcc, `-Wall -Wextra -Wconversion -Wsign-conversion ...`) | 0 errors, 0 warnings in fps files |
| Determinism (`--screenshot` twice, byte compare) | PASS — byte-identical |
| Fire behavioral delta (`--autodrive-fire` vs idle) | PASS — 8.18% pixels differ (tracer/muzzle/hitmarker) |
| Scene richness (distinct color buckets) | PASS — 40 buckets |
| Signature colors (bots, gun, HUD) | PASS — all 7 signatures found |

Reproduce:

```bash
cd cpp-folders/build_vcpkg/src/hello-3d-demos/fps
SDL_VIDEODRIVER=dummy ./HelloFPSDemo --screenshot /tmp/fps_idle.bmp --frame=60
SDL_VIDEODRIVER=dummy ./HelloFPSDemo --autodrive-fire --screenshot /tmp/fps_fire.bmp --frame=33
python3 cpp-folders/_diag_fps/check_frames.py
python3 cpp-folders/_diag_fps/check_signatures.py
```

## Pod map

```
config/        difficulty.hpp, levels/fps_level_01.hpp      (tuning + level data)
domains/
  matrix/      contract/action/event/reducer                (pure sim core)
  spatial_fx/  contract/meshes/plan                         (mesh vocab + render plan)
  progression/ contract/reducer                             (event-driven score)
edges/
  input/       SDL poll -> UserCommands
  audio/       CombatEvents -> procedural synth (SPSC ring)
  rasterizer/  tiled multithreaded barycentric + Z-buffer
  ui/          HUD, health bars, tracers (screen-space)
hello_fps_demo.cpp                                          (main edge wiring only)
```

## Remaining work / known limitations

- `FrameMemoryResource` is still per-demo; hoist into the shared renderer
  library alongside the other demos' copies.
- Bot meshes are static poses (no walk animation); bob is a vertical hover.
- Projectiles are unlit emissive boxes; no point-light contribution.
- Audio device open failure is silently tolerated (game runs mute).