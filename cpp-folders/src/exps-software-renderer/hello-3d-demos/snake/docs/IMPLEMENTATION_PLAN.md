# snake/docs/IMPLEMENTATION_PLAN.md

> **HISTORICAL — superseded by `docs/STATUS.md`.** The stray-file cleanup this plan requests
> (item 3 below) was COMPLETED on 2026-08-22; see STATUS.md §0 "DOCS RELOCATED + DUPLICATION PURGE".

> **Status note (2026-08-22):** this file was rewritten to match reality. The original plan described an
> architecture that never existed on disk — for the authoritative current state read `STATUS.md` instead.
> This file now records only: what is DONE, and the remaining roadmap.

## DONE (verified: 0 errors / 0 snake-file warnings via build_vcpkg)

- [x] Multi-pod VOP/DOD layout mirroring tetris: matrix (state machine), spatial_fx (render planning),
      progression (scoring), edges/input (SDL boundary) — all pure except main.
- [x] Canonical renderer path: world-space boxes → vp transform → pre-shaded `ProcessedTriangle` list
      (`PipelineExecutionPlan`) → tiled parallel rasterization with frustum clip + barycentric depth test.
- [x] Gameplay core: grid movement, reverse rejection, soft/solid walls, self-collision (tail vacates),
      deterministic food table cycling, grow-at-head on eat (+10 score event).
- [x] Progression pod wired: event-driven scoring + speed ramp computation + high-score capture on death.
- [x] Death FX: one-shot shatter burst at head on alive→dead transition (main edge, deterministic LCG).
- [x] Contract naming fixed (`spatial_fx.contract.hpp`) so every bare include resolves uniquely.
- [x] CMake include dirs pruned to exactly what sources reference (`config`, `config/levels`,
      `domains/matrix`); shared renderer inherited from parent aggregator.

## REMAINING (in priority order)

1. **Game-over reset/restart** — add an R-restart command path: input edge emits a RESTART intent,
   reducer re-initializes from `SnakeLevel01` spawn data, progression resets via `ScoreState::fresh()`.
2. **Time-based stepping** — replace per-frame stepping with an accumulator driven by
   `score_state.speed_mult` so the speed ramp actually affects tick rate.
3. ~~**Stray-file cleanup**~~ — DONE 2026-08-22 (all five deleted; see STATUS.md §0).
4. **Audio** — either add real `assets/snake/sfx_*.wav` + wire `edges/audio/snake.audio.hpp`, or mirror
   tetris's procedural synth callback (preferred; no binary assets needed).
5. **HUD** — draw score/high-score/length overlay in main after rasterization (tetris has a 2D HUD pass
   to copy).
6. **Optional polish** — food pulse animation, camera shake on death, difficulty toggle key
   (soft ↔ solid walls).

## Build & Run

```bash
wsl.exe -d Ubuntu-24.04 --cd /home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/build_vcpkg -- bash -c "cmake . && cmake --build . --target Hello3DSnake -j$(nproc)"
./cpp-folders/build_vcpkg/src/hello-3d-demos/snake/Hello3DSnake
```

See `docs/dev/cpp_compilation_workflow.md` for the full agent workflow, shell-quoting traps, and the
C++ pitfalls list (GLM double-brace init, contract shadowing, sibling-namespace lookup, strict warnings).