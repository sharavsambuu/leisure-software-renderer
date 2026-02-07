# SHS Renderer Lib Migration: Port Plan + Coverage Checklist

This document keeps two things together:

1. The concrete port roadmap from `hello_pbr_light_shafts.cpp` to `hello_pass_plumbing.cpp`.
2. A coverage checklist showing what `hello_pbr_light_shafts.cpp` exercises from `shs_renderer.hpp`.

---

## A) Port Roadmap (Execution Order)

1. Lock a visual baseline from `hello_pbr_light_shafts.cpp`.
- Capture 3 fixed camera screenshots + FPS text for regression comparison.

2. Stabilize lib contracts before larger moves.
- Keep a single `PassContext` definition.
- Keep `FrameParams`, `Scene`, and pass `Inputs` as the canonical public contracts.

3. Add a temporary legacy bridge.
- Map legacy runtime structures to:
  - `shs::Scene`
  - `shs::FrameParams`
  - `RTRegistry` handles
- Goal: extract code pass-by-pass, avoid big-bang rewrite.

4. Port passes in render order.
- `PassShadowMap`
- `PassPBRForward`
- `PassTonemap`
- `PassLightShafts`
- Prefer extraction of existing logic over rewriting from scratch.

5. Extract IBL as a shared resource module.
- Move `CubeMapLinear`, `PrefilteredSpecular`, `EnvIBL`, and precompute helpers into lib resources.

6. Make `hello_pass_plumbing.cpp` a true pipeline demo.
- Replace current stub with full setup:
  - scene + frame params
  - RT creation/registration
  - `PipelineLightShafts::render(...)`

7. Enforce parity gates before cleanup.
- Visual parity at fixed views.
- Feature parity (PCSS + PBR + shafts).
- Build parity (`hello-plumbing` links only `shs::renderer`).

8. After parity, split to real static/shared library.
- Convert INTERFACE-only target to compiled lib target and migrate heavier code to `.cpp`.

---

## B) Coverage Checklist (`shs_renderer.hpp` vs SOTA demo)

Method:
- Source of symbols: class/struct declarations in `src/shs-renderer/shs_renderer.hpp`
- `Used in SOTA`: symbol appears in `src/hello-render-target/hello_pbr_light_shafts.cpp`
- `Used in other demos`: symbol appears in repo `src/**/*.cpp` but not in the SOTA demo

### B1. Used by SOTA demo (`hello_pbr_light_shafts.cpp`)

- `AbstractObject3D`
- `AbstractSceneState`
- `AbstractSky`
- `AbstractSystem`
- `AnalyticSky`
- `Buffer`
- `Camera3D`
- `Canvas`
- `Color`
- `CommandProcessor`
- `CubeMap`
- `CubeMapSky`
- `ModelGeometry`
- `MotionBuffer`
- `MoveBackwardCommand`
- `MoveForwardCommand`
- `MoveLeftCommand`
- `MoveRightCommand`
- `ShadowMap`
- `Texture2D`
- `ThreadedPriorityJobSystem`
- `Viewer`
- `WaitGroup`
- `ZBuffer`

Interpretation:
- These are strongly represented by the SOTA demo and are good primary extraction targets.

### B2. Not used in SOTA, but used elsewhere in demos

- `Command`
- `RT_ColorDepthVelocity`
- `RT_ColorDepth`
- `Varyings`

Interpretation:
- If you migrate only from SOTA, these can be under-covered and should stay in compatibility scope until another demo is ported.

### B3. Currently not used in demo `.cpp` files (dead or header-internal candidates)

- `JobEntry`
- `Obj3DFile`
- `ProceduralSky`
- `RT_Color`
- `RawTriangle`
- `ThreadSafePriorityQueue`

Interpretation:
- These are safe to postpone, isolate, or deprecate after verification.

---

## C) Recommendation

`hello_pbr_light_shafts.cpp` is sufficient to drive migration of the modern render core, but not sufficient to guarantee full coverage of every `shs_renderer.hpp` core symbol alone.

Use this policy:

1. Migrate SOTA-covered symbols first (Section B1).
2. Keep B2 behind compatibility layer until a second representative demo is ported.
3. Treat B3 as optional/cleanup unless a real consumer appears.

