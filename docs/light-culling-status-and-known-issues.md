# Light Culling Status and Notes

Last updated: 2026-02-16

## 1) Scope

This document summarizes the status of culling integration based on Jolt shape/volume primitives.

## 2) Current Status (Completed)

The core Jolt shape/volume-based culling functionality is considered complete for the current renderer phase.

Completed baseline set:

- `SceneShape` representation backed by `jolt_shapes`
- View frustum culling flow
- Shadow frustum culling flow
- View occlusion culling (runtime toggle)
- Shadow occlusion culling (runtime toggle, default OFF)
- Culling statistics and debug AABB overlay validation flow

## 3) Important Clarification

Not every light-culling path in the repository is strictly Jolt-based.

- CPU/pluggable paths: mostly Jolt shape/volume based
- Some Vulkan compute stress/demo paths: custom GPU culling logic

This is an intentional architecture choice, not necessarily a conflict.

## 4) Stable Default Configuration

Current recommended stable defaults:

- View occlusion: ON
- Shadow occlusion: OFF
- Shadow frustum culling: ON

This setup reduces visual popping and query-related instability.

## 5) Archived Issue Notes

Earlier observations around tiled/tiled-depth/clustered instability are now partly resolved and should be tracked with clearer status:

- Resolved:
  - rectangular block/patch lighting artifacts in GPU culling modes
  - root cause: mismatch between light influence evaluation and culling bounds in some local light types
  - fixes landed:
    - stricter shape-aware influence checks in scene lighting evaluation
    - conservative cull-bound construction for anisotropic light volumes
    - conservative compute culling fallback bounds in stress path
- Remaining:
  - continue monitoring visual stability/perf tradeoffs when adding new light types or new culling kernels

Practical status:

- current issue is considered fixed for active demos
- keep regression checks when touching light packing or culling shaders

## 6) Connection To The Next Phase

Next major goal:

- dynamic compositional and configurable render paths

Plan document:

- `docs/dynamic-render-path-composition-plan.md`

## 7) Reusable Light Runtime and Shader Contract Notes

Recent work added reusable light runtime abstractions and shader-level reuse that are worth preserving:

- shared C++ light runtime abstractions:
  - `cpp-folders/src/shs-renderer-lib/include/shs/lighting/light_runtime.hpp`
- shared Vulkan shader common blocks:
  - `cpp-folders/src/shs-renderer-lib/shaders/vulkan/common/*.glsl`

Practical effect:

- less duplicated light-type logic across SW/VK demos
- lower risk of contract drift between CPU light packing and GPU light evaluation

## 8) Next Phase Handoff

With light-volume and culling stability in place, the next phase is render-path composition:

- introduce shared render-path interfaces/contracts
- keep current Forward/Forward+ behavior as baseline path
- add an additional path (Deferred) on top of same light/culling inputs
- support runtime path switching with comparable stats

See:

- `docs/dynamic-render-path-composition-plan.md`
