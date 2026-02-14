# Light Culling Status and Known Issues

Last updated: 2026-02-15

## Current Integration Status

### Jolt-based light culling paths

- `shs-renderer-lib/include/shs/lighting/jolt_light_culling.hpp`
  - `cull_lights_tiled(...)`
  - `cull_lights_tiled_depth01_range(...)`
  - `cull_lights_tiled_view_depth_range(...)`
  - `cull_lights_clustered(...)`
- `shs-renderer-lib/include/shs/pipeline/pass_adapters.hpp`
  - local lights are converted to `SceneShape` via `append_local_light_shapes_from_set(...)`
  - per-tile classification runs through `classify_vs_cell(...)` on Jolt-backed `SceneShape`

### Non-Jolt light culling paths

- `exp-plumbing/hello_forward_plus_stress_vulkan.cpp` GPU compute path
  - packs lights into `CullingLightGPU` via `make_*_culling_light(...)`
  - uses screen-space + depth-range + sphere/proxy tests in compute shaders
  - does **not** classify Jolt `SceneShape` directly in compute

## Important Clarification

It is not accurate to say that **all** light culling in the repository is Jolt-based today.

- CPU/pluggable pipeline culling path: mostly Jolt-based.
- Vulkan Forward+/stress compute culling path: custom GPU culler (non-Jolt).

## Known Issue: Tiled and Tiled-Depth Culling

Status: **Open**

Affected combinations (confirmed):

- `TechniqueMode::ForwardPlus` with `LightCullingMode::Tiled`
- `TechniqueMode::TiledDeferred` with `LightCullingMode::TiledDepthRange`

Observed behavior:

- visible instability / incorrect light assignment in tiles
- clustered and other combinations look comparatively stable

Recent fixes already applied:

- depth semantics and naming cleanup (`depth01` vs linear view depth)
- exact LH_NO perspective conversion for CPU clustered slice depth mapping
- conservative near-plane fallback in GPU light screen projection

Even after these fixes, the issue persists and needs a deeper pass.

## Suggested Next Investigation (when resumed)

1. Add debug visualization of tile list occupancy and per-tile depth ranges in screen space.
2. Capture and compare CPU reference culling vs GPU tile output for the same camera/light frame.
3. Validate frag-space tile indexing against compute tile indexing under Y-flip and viewport setup.
4. Add a deterministic regression scene with fixed camera, fixed seeds, and golden tile-count snapshots.
