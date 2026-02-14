# SHS Lighting Space and Jolt Space Notes

Last updated: 2026-02-15

## Goal

Keep all lighting and shadow computations in SHS world space (LH, +Z forward), while still using Jolt shapes/volumes for culling and collision.

This avoids direction/sign drift between:

- viewer direction
- light direction
- shadow camera direction
- shadow sampling projection

## Space Ownership Rules

1. SHS space owns all render-space math.
- world/view/projection matrices used for shading
- camera vectors
- light vectors
- BRDF inputs (`N`, `L`, `V`, `H`)
- shadow map sampling coordinates

2. Jolt space is a geometry backend only.
- shape construction and storage
- broad-phase/narrow-phase culling and volume tests
- physics/collision representations

3. Conversion boundary is explicit.
- convert SHS -> Jolt only at adapter APIs (`shs::jolt::to_jph(...)`)
- convert Jolt -> SHS only at adapter APIs (`shs::jolt::to_glm(...)`)
- do not run shading math on `JPH::Vec3`/`JPH::Mat44`

## Direction Semantics (Canonical)

Use this canonical name and meaning everywhere:

- `sun_dir_to_scene_ws`: normalized direction from the sun/light toward the scene (world space, SHS)

Derived vectors:

- shading incoming light direction: `L = normalize(-sun_dir_to_scene_ws)`
- view direction: `V = normalize(camera_pos_ws - world_pos_ws)`
- half vector: `H = normalize(L + V)`

Do not mix these meanings:

- direction from scene to sun
- direction from sun to scene
- light forward axis in local model space

If the semantic is unclear, rename before debugging.

## Shadow Camera and Shadow Sampling

Directional light camera:

- `build_dir_light_camera_aabb(...)` input direction must match `sun_dir_to_scene_ws`
- shadow caster bounds for light setup should come from SHS render geometry bounds
- avoid building light-space bounds from Jolt bounds in lighting code paths

Projection convention:

- SHS frustum/culling extraction uses LH NO (`z in [-1, 1]`)
- software shadow sampling converts NO depth to `[0, 1]` in `shadow_sample.hpp`
- Vulkan shadow sampling expects `[0, 1]`; apply NO->ZO mapping matrix before sampling

Keep this split explicit and documented at the pass boundary.

## Known Pitfalls

1. Using Jolt world AABB for light/shadow setup can skew visual-light alignment.
2. Mixing `sun_dir_to_scene_ws` with scene-to-sun vectors flips diffuse/specular response.
3. Forgetting NO->ZO conversion in Vulkan shadow sampling causes depth mismatch.
4. Mixing normal reconstruction conventions (winding/cross order) flips lit faces.
5. Using non-normalized direction vectors causes unstable specular and bias behavior.

## Practical Checklist (Before/After Lighting Changes)

1. Confirm all shading vectors are SHS-space `glm` vectors.
2. Confirm `sun_dir_to_scene_ws` is the only directional-light world vector in shader UBOs.
3. Confirm `L = -sun_dir_to_scene_ws` in both software and Vulkan shading paths.
4. Confirm directional shadow camera receives the same `sun_dir_to_scene_ws`.
5. Confirm shadow caster bounds are derived from render mesh bounds in SHS space.
6. Confirm Jolt conversion functions are only used at culling/physics boundaries.
7. Confirm software and Vulkan demos show matching light-facing surfaces for the same camera/light pose.

## Recommended Naming Standard

Prefer these names:

- `sun_dir_to_scene_ws`
- `camera_pos_ws`
- `world_pos_ws`
- `light_view_proj_no` (if NO)
- `light_view_proj_zo` (if ZO)

Avoid ambiguous names:

- `light_dir`
- `sun_dir`
- `shadow_dir`

## Quick Debug Procedure

When lighting looks wrong:

1. Freeze camera and sun animation.
2. Print/inspect one object's `N`, `L`, `dot(N,L)`, and shadow visibility.
3. Verify the same object is classified visible by both frustum and shadow passes.
4. Toggle shadow off; verify diffuse/specular alone look correct.
5. Toggle shadow on; verify only visibility term changes.
6. Compare software and Vulkan at the same frame pose.

If software is correct but Vulkan is wrong, check projection convention and shadow sampling matrix first.
