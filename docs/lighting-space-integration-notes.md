# SHS Lighting Space and Jolt Space Integration Notes

Last updated: 2026-02-16

## 1) Goal

Keep all lighting and shadow calculations in SHS world space (LH, +Z forward),
while using Jolt as a geometry/collision/culling backend.

This minimizes semantic drift in:

- camera/view direction
- light direction
- shadow camera direction
- shadow sampling projection

## 2) Space Ownership Rules

1. SHS space owns render-space math.
- world/view/projection used by shading
- camera and light vectors
- BRDF inputs (`N`, `L`, `V`, `H`)
- shadow sampling coordinates

2. Jolt space is geometry backend.
- shape construction and storage
- broad-phase/narrow-phase culling
- physics/collision representation

3. Conversion boundaries must stay explicit.
- SHS -> Jolt only via adapter APIs (`shs::jolt::to_jph(...)`)
- Jolt -> SHS only via adapter APIs (`shs::jolt::to_glm(...)`)
- avoid direct shading math on `JPH::Vec3`/`JPH::Mat44`

4. Unit boundaries must stay explicit.
- SHS and Jolt both use SI-style runtime units (`meter`, `kilogram`, `second`, `radian`)
- unit scale is `1:1` across the adapter boundary
- avoid per-demo ad-hoc unit scaling in Jolt integration code

## 3) Canonical Direction Semantics

Standard name:

- `sun_dir_to_scene_ws`

Meaning:

- normalized world-space direction from the sun/light toward the scene

Derived vectors:

- incoming light: `L = normalize(-sun_dir_to_scene_ws)`
- view direction: `V = normalize(camera_pos_ws - world_pos_ws)`
- half vector: `H = normalize(L + V)`

Do not mix these meanings:

- scene -> sun
- sun -> scene
- model-local forward axis

## 4) Shadow Camera and Shadow Sampling Conventions

Directional shadow camera:

- `build_dir_light_camera_aabb(...)` should receive a direction consistent with `sun_dir_to_scene_ws`
- light setup bounds should be derived from SHS render geometry bounds

Projection conventions:

- SHS frustum/culling extraction: LH NO (`z in [-1, 1]`)
- software shadow sampling: convert NO depth to `[0, 1]`
- Vulkan shadow sampling: expects `[0, 1]`, so apply NO->ZO mapping explicitly at pass boundary

## 5) Common Pitfalls

1. Using Jolt world AABB directly for lighting setup
2. Mixing `sun_dir_to_scene_ws` with scene->sun semantics
3. Missing NO->ZO conversion in Vulkan shadow path
4. Winding/cross-order mismatch that flips normals
5. Non-normalized direction vectors causing unstable specular/bias behavior

## 6) Practical Checklist For Lighting Changes

1. Confirm shading vectors are all SHS `glm` vectors.
2. Confirm directional light world vector semantic is `sun_dir_to_scene_ws`.
3. Confirm `L = -sun_dir_to_scene_ws` is applied consistently.
4. Confirm shadow camera uses the same directional semantic.
5. Confirm shadow caster bounds come from SHS render bounds.
6. Confirm Jolt adapter conversion is only used at the boundary.
7. Confirm SW and VK demos match on light-facing behavior at identical pose.

## 7) Naming Standard

Preferred names:

- `sun_dir_to_scene_ws`
- `camera_pos_ws`
- `world_pos_ws`
- `light_view_proj_no` (for NO)
- `light_view_proj_zo` (for ZO)

Avoid ambiguous names:

- `light_dir`
- `sun_dir`
- `shadow_dir`

## 8) Quick Debug Procedure

When lighting looks wrong:

1. Freeze camera and sun animation.
2. Inspect one object’s `N`, `L`, `dot(N,L)`, and shadow visibility.
3. Verify the object is visible in both view and shadow culling passes.
4. Disable shadow and validate diffuse/specular baseline.
5. Re-enable shadow and verify only visibility term changes.
6. Compare SW and VK at the same frame pose.

If SW is correct but VK is wrong, check projection convention and shadow sampling matrix first.
