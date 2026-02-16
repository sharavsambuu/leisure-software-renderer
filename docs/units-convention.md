# SHS and Jolt Units Convention

Last updated: 2026-02-16

## 1) Canonical Runtime Units

SHS now uses SI-style runtime units aligned with Jolt integration.

- Distance: `meter`
- Mass: `kilogram`
- Time: `second`
- Angle: `radian`

Key interpretation:

- `1.0` world unit = `1 meter`
- velocities are `m/s`
- accelerations are `m/s^2`
- angular velocities are `rad/s`
- forces are `N` (`kg*m/s^2`)
- torques are `N*m`

## 2) Gravity Convention

SHS uses `+Y` as up, so gravity points downward on Y.

- standard gravity magnitude: `9.81 m/s^2`
- world gravity vector: `(0, -9.81, 0)`

## 3) Jolt Mapping

Jolt and SHS are mapped with 1:1 unit scale.

- distance scale SHS -> Jolt: `1.0`
- distance scale Jolt -> SHS: `1.0`

This means there is no hidden distance scaling at the adapter boundary.
Coordinate conversion still performs handedness conversion (Z-flip), but not unit scaling.

## 4) Code Anchors

Canonical constants and helpers live in:

- `cpp-folders/src/shs-renderer-lib/include/shs/core/units.hpp`

Jolt adapter unit bridge lives in:

- `cpp-folders/src/shs-renderer-lib/include/shs/geometry/jolt_adapter.hpp`

Minimum shape-size clamps use canonical unit constants in:

- `cpp-folders/src/shs-renderer-lib/include/shs/geometry/jolt_shapes.hpp`

## 5) Practical Guidelines for New Code

1. Treat all scene positions/sizes as meters.
2. Keep time deltas (`dt`) in seconds.
3. Keep camera move speeds in `m/s`.
4. Use radians for all API-level angle values.
5. Avoid magic numeric scales in Jolt boundary code; use unit helpers/constants.

## 6) Human-Scale Demo Baseline

To keep demos readable and realistic by default:

- camera start height is around `1.7m` to `5.5m` depending on scene coverage needs
- walk speed defaults around `6-8 m/s` (with boost for fast inspection)
- floor extents target roughly `48m` to `64m` total width
- local light ranges generally stay in `3m` to `8m`
- local light motion heights generally stay in `1.5m` to `5m`

Updated demos following this baseline:

- `cpp-folders/src/exp-plumbing/hello_light_types_culling_sw.cpp`
- `cpp-folders/src/exp-plumbing/hello_light_types_culling_vk.cpp`
- `cpp-folders/src/exp-plumbing/hello_rendering_paths.cpp`
