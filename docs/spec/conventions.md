# SHS Renderer Conventions & Specifications

This document defines the mathematical and spatial "laws" of the SHS Renderer. It covers units, coordinate systems, lighting semantics, and integration rules for backends (Vulkan/Software) and physics (Jolt).

---

## 1. Units & Constants

SHS uses SI-style runtime units. There is a **1:1 scale mapping** between SHS and Jolt physics.

| Quantity | Unit | Note |
| :--- | :--- | :--- |
| **Distance** | Meter (m) | `1.0` world unit = `1 meter` |
| **Mass** | Kilogram (kg) | |
| **Time** | Second (s) | |
| **Angle** | Radian (rad) | |
| **Velocity** | m/s | |
| **Acceleration** | m/s² | |
| **Force** | Newton (N) | `kg * m/s²` |

### Gravity
- **Direction**: `-Y` (Points down)
- **Magnitude**: `9.81 m/s²`
- **World Vector**: `(0, -9.81, 0)`

### Code Anchors
- **Constants**: `shs/core/units.hpp`
- **Jolt Bridge**: `shs/geometry/jolt_adapter.hpp`

---

## 2. SHS Coordinate System (Render Space)

The renderer operates primarily in **Left-Handed (LH)** space with **+Y up** and **+Z forward**.

### pipeline Stages
`Model -> World -> View -> Projection -> NDC -> Screen -> shs::Canvas`

### Axes (Model/World/View)
- **+X**: Right
- **+Y**: Up
- **+Z**: Forward (Into the screen)

### NDC (Normalized Device Coordinates)
- **Z-Range**: `[-1.0, 1.0]` (OpenGL style, handled via `glm::perspectiveLH_NO`)
- **+Y**: Up

### Screen vs. Canvas (2D)
- **Screen**: Origin `(0,0)` at **Top-Left**. Handles windowing/UI coordinates.
- **shs::Canvas**: Origin `(0,0)` at **Bottom-Left**. Used for software rasterization.
- **Conversion**: `y_canvas = SCREEN_HEIGHT - y_screen`

---

## 3. Jolt Integration (Physics Space)

Jolt uses a **Right-Handed (RH)** system with **+Y up**. It shares X and Y with SHS but has an opposite Z axis.

### Conversion Rules
All conversions reduce to **negating the Z component**.

| Type | Conversion (SHS -> Jolt) |
| :--- | :--- |
| **Position** | `(x, y, -z)` |
| **Direction** | `(x, y, -z)` |
| **Quaternion**| `(-x, -y, z, w)` (Z-flip conjugation) |
| **AABB** | `min = (shs_min.x, shs_min.y, -shs_max.z)`, `max = (shs_max.x, shs_max.y, -shs_min.z)` |
| **Matrix** | `M_jolt = S * M_shs * S` where `S = diag(1, 1, -1, 1)` |

---

## 4. Lighting & Shading Semantics

To prevent "direction drift," all shading math remains in **SHS World Space**.

### canonical Direction: `sun_dir_to_scene_ws`
- **Meaning**: Normalized vector pointing **from the sun toward the scene**.
- **Incoming Light (L)**: `normalize(-sun_dir_to_scene_ws)`
- **View (V)**: `normalize(camera_pos_ws - world_pos_ws)`
- **Half (H)**: `normalize(L + V)`

### Space Ownership
1. **SHS Space**: Owns render-space math, camera/light vectors, and BRDF inputs.
2. **Jolt Space**: Owns geometry storage, culling backends, and physics representation.

---

## 5. Backend Conventions (Vulkan vs. Software)

### Vulkan
- **Z-Range**: Vulkan native is `[0, 1]`. SHS projects to `[-1, 1]` and performs an explicit Mapping at the pass boundary or via `vkCmdSetViewport`.
- **Y-Axis**: Vulkan native Y is down. SHS uses a **negative-height viewport** to maintain **+Y up** across both backends.

### Software
- **Rasterization**: Happens in `shs::Canvas` (Bottom-Left origin).
- **Presentation**: Rows are vertically flipped during swapchain upload to match screen space (Top-Left).

---

## 6. Common Pitfalls
1. **Z-Flip**: Neglect of the Z-flip when passing SHS AABBs to Jolt for culling.
2. **Sun Direction**: Mixing up "to scene" vs "to light" directions. Always use `sun_dir_to_scene_ws`.
3. **NDC Mapping**: Forgetting the `[-1, 1]` to `[0, 1]` conversion when porting software shadow logic to Vulkan.

---

## 7. Constitutional Link: VOP

This document is Constitution I. SHS renderer also defines Constitution II for Value-Oriented Programming (VOP).

- **Constitution II**: `docs/spec/value_oriented_programming.md`
- **Constitutional rule of thumb**: keep pure value transforms in the center, keep backend side effects at execution boundaries.
