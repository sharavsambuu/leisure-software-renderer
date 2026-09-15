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
4. **2D Canvas vs. Screen Space in HUDs**: Forgetting the `y_canvas = (HEIGHT - 1) - y_screen` conversion when rendering 2D vector text or health bars alongside 3D projected screen coordinates.

---

## 7. No User Lock-In (Pluggability)

The renderer must never lock users into the engine's capacity. Users plug
whatever parts they want, whenever they want:

1. **Additive, not closed** — passes, techniques, materials, lights, light
   volumes, and compute effects are added *through* extension points
   (registries, contracts, recipes), never by editing the core. Builtin enums
   (`PassId`, technique/light presets) must eventually expose open registered
   ranges so consumer-owned abstractions need no core change (tracked in
   `docs/arch/render_path_architecture.md` §4).
2. **Backend choice is never a fork** — `SHS_RENDER_BACKEND` selects software /
   OpenGL / Vulkan at runtime; a build without GPU support degrades gracefully
   to software (backend factory fallback + hybrid auxiliary backends), never
   hard-fails. Software vs GPU is a driver-pod selection, not a product split.
3. **Shader language is a target, not a cage** — the same authored shading
   logic must be emittable to GLSL / Slang / C++ (material-system roadmap);
   users are never forced into one shader language.
4. **Demos and users are first-class extension authors** — custom passes and
   techniques register through the same mechanism builtin ones use; proven
   experiments graduate into builtin presets, never the reverse dependency.
5. **Replacement over abandonment** — any builtin part (preset, pass, technique,
   backend driver) must be replaceable by a user implementation honoring the
   same value-desc contract; the core must stay implementable as a thin,
   GPU-free-testable library (`shs-renderer-lib` end state).

*Formal extension-point contract: `docs/arch/render_path_architecture.md` §3–4.*

---

## 8. Code Style Law — Vertical Alignment (NASA/JPL rule)

All C/C++/Slang/GLSL sources in this repository follow the **NASA/JPL vertical
alignment style** (as used in JPL's C coding standard and Gerard Holzmann's
*The Power of Ten*). The purpose is human scan-ability: a reviewer must be able
to compare values, types, and names *down a column*, not token by token.

### The Law

1. **Aligned declaration blocks** — when consecutive lines declare related
   entities, align the type column, the name column, and the initializer column.
2. **Aligned assignment blocks** — in a block of related assignments, align the
   `=` signs (or the expression start) so values form a column.
3. **Aligned trailing comments** — when several lines carry end-of-line
   comments, align the comments into one column.
4. **Aligned member tables** — struct definitions that act as data tables
   (vertex layouts, state structs, descriptor setups) align type, name, and
   comment columns.
5. **Never break alignment silently** — renaming/re-typing a member realigns
   the whole block it belongs to. A diff that changes one identifier but leaves
   its column ragged is an incomplete change.

### Compliant example

```cpp
struct T0Vertex
{
    float    pos[3];    // NDC position (GLM_FORCE_DEPTH_ZERO_TO_ONE)
    float    col[4];    // vertex color (linear, unfiltered)
    float    uv[2];     // texture coords, may exceed 1.0 (REPEAT wrap)
};

const int    width    = 640;               // framebuffer width
const int    height   = 480;               // aspect-fixed height
const float  aspect   = width / height;    // guarded against div-by-zero
const size_t vcount   = quads.size();      // 6 verts per quad, 2 quads
```

### Non-compliant example

```cpp
// WRONG: ragged columns — the eye must re-focus on every line
const int width = 640;
const float aspect = width / height; // aspect
const size_t vcount = quads.size();
```

### Scope & limits

- **Applies to**: declarations/struct tables, related assignment blocks, enum
  and constant tables, trailing comment columns.
- **Does not apply to**: single isolated statements, control-flow bodies, or
  expressions where alignment would obscure operator precedence.
- **Tooling note**: alignment is whitespace-only and must never alter token
  semantics; keep `clang-format` exclusions minimal and prefer manual column
  alignment inside table-like blocks.
- Anchor example of the desired end state: the `VkDraw` table in
  `exps-rendering-adventures/tier0-rasterization-foundations/04_texture_sampling_scissor/texture_sampling_vk.cpp`.

---

## 9. Constitutional Links

 
 This document is Constitution I. SHS renderer also defines Constitution II for Value-Oriented Programming (VOP) and Constitution III for Data-Oriented Design (DOD).

 - **Constitution II (VOP & DOD)**: `docs/spec/value_oriented_programming.md` (Formal Specification)
 - **Constitution III (DOD & ECS)**: `docs/spec/dod_ecs_architecture.md`
 - **Constitutional rule of thumb**: keep pure value transforms in the center, keep backend side effects at execution boundaries, and prioritize cache-friendly Data-Oriented Design (SoA, ECS) for logic.
 - **Pluggability law (§7)**: no user lock-in — every part pluggable through extension points; see §7 and the §3–4 extension contract in `docs/arch/render_path_architecture.md`.
