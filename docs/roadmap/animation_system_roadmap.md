# Roadmap: Animation System

This roadmap outlines the steps to implement a skeletal animation system, building on the **Core Engine Foundation**.

## Dependencies
- 🛣️ **[Core Engine Foundation](./core_engine_foundation_roadmap.md)** (Requires Phase 1: Resource Registry)

## Phase 1: Foundation (Skeletal Basics)
- **Action: glTF Skin Loading**
    - Integrate with the **Foundation Resource Registry** to parse and cache skeletal data.
- **Action: CPU Skeletal Solver**
    - Implement the hierarchical transform resolver for joints.

## Phase 2: Performance (GPU Skinning)
- **Action: Bone Buffer Management**
    - Implement matrix block uploads to the GPU via the renderer's SSBO/UBO services.
- **Action: GPU Skinning Shader**
    - Write the vertex shader logic for bone-weighted transformation.

## Phase 3: Blending & Logic
- **Action: Animation Blending**
    - Implement LERP/SLERP for track mixing.
- **Action: State Machine Engine**
    - Implement the logic for clip transitions and layering.

## Phase 4: Procedural Magic (IK & Secondary)
- **Action: Two-Bone IK Solver**
    - Implement analytic IK for foot/hand placement.
- **Action: Procedural Shake/Jiggle**
    - Implement noise-based transform offsets for secondary motion.

## Phase 5: Bridge: Physics Ragdoll
- **Action: Skeleton-to-Body Mapping**
    - Leverage the **Physics System** to map bones to physical colliders.
    - Implement the "State Switch" logic where the **Physics System** takes over bone transforms.
    - *Note: This phase is a collaboration with the [Physics System Roadmap](./physics_system_roadmap.md).*

---

> [!TIP]
> **Priority Path**: Get the **Resource Registry** (Foundation Phase 1) working first so you don't re-write the `.gltf` parser logic.
