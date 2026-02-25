# Constitution III: Data-Oriented Design & ECS

This document defines **Constitution III**, the core tenets for structuring high-level logic and high-volume data processing within the SHS engine. It mandates the transition from OOP-based systems (`ILogicSystem` / `IRenderSystem`) towards a strict Data-Oriented Design (DOD) and Entity Component System (ECS) architecture.

## 1. The Core Philosophy

Object-Oriented Programming (OOP) focuses on the "things" (identities, encapsulated state, and behaviors). 
**Data-Oriented Design (DOD)** focuses on the **data** (how it's stored, how it's transformed, and how the CPU cache accesses it).

In SHS, the CPU cache is king. All engine architecture must prioritize cache-friendly data layouts and parallelizable, lock-free transformations over conceptual object hierarchies.

## 2. Struct of Arrays (SoA) over Array of Structs (AoS)

For any system processing hundreds or thousands of elements per frame (frustum culling, physics updates, transform hierarchies), **AoS is deprecated**.

*   **AoS (Bad)**: `std::vector<SceneObject>` where `SceneObject` contains a `Transform`, an `AABB`, a `Name`, and a `MaterialID`. Iterating to gather just the `AABB`s pollutes the CPU cache with unused `Name` and `MaterialID` data.
*   **SoA (Good)**: `std::vector<Transform>`, `std::vector<AABB>`. A culling system strictly iterates over contiguous `AABB` memory, achieving maximum cache-line utilization.

All new high-volume simulation code must default to SoA or hybrid AoSoA (Array of Struct of Arrays) layouts.

## 3. ECS as the High-Level Backbone

The high-level engine loop will be rewritten to use a strict Entity Component System (ECS). This replaces virtual inheritance trees.

1.  **Entities**: Are just integer IDs (`uint32_t`). They have no logic and no data.
2.  **Components**: Pure Plain Old Data (POD) structs. They are stored in dense, contiguous arrays (SoA).
3.  **Systems**: Pure, free functions that iterate over specific combinations of Component arrays. They contain **no internal state**.

### Example: Wait-Free Physics System
```cpp
// Pure function, no state. Takes inputs as read-only spans, outputs to exclusive spans.
void update_physics(std::span<const Transform> in_transforms, 
                    std::span<const Velocity> in_velocities, 
                    std::span<Transform> out_transforms, 
                    float dt) {
    for (size_t i = 0; i < in_transforms.size(); ++i) {
        out_transforms[i].position = in_transforms[i].position + (in_velocities[i].linear * dt);
    }
}
```

## 4. Generational Handles

Direct pointers (`Skin*`, `Material*`) and OS-level smart pointers (`std::shared_ptr`, `std::unique_ptr`) are forbidden for cross-object relationships in the simulation loop. They cause cache misses and make serialization/snapshots impossible.

All relationships must use **Generational Handles**:
*   A 32-bit or 64-bit integer combining an `index` into a flat array, and a `generation` counter to detect stale references (ABA problem).

```cpp
struct MaterialHandle {
    uint32_t index : 24;
    uint32_t generation : 8;
};
```

## 5. Wait-Free Concurrency Guarantee

Systems must be designed for **lock-free, wait-free parallel execution**. 
*   **No Mutexes/Atomics**: Systems must not use `std::mutex` or `std::atomic` during the simulation update.
*   **Exclusive Output**: A parallel job must be guaranteed exclusive write access to its slice of the output span.
*   **Read-Only Input**: Jobs read from immutable spans (`std::span<const T>`) populated in the previous frame or by a previous, fully completed pipeline stage. 

## 6. Zero-Allocation Simulation Loop
 
 To maintain wait-free concurrency, the simulation loop must never trigger OS-level heap locks. Standard global allocations via `new`, `malloc`, or `std::vector::push_back` (when it resizes) are strictly prohibited during the update frame.
 
 *   **Arena Allocators**: All transient jobs must leverage `std::pmr::monotonic_buffer_resource` initialized from pre-allocated, per-frame memory gigabytes.
 *   **Zero Collection**: At the start of a new frame, the arena pointer is simply reset to zero; individual objects are never `delete`d.
 
 ## 7. The Endgame: GPU-Driven Rendering
 
 The final state of the CPU ECS loop involves doing as little rendering work as possible.
 
 *   **Broad-Phase Only**: The CPU processes high-level logic and coarse bounding volume updates.
 *   **GPU Hand-off**: The CPU hands flat, contiguous buffers (SoA components) directly to the GPU.
 *   **Compute Culling & Indirect Draw**: Vulkan Compute Shaders perform all frustum/occlusion culling and generate `vkCmdDrawIndexedIndirect` commands, completely offloading the CPU from iterating over visible renderer instances.
 
 ## Summary
 By enforcing DOD, SoA, ECS, Generational Handles, and Zero-Allocation Loops, the engine achieves deterministic, high-performance, and infinitely scalable simulation capabilities, perfectly complementing the Value-Oriented Programming (Constitution II) rendering backend.
