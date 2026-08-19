# Constitution III: Data-Oriented Design & ECS

This document defines **Constitution III**, the core tenets for structuring high-level logic and high-volume data processing within the SHS engine. It mandates the transition from OOP-based systems towards a strict Data-Oriented Design (DOD) and Entity Component System (ECS) architecture.

- **Constitution I**: `docs/spec/conventions.md` (Units, Coordinate Systems, Physics Bridge, Lighting Semantics)
- **Constitution II**: `docs/spec/value_oriented_programming.md` (Value-Oriented Programming & Reducer Architecture)
- **Constitution III (This Document)**: Data-Oriented Design & Entity Component System

---

## 1. The Core Philosophy

Object-Oriented Programming (OOP) focuses on the "things" (identities, encapsulated state, and behaviors). 
**Data-Oriented Design (DOD)** focuses on the **data** (how it's stored, how it's transformed, and how the CPU cache accesses it).

In SHS, the CPU cache is king. All engine architecture must prioritize cache-friendly data layouts and parallelizable, lock-free transformations over conceptual object hierarchies.

---

## 2. Struct of Arrays (SoA) over Array of Structs (AoS)

For any system processing hundreds or thousands of elements per frame (frustum culling, physics updates, transform hierarchies), **AoS is deprecated**.

* **AoS (Forbidden)**: `std::vector<SceneObject>` where `SceneObject` contains a `Transform`, an `AABB`, a `Name`, and a `MaterialID`. Iterating to gather just the `AABB`s pollutes the CPU cache with unused `Name` and `MaterialID` data.
* **SoA (Mandatory)**: Parallel contiguous arrays (`std::vector<Transform>`, `std::vector<AABB>`). A culling system strictly iterates over contiguous `AABB` memory, achieving maximum L1/L2 cache-line utilization and automatic SIMD (AVX2/AVX-512) vectorization.

All high-volume simulation code must default to SoA or Archetype Chunked SoA layouts.

---

## 3. ECS as the High-Level Backbone

The high-level engine loop utilizes a strict Entity Component System (ECS) that completely replaces virtual inheritance trees (`class Monster : public Actor`).

1. **Entities**: Are just lightweight integer IDs (`uint32_t`). They have no logic and no data.
2. **Components**: Pure Plain Old Data (POD) structs. They are stored in dense, contiguous Archetype SoA chunks ($16\,\text{KB}$ cache-aligned chunks).
3. **Systems (VOP Reducers)**: Pure, stateless free functions that iterate over specific combinations of component arrays. They contain **no internal mutable state** and emit discrete events.

### Example: Wait-Free Physics System
```cpp
// Pure function, no state. Takes inputs as read-only spans, outputs to exclusive spans.
void update_physics(std::span<const glm::vec3> in_positions,
                    std::span<const glm::vec3> in_velocities,
                    std::span<glm::vec3>       out_positions,
                    float dt) {
    for (size_t i = 0; i < in_positions.size(); ++i) {
        out_positions[i] = in_positions[i] + (in_velocities[i] * dt);
    }
}
```

---

## 4. Topologically Sorted Flat Hierarchies

Scene graphs and skeletal bone hierarchies must never use recursive pointer trees (`parent->children[]`).
* All hierarchies must be stored as **Topologically Sorted Flat Arrays** with integer parent indices (`parent_index < child_index`).
* Evaluating bone/scene transforms occurs in a **single, forward linear pass** with zero recursion, zero pointer dereferences, and zero cache misses:

```cpp
struct HierarchyNode {
    uint32_t parent_index; // Must be strictly smaller than current index
    glm::mat4 local_transform;
};

void evaluate_hierarchy(std::span<const HierarchyNode> nodes, 
                        std::span<glm::mat4>           out_world_transforms) {
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (nodes[i].parent_index == i) {
            out_world_transforms[i] = nodes[i].local_transform; // Root
        } else {
            out_world_transforms[i] = out_world_transforms[nodes[i].parent_index] * nodes[i].local_transform;
        }
    }
}
```

---

## 5. Generational Handles

Direct pointers (`Skin*`, `Material*`, `Entity*`) and OS-level smart pointers (`std::shared_ptr`, `std::unique_ptr`) are strictly forbidden for cross-object relationships in the simulation loop. They cause cache misses and make serialization/snapshots impossible.

All relationships must use **Generational Handles**:
* A 32-bit packed integer combining a **24-bit index** into a dense component array and an **8-bit generation counter** to detect stale references (ABA problem).

```cpp
struct EntityHandle {
    uint32_t index      : 24; // Up to 16.7 million entities
    uint32_t generation : 8;  // Recycled up to 256 times
};
```

---

## 6. Wait-Free Concurrency Guarantee

Systems must be designed for **lock-free, wait-free parallel execution**:
* **No Mutexes/Atomics**: Systems must not use `std::mutex` or `std::atomic` during simulation updates.
* **Exclusive Output**: A parallel job must be guaranteed exclusive write access to its slice of the output span.
* **Read-Only Input**: Jobs read from immutable spans (`std::span<const T>`) populated in the previous frame or by a previous, fully completed pipeline stage.

---

## 7. Zero-Allocation Simulation Loop

To maintain wait-free concurrency, the simulation loop must never trigger OS-level heap locks. Standard global allocations via `new`, `malloc`, or `std::vector::push_back` (when it resizes) are strictly prohibited during the update frame.

* **Arena Allocators**: All transient jobs must leverage `std::pmr::monotonic_buffer_resource` initialized from pre-allocated, per-frame memory buffers ($8\text{–}64\,\text{MB}$).
* **Zero Collection Overhead**: At the start of a new frame, the arena pointer is simply reset to zero in $\mathcal{O}(1)$; individual objects are never `delete`d.

---

## 8. The Endgame: GPU-Driven Rendering

The final state of the CPU ECS loop involves doing as little rendering work as possible:

* **Broad-Phase Only**: The CPU processes high-level logic, game rules, and coarse bounding volume updates.
* **GPU Hand-off**: The CPU hands flat, contiguous buffers (SoA components) directly to GPU Storage Buffers (SSBOs).
* **Compute Culling & Indirect Draw**: Vulkan Compute Shaders perform all frustum/occlusion culling and generate `vkCmdDrawIndexedIndirect` commands, completely offloading the CPU from iterating over visible renderer instances.

---

## Summary
By enforcing DOD, SoA, ECS, Generational Handles, Flat Hierarchies, and Zero-Allocation Loops, the engine achieves deterministic, high-performance, and infinitely scalable simulation capabilities, perfectly complementing the Value-Oriented Programming (Constitution II) rendering backend.