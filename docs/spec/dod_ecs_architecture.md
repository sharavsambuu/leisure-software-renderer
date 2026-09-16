# Constitution III: Domain-Owned Data Layout & Execution

This document defines **Constitution III**, the storage and execution laws supporting **Kleisli Domain Boundary Architecture (KDBA) in C++23**. **Domain separation and ownership are foundational; KDBA is the chosen programming paradigm, not a combination of legacy programming models.** ECS is explicitly rejected as the programming model. SoA, contiguous arrays, handles, arenas, and chunked jobs are implementation techniques under domain ownership, not an ECS mandate. Composition quality, explicit failure handling, deterministic transitions, and efficient execution govern review; a uniform gateway signature is not a goal in itself, and infallible transforms must not acquire invented error channels merely for visual conformity. Historical analogies do not define the architecture. **This governing clarification (2026-09-17) supersedes older ECS-programming-model and universal-signature mandates wherever restated in the constitutions or backlogs**, without relaxing Core 4, domain-write isolation, purity, fact preservation, atomic commit for fallible workflows, or memory lifetime laws. The legacy filename is retained for stable links.

- **Constitution I**: `docs/spec/conventions.md` (Units, Coordinate Systems, Physics Bridge, Lighting Semantics)
- **Constitution II**: `docs/spec/value_oriented_programming.md` (Value-Oriented Programming & Gateway Architecture)
- **Constitution III (This Document)**: Domain-Owned Data Layout & Execution

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

## 3. Domain Pods Own Behavior; Tables Support Execution

The high-level engine is organized by domain ownership and typed KDBA composition, not a global entity/component world or inheritance hierarchy.

1. **Domain identity**: Use domain-owned IDs and generational handles for stable relationships. Integer handles do not imply an ECS entity model.
2. **Domain storage**: Plain value contracts own contiguous SoA tables for high-volume data. Chunking follows the memory laws, not a mandatory ECS archetype model.
3. **Stages and kernels**: Pure stages compose decisions; fallible stages use `std::expected`, while infallible transforms return values. Batch kernels stream over immutable inputs and exclusive outputs. Persistent buffers are not mutated speculatively mid-chain; the boundary commits accepted transitions atomically (Constitution II §2.2(4)).
4. **Orchestration and scheduling**: Cross-domain workflows belong to orchestrator pods with their own contracts, actions, transitions, and events (Constitution II Rules 11–12). Execution-edge schedulers dispatch jobs; they do not own domain policy or cross-domain state.

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
* **Explicit Schedulers**: Standard parallel algorithms (`std::execution::par_unseq`) surrender scheduling, core pinning, and arena awareness to the implementation. Hot paths use explicit chunked workers with exclusive output spans; implementation-scheduled parallelism is allowed only where scheduler and arena behavior are explicit and pinned by test (Virtual SPU trajectory).

---

## 7. Zero-Allocation Simulation Loop

To maintain wait-free concurrency, the simulation loop must never trigger OS-level heap locks. Standard global allocations via `new`, `malloc`, or `std::vector::push_back` (when it resizes) are strictly prohibited during the update frame.

* **Arena Allocators**: All transient jobs must leverage `std::pmr::monotonic_buffer_resource` initialized from pre-allocated, per-frame memory buffers ($8\text{–}64\,\text{MB}$).
* **Zero Collection Overhead**: At the start of a new frame, the arena pointer is simply reset to zero in $\mathcal{O}(1)$; individual objects are never `delete`d.

---

## 8. The Endgame: GPU-Driven Rendering

For GPU-driven backends, domain-owned CPU planning aims to minimize rendering work:

* **Broad-Phase Only**: The CPU processes high-level logic, game rules, and coarse bounding volume updates.
* **GPU Hand-off**: The CPU hands flat, contiguous buffers (SoA components) directly to GPU Storage Buffers (SSBOs).
* **Compute Culling & Indirect Draw**: Vulkan Compute Shaders perform all frustum/occlusion culling and generate `vkCmdDrawIndexedIndirect` commands, completely offloading the CPU from iterating over visible renderer instances.

---

## Summary
Domain-owned SoA tables, generational handles, flat hierarchies, and zero-allocation loops support efficient KDBA execution. These implementation techniques do not introduce an ECS programming model or replace Constitution II's domain ownership and composition laws.