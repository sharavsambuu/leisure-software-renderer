# Roadmap: Physics System (Jolt Integration)

This roadmap outlines the steps to integrate the Jolt Physics engine, building on the **Core Engine Foundation**.

## Dependencies
- 🛣️ **[Core Engine Foundation](./core_engine_foundation_roadmap.md)** (Requires Phase 2: ECS Factory)

## Phase 1: Foundation (The Bridge)
- **Action: Jolt Integration**
    - Integrate the Jolt Physics library into the build system.
- **Action: Coordinate System Mapping**
    - Establish the conversion layer between SHS and Jolt.
- **Action: Component Definitions**
    - Register `RigidBody` and `Collider` components with the **Foundation Component Factory**.

## Phase 2: The Simulation Processor
- **Action: PhysicsSystem (Sim Step)**
    - Implement the step logic and transform sync.
- **Action: Shape Handling**
    - Implement the primitive and convex hull generation.

## Phase 3: Constraints & Prefabs
- **Action: Joint & Constraint Components**
    - Implement the physical constraints.
- **Action: Layered Physics Prefabs**
    - Update the **Foundation Resource Registry**'s prefab instantiation to handle joints and multi-body links.

## Phase 4: Scripting & Events
- **Action: Lua Physics Bindings**
    - Register physics functions into the **Foundation Binding Service**.
- **Action: Event Bridge**
    - Map Jolt contact events to the **Foundation Event Dispatcher**.

## Phase 5: Inter-System Bridges
- **Action: Ragdoll Support**
    - Implement the bridge to the **Animation System**.
    - Sync bone transforms to physical colliders.
- **Action: Character Controller**
    - Implement the kinematic/dynamic controller logic.

---

> [!TIP]
> **Data Integrity**: Ensure the **Physics System** is the "Source of Truth" for transforms when an entity is dynamic. The ECS `TransformComponent` should be a read-only mirror of the physics state during simulation.
