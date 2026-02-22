# Architecture: Physics System (Jolt & ECS)

Integrating a physics engine like **Jolt** into an ECS-driven architecture requires a strict separation of "Simulation Data" (Physics) and "Gameplay State" (ECS Components).

## 1. Physics as a First-Class Component
In a "Component-First" design, physics isn't a global manager; it's a set of data-driven components.
- **RigidBodyComponent**: Stores mass, friction, restitution, and a handle to the Jolt `BodyID`.
- **ColliderComponent**: Stores the shape (Box, Sphere, Capsule, Mesh) and local offset.
- **ConstraintComponent**: Defines joints (Hinge, Point, Slider) between two entities.

## 2. Layered Physics Composition
As you suggested, complex game elements are formed by **composing** these components.
- **Character Controller**: [Transform, RigidBody (Capsule), LogicScript (Lua)].
- **Multi-Part Vehicle**: A parent Entity with a `RigidBody`, and children Entities with `Wheel` and `Suspension` constraints.
- **Scripted Control**: Lua can dynamically add/remove `ConstraintComponents` or apply forces to specific `RigidBody` handles to create complex behaviors (e.g., a grappling hook).

## 3. The Sync Cycle (Simulation vs Rendering)
The most critical part of the architecture is the **Interpolation Bridge**.
1.  **Step 1: Input/Lua**: Scripting applies forces or adjusts velocities in the ECS components.
2.  **Step 2: Physics Sync (Write)**: The `PhysicsSystem` updates Jolt's internal state to match any changes in the ECS (e.g., manual teleporting).
3.  **Step 3: Sim Step**: Jolt simulates the world in its own optimized memory space.
4.  **Step 4: ECS Sync (Read)**: The `PhysicsSystem` reads Jolt's results and updates the `TransformComponent` of all active entities.

## 4. Multithreading & Jolt
Jolt is designed for high-performance multithreading.
- **Job System**: Jolt can use your engine's existing `JobSystem` to parallelize collision detection and constraint solving.
- **System Isolation**: The `PhysicsProcessor` can run in parallel with the `AnimationProcessor` or `ScriptProcessor`, as long as their dependencies are managed (e.g., Animation only drives bones, Physics only drives root transforms).

## 5. Workflow: Blender to Jolt
- **Collision Meshes**: The Blender exporter identifies objects tagged with `type: "COLLIDER"`.
- **Simplify**: Use Blender's "Custom Properties" to define if a mesh should be exported as a "Convex Hull" or a simplified "Bounding Box" for physics performance.
- **Triggers**: Tag non-colliding volumes in Blender to be exported as Jolt **Sensors**.
