# Roadmap: ECS Architecture & Lua Scripting

This roadmap outlines the steps to build high-level gameplay logic using the **ECS Backbone** (Data-Oriented Design) established in [Constitution III](../spec/dod_ecs_architecture.md).

## Dependencies
- 🛣️ **[Core Engine Foundation](./core_engine_foundation_roadmap.md)** (Requires Phase 3: Brain Bridge)
- 📜 **[Constitution III](../spec/dod_ecs_architecture.md)** (Strict ECS and DOD adherence)

## Phase 1: The Core ECS Layer (C++)
Establish the Entity Component System as the definitive way C++ game logic executes.
- **Action: Entity Registry & Handles**
    - Implement a dense Entity Registry using Generational Handles (no pointers).
- **Action: Component SoA Storage**
    - Implement tightly packed Struct of Arrays (SoA) component storage for cache-friendly iteration.
- **Action: Wait-Free Systems**
    - Ensure Systems are pure C++ functions processing read-only component spans and outputting to exclusive writable spans.

## Phase 2: The Scripting Layer (Lua)
Integrate Lua scripting *on top* of the C++ ECS backbone.
- **Action: ScriptComponent & ScriptSystem**
    - Define the `ScriptComponent` to hold file paths and Lua state handles from the **Foundation Binding Service**.
    - Implement the `ScriptSystem` to handle `OnInit`, `OnUpdate`, and `OnDestroy` lifecycles for entities with attached scripts.
- **Action: Sandboxing & Scoping**
    - Ensure each entity's script runs in its own Lua table to prevent variable leakage between entities.

## Phase 3: Event-Driven Gameplay
Move beyond polling and into efficient event handling.
- **Action: Engine Event Bridge**
    - Register standard engine events (Collision, TriggerEnter, InputPressed) into the **Foundation Event Dispatcher**.
- **Action: Gameplay Module**
    - Implement a central "Lua Gameplay Module" to handle global game rules, score tracking, and win/loss conditions.

## Phase 4: Workflow Integration
Connect the scripting system to the content authoring pipeline.
- **Action: Blender Script Picker**
    - Update the Blender exporter to recognize `.lua` file paths in custom properties.
- **Action: Scene Loader Binding**
    - Update the **Foundation Component Factory** to automatically attach `ScriptComponents` when script tags are present in the JSON entity definitions.

---

> [!TIP]
> **Priority Path**: The ECS is not just for Lua; it's the primary way C++ engine logic should run. Use the **Foundation Binding Service** to expose C++ components to Lua. Do not write manual binding code inside the Scripting System; keep it centralized in the Foundation Layer.
