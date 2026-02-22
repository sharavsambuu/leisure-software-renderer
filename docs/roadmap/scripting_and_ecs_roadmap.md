# Roadmap: Lua & ECS Integration

This roadmap outlines the steps to build high-level gameplay logic using the foundation established in the **Core Engine Foundation**.

## Dependencies
- 🛣️ **[Core Engine Foundation](./core_engine_foundation_roadmap.md)** (Requires Phase 3: Brain Bridge)

## Phase 1: The Scripting Layer
Implement the ECS system that manages the execution of scripts.
- **Action: ScriptComponent & ScriptSystem**
    - Define the `ScriptComponent` to hold file paths and Lua state handles from the **Foundation Binding Service**.
    - Implement the `ScriptSystem` to handle `OnInit`, `OnUpdate`, and `OnDestroy` lifecycles.
- **Action: Sandboxing & Scoping**
    - Ensure each entity's script runs in its own Lua table to prevent variable leakage between entities.

## Phase 2: Event-Driven Gameplay
Move beyond polling and into efficient event handling.
- **Action: Engine Event Bridge**
    - Register standard engine events (Collision, TriggerEnter, InputPressed) into the **Foundation Event Dispatcher**.
- **Action: Gameplay Module**
    - Implement a central "Lua Gameplay Module" to handle global game rules, score tracking, and win/loss conditions.

## Phase 3: Workflow Integration
Connect the scripting system to the content authoring pipeline.
- **Action: Blender Script Picker**
    - Update the Blender exporter to recognize `.lua` file paths in custom properties.
- **Action: Scene Loader Binding**
    - Update the **Foundation Component Factory** to automatically attach `ScriptComponents` when script tags are present in the JSON entity definitions.

---

> [!TIP]
> **Priority Path**: Use the **Foundation Binding Service** to expose C++ components. Do not write manual binding code inside the Scripting System; keep it centralized in the Foundation Layer.
