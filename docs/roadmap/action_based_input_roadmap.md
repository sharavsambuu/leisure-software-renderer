# Roadmap: Action-Based Input System

This roadmap outlines the steps to implement a unified input system, building on the **Core Engine Foundation**.

## Dependencies
- 🛣️ **[Core Engine Foundation](./core_engine_foundation_roadmap.md)** (Requires Phase 1: Serialization & Phase 2: ECS Factory)

## Phase 1: The Action Manifest
- **Action: Action Registry**
    - Implement the C++ system to define and store semantic action names (e.g., `Jump`).
- **Action: Binding Loader**
    - Leverage the **Foundation Serializer** to load JSON input maps for keyboard and gamepad.

## Phase 2: The Agent Interface
- **Action: InputComponent**
    - Register the `InputComponent` with the **Foundation Component Factory**.
- **Action: Hardware Agent**
    - Implement the platform-specific polling (GLFW/SDL) that writes into the `InputComponent`.

## Phase 3: Agency Expansion (AI & Scripting)
- **Action: Orchestration Bridge**
    - Create Behavior Tree nodes in the **State Orchestration System** that can write to the `InputComponent`.
- **Action: Lua Bindings**
    - Register the input action check functions into the **Foundation Binding Service**.

## Phase 4: Recording & Replays
- **Action: Action Serialization**
    - Utilize the **Foundation Serializer** to record the stream of input actions for physics-accurate replays.

---

> [!TIP]
> **Agency Neutrality**: By writing your gameplay code to read the `InputComponent`, it becomes automatically compatible with Replay systems and AI without any additional code.
