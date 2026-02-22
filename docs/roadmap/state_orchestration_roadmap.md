# Roadmap: State Orchestration (Behavior & Logic)

This roadmap outlines the steps to implement a centralized "Director" system, building on the **Core Engine Foundation**.

## Dependencies
- 🛣️ **[Core Engine Foundation](./core_engine_foundation_roadmap.md)** (Requires Phase 1: Serialization & Phase 2: ECS Factory)

## Phase 1: Finite State Machines (FSM)
Implement the first layer of high-level state logic.
- **Action: The State Machine Processor**
    - Define the `StateComponent` that holds a handle to a simple JSON-based FSM.
    - Leverage the **Foundation Serializer** for state transition definitions.
- **Action: High-Level Game states**
    - Use the FSM system to drive the "Loading → Menu → Gameplay" transitions.

## Phase 2: Behavior Trees (The Action System)
Implement the core of the engine's intelligence.
- **Action: Behavior Tree Runner**
    - Implement the logic for `Selector`, `Sequence`, and `Decorator` nodes.
    - Use the **Foundation Blackboard** for all condition checks.
- **Action: Condition & Action Interfaces**
    - Create the C++ base classes that allowed systems to register "Conditions" and "Actions" into the **Foundation Binding Service**.

## Phase 3: AI Perception & Senses
Give the Orchestrator "Eyes" and "Ears."
- **Action: AISenseSystem**
    - Implement a system to update the **Foundation Blackboard** with world data.
- **Action: Navigation Bridge**
    - Integrate a NavMesh system (like Recast) and add "MoveTo" nodes to the Behavior Tree.

## Phase 4: Workflow & Scripting
Connect the orchestrator to the content creators.
- **Action: Lua Node Bridge**
    - Allow "Leaf Nodes" in the Behavior Tree to execute Lua scripts via the **Foundation Binding Service**.
- **Action: JSON Logic Export**
    - Update the **Foundation Resource Registry** to support loading of `.behavior` JSON files.
- **Action: Real-time Debugging**
    - Use the console/overlay to display the active node path from the Blackboard.

---

> [!TIP]
> **Modular Logic**: Keep the Behavior Tree logic in JSON and the individual Action code in Lua. This ensures that "Logic" and "Implementation" are perfectly separated.
