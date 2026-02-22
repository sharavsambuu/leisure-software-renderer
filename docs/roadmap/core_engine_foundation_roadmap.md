# Roadmap: Core Engine Foundation

To ensure maximum modularity and prevent duplication, this roadmap establishes the "Common Services" used by all other sub-systems (Rendering, Physics, AI, etc.).

## Phase 1: The "Standard Library" (Serialization & Reflection)
Common utilities for data-driven persistence.
- **Action: Core Serializer Bridge**
    - Integrate a unified JSON library (e.g., `nlohmann/json`).
    - Create the `Serializable` interface for all engine components.
- **Action: Console Variables (CVars)**
    - Implement the `CVar` registry for global engine settings (e.g., `r_ssao_radius`).
    - Support loading/saving CVars to `.cfg` files.

## Phase 2: The "Spine" (Advanced ECS + Tuning UI)
The foundation for all system integration.
- **Action: Component Factory**
    - Implement a centralized registry that can instantiate any component by its JSON string name.
- **Action: Dear ImGui Tuning UI**
    - Integrate the backend for real-time ECS component and CVar tweaking.

## Phase 3: The "Brain Bridge" (Universal Scripting)
Establish the Lua environment once.
- **Action: Universal Lua State**
    - Integrate `sol2` and create a persistent, high-performance Lua environment.
- **Action: Binding Service**
    - Create the `BindingRegistry` where each sub-system (Physics, Animation) registers its own Lua API without knowing about the other systems.

## Phase 4: The "Threading" (Job System)
- **Action: Task-Based Parallelism**
    - Implement a task-stealing job system that all system "Processors" use to parallelize their work.

---

> [!IMPORTANT]
> **Modularity Rule**: Any code that needs to be used by more than one sub-system (e.g., Reading a file, Binding a C++ struct to Lua) belongs here in the **Foundation**, not in the specific system's code.
