# Roadmap: Game Lifecycle & UI Systems

This roadmap outlines the steps to implement the "Meta-Systems" that surround the game simulation.

## Phase 1: High-Level State (The Frame)
Establish the basic application lifecycle.
- **Action: GameStateManager**
    - Implement the high-level FSM that manages states like `MainMenu`, `Gameplay`, and `Paused`.
- **Action: Global Event Dispatcher**
    - Integrate with the **Foundation Event Service** to broadcast lifecycle events (e.g., `OnGamePaused`).

## Phase 2: User Interface (The HUD)
Build the visual layer of interaction.
- **Action: UI Library Integration**
    - Integrate a UI library (e.g., Dear ImGui for debug, RmlUi for HUD).
- **Action: HUD Component System**
    - Implement the logic for UI elements to read data from ECS components (e.g., `HealthBar` -> `HealthComponent`).

## Phase 3: Configuration & Persistence
Handle the "Outside World."
- **Action: Config System**
    - Implement a JSON-based settings loader for video, audio, and controls.
- **Action: Save/Load Service**
    - Leverage the **Foundation Serializer** to save the state of any entity with the `Persistent` tag.

## Phase 4: Polish & Tools
- **Action: Loading Screen System**
    - Implement an asynchronous loading system with progress updates from the **Foundation Resource Registry**.
- **Action: Debug Console**
    - Create a Quake-style overlay for real-time engine monitoring and Lua command execution.

---

> [!TIP]
> **Pause correctly**: When implementing "Pause," don't stop the main loop. Instead, use a **Time Scale** of `0.0` for the Physics and Animation processors while keeping the UI processor at `1.0`.
