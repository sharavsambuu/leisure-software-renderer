# Architecture: Game Lifecycle & UI Systems

While the ECS handles the "Simulation," these Meta-Systems handle the "Frame" of the game—the transitions between menus, world loading, and the player's interaction with the HUD.

## 1. Game State (The Life Cycle)
This is the high-level Finite State Machine (FSM) that controls the major modes of the application.
- **States**: `Startup`, `MainMenu`, `Loading`, `Gameplay`, `Paused`, `GameOver`.
- **Transitions**: 
    - `Gameplay` → `Paused`: Freezes the Physics and AI processors but keeps the UI system ticking.
    - `MainMenu` → `Loading`: Triggers the Scene Loader and clears the current ECS world.

## 2. Game Config (The Environment)
Config is persistent data that lives outside the ECS world.
- **Implementation**: A global `ConfigSystem` that loads/saves a JSON file.
- **Content**:
    - **Video**: Resolution, Fullscreen, Quality Presets.
    - **Audio**: Master, Music, SFX volumes.
    - **Input**: The hardware-to-action bindings discussed in the [Input Architecture](./action_based_input_architecture.md).

## 3. GUI Elements (The Interface)
UI is unique because it is often 2D and "Overlay" based, but needs to talk to the 3D ECS world.
- **The Library**: Integration with a library like **Dear ImGui** (for tools/debug) or **RmlUi** (for game HUDs using HTML/CSS style).
- **The ECS Link**:
    - **UI-to-World**: Clicking a button in the UI writes a "Command" or "Action" into an entity's `InputComponent`.
    - **World-to-UI**: The HUD reads a `HealthComponent` from the "Player" entity to update the HP bar.

## 4. The "Game Pause" Logic
Pause is not a stop; it is a **Processor Filter**.
- When `IsPaused = true`:
    - **Physics System**: Skips its simulation step.
    - **AI System**: Skips its behavior tree traversal.
    - **Animation System**: Freezes at the current frame.
    - **UI/Rendering/Input**: Continue to run so the player can navigate menus.

## 5. Persistence: The Save System
- **Serialization**: Leveraging the **Foundation Serializer** to dump the state of all components with the `SaveGame` tag into a JSON file.
- **Restoration**: The Scene Loader instantiates the level first, then overwrites properties from the save file.

---

> [!TIP]
> **Decoupling**: Keep your UI logic purely in Lua or a dedicated UI system. The `MeshComponent` shouldn't know anything about the "Options Menu."
