# Tuning & Control: The "Director" Layer

A game engine is useless if you have to recompile it every time you want to change a light's intensity or a player's run speed. We handle this through a two-tier "Director" system.

## 1. CVars (Console Variables)
CVars are the "Global Knobs" of the engine. They live in the **Foundation Layer** and are accessible from everywhere.
- **The Concept**: A simple string-to-value map (e.g., `r_ssao_radius = 0.5`).
- **Persistence**: CVars are saved to a `.cfg` file and loaded on startup.
- **The Engine Console**: You can open a dev console (using the **UI System**) and type `r_shadow_softness 2.0` to see the effect instantly.

## 2. The Director (Blackboard)
While CVars are for technical settings, the **Director** manages gameplay state.
- **Global Blackboard**: A shared data structure (Component-like) that stores high-level facts about the world (e.g., `is_daytime: true`, `alert_level: 5`).
- **Tunables**: Lua scripts read from this blackboard to decide how many enemies to spawn or how aggressive they should be.

## 3. Real-time Tweaking (Dear ImGui)
We integrate **Dear ImGui** to provide a visual "Tweak Panel."
- **Auto-Discovery**: Our UI system can iterate through the ECS components and generate a "Properties" window.
- **Value Sliders**: Slide a float in the UI, and the C++ data changes instantly in the ECS.
- **Material Editing**: Adjust a shader parameter in the UI, and the **Material System** re-uploads the uniform buffer to Vulkan without a hitch.

## 4. Case Study: The Day/Night Cycle
This is the perfect example of "Total Mutability."
1. **The State**: A global `TimeComponent` tracks the hour of the day.
2. **The Logic**: A `SkyboxSystem` (C++ or Lua) reads the time and calculates the sun's position.
3. **The Mutation**: The system updates the `DirectionalLightComponent` and the `SkyboxMaterial` uniforms every frame.
4. **The Result**: The sun moves, shadows lengthen, and the sky color shifts—all because we're mutating common components in real-time.

---

> [!TIP]
> **Developer Command**: If you find yourself hard-coding a magic number in C++, **stop**. Move it to a CVar or a Component so you can tune it while the game is running.
