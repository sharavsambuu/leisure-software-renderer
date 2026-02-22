# Roadmap: UI System (High-Performance 2D)

This roadmap outlines the steps to build a modern, high-performance UI system for the renderer.

## Phase 1: The 2D Bridge (Basic Rendering)
Establish the ability to draw simple shapes on top of the 3D world.
- **Action: 2D Overlay Pipeline**
    - Create a dedicated Vulkan pipeline for UI with depth testing disabled.
- **Action: Box & Line Renderer**
    - Implement a simple batching system to draw colored rectangles and lines (used for debug UI).

## Phase 2: Text & Typography
Implement high-quality text rendering.
- **Action: SDF Font Generator**
    - Implement (or integrate) a tool to convert `.ttf` fonts into Signed Distance Field (SDF) textures.
- **Action: SDF Text Shader**
    - Write the specialized UI shader that handles SDF rendering with support for outlines and glows.

## Phase 3: Layout & Interaction
Moving from static boxes to a dynamic interface.
- **Action: Dynamic Layout Engine**
    - Implement a basic "Container/Parent" system (Flexbox-lite) to handle resizing and alignment.
- **Action: Input Interceptor**
    - Integrate the UI with the **Action-Based Input System** to allow UI elements to consume mouse/keyboard events.

## Phase 4: Modern Integration (HUD)
- **Action: Dear ImGui Integration**
    - Use Phase 1 & 2 to host the **Dear ImGui** backend for full developer tooling.
- **Action: RmlUi / Custom HUD**
    - (Option A) Integrate **RmlUi** for a data-driven HUD using HTML/CSS.
    - (Option B) Build a simple Lua-driven HUD system where UI elements are just "2D Entities" in the ECS.

---

> [!IMPORTANT]
> **Performance Rule**: Always use **Texture Atlasing**. Swapping textures in the middle of a UI draw call is a heavy performance hit in Vulkan.
