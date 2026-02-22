# Deep-Dive: UI System Implementation

A game UI system is a specialized 2D renderer that overlays the 3D world. For your renderer, we need to handle rendering, layout, and text with high performance.

## 1. The Rendering Pipeline
UI is rendered as the final "Overlay" pass in the frame.
- **Coordinate System**: Screen-space (pixels) or Normalized Device Coordinates (NDC, -1 to 1). Using pixels (0 to width/height) is usually easier for designers.
- **Orthographic Projection**: A simple matrix that maps pixel coordinates to NDC.
- **Vulkan Pipeline**:
    - **Depth Test**: Disabled (we use paint order/Z-index).
    - **Blending**: Enabled (Alpha blending for transparency).
    - **Vertex Format**: `[vec2 pos, vec2 uv, u8vec4 color]`.

## 2. Text Rendering (SDF vs. Atlas)
Text is the hardest part of UI.
- **Option A: Bitmap Atlas**: Fast, but gets blurry when scaled.
- **Option B: Signed Distance Fields (SDF)**: **(Recommended)**. 
    - Vertices store UVs pointing to an SDF texture.
    - Shader performs an alpha-test to find the edge.
    - Benefits: Crisp edges at any zoom level and easy "Glow/Outline" effects.

## 3. The Layout Engine (Flexbox vs. Absolute)
- **Absolute**: Simple "X, Y" positioning. Hard to handle different resolutions.
- **Flexbox**: (Like CSS). Elements automatically align, wrap, and fill space. 
    - Integration: Use a library like **Yoga** (C++) to calculate the bounding boxes, then generate the vertices.

## 4. Input Routing (The "Blocking" Problem)
The UI must intercept input before the 3D world.
1.  **Input Event** (e.g., Mouse Click).
2.  **UI Check**: Does the cursor hit a UI element?
3.  **Block/Pass**:
    - If **HIT**: UI consumes event, logic executes (e.g., button click).
    - If **MISS**: Pass event to the `CharacterInputProcessor` in the ECS.

## 5. Modern UI Libraries for C++
- **Dear ImGui**: Industry standard for debug menus and tools. Immediate mode.
- **RmlUi**: High-performance "XHTML/CSS" based UI. Retained mode.
- **Custom (SDF-Based)**: If you want total control and tiny footprint, build a simple "Box & Text" renderer using SDFs.

---

> [!TIP]
> **Batching is Key**: Try to render the entire UI in a single Vulkan Draw Call. Use a **Texture Atlas** containing all UI icons and font characters to avoid mid-UI texture swaps.
