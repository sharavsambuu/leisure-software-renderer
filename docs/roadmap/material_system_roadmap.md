# Roadmap: Composed Material Graph System

This roadmap outlines the implementation phases for transitioning the renderer to a fully dynamic, node-based material system.

## Phase 1: Modular GLSL Library
- **Goal**: Standardize core lighting and PBR math into reusable includes.
- **Action**: Move all lighting equations (Cook-Torrance, Blinn-Phong) into a standardized library in `shaders/lib/`.
- **Action**: Unify scene and object constant layouts into shared headers.

## Phase 2: Shader Templating & C++ Assembler
- **Goal**: Implement the runtime logic to "stitch" shaders together.
- **Action**: Create skeleton "Pass Templates" for Forward and Deferred paths.
- **Action**: Implement a C++ `ShaderAssembler` that combines Library code, Template code, and Material-specific logic into a final GLSL string.

## Phase 3: Material Graph Compiler (Node-to-GLSL)
- **Goal**: Automate the generation of material logic.
- **Action**: Define a node-based schema (JSON/C++) for mathematical operations and texture samples.
- **Action**: Implement a compiler that traverses these graphs and emits the `shs_evaluate_material()` function.

## Phase 4: Production Optimization
- **Goal**: Ensure the system is performant and scalable.
- **Action**: Implement a **Vulkan Pipeline Cache** to reuse PSOs for identical graphs.
- **Action**: Integrate **Bindless Material Indexing** to allow switching material parameters without CPU-side descriptor updates.
- **Action**: Implement **Static Switches** to prune inactive branches from the generated shaders.

---

> [!NOTE]
> This document is part of the long-term architectural vision and does not require immediate implementation in current demo code.
