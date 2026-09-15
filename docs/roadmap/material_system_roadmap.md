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

## Phase 3b: Multi-Target Shader Emission (GLSL / Slang / C++)
- **Goal**: The same authored shading logic must run on every backend — GLSL/Slang
      on GPU drivers, plain C++ (glm math) on the software driver.
- **Action**: Generalize the graph compiler's emission into targets: **GLSL**
      (SPIR-V via glslang/shaderc), **Slang** (Kronos-group language, compiles to
      SPIR-V/DXIL; adds modules/interfaces where the C++ side uses concepts), and
      **C++ (glm)** — a CPU evaluation of `shs_evaluate_material()` for
      `drivers/software`, keeping the software/Vulkan contract parity intact
      (material logic is shared; only the emission target changes).
- **Action**: Keep the node schema backend-neutral; language-specific constructs
      live in per-target emitter modules, not in material definitions.

## Phase 4: Production Optimization
- **Goal**: Ensure the system is performant and scalable.
- **Action**: Implement a **Vulkan Pipeline Cache** to reuse PSOs for identical graphs.
- **Action**: Integrate **Bindless Material Indexing** to allow switching material parameters without CPU-side descriptor updates.
- **Action**: Implement **Static Switches** to prune inactive branches from the generated shaders.

---

> [!NOTE]
> This document is part of the long-term architectural vision and does not require immediate implementation in current demo code.
