# Modern Material System Architecture

Moving from a fixed PBR shader to a flexible material system involves shifting from **monolithic "Uber-shaders"** to **composed material graphs**. This allows for complex effects like layered snow, animated lava, or custom stylized shading.

## 1. Material Graphs (The Node-Based Approach)
In AAA engines, materials are rarely "hand-coded." Instead, they are represented as a Directed Acyclic Graph (DAG) of operations.

- **Nodes**: Math ops (Add, Mult), Texture Samples, Vector Math, and Time-based inputs.
- **The Core Problem**: GPUs don't execute "graphs" directly. They execute compiled SPIR-V/GLSL code.
- **The Solution**: **Shader Permutation Generation**.
    - The graph is traversed from the "Output" node (Emissive, Albedo, Roughness) back to the inputs.
    - Code fragments are stitched together to produce a specialized shader for that specific material.

## 2. Parameterization vs. Specialization
There are two main ways to handle many different materials:

### A. The "Uber-Shader" (Current State)
One massive shader with many toggles and parameters.
- **Pros**: Low pipeline churn (you only have one pipeline).
- **Cons**: High "Instruction Pressure" (lots of `if` statements or branchy code), wasted registers.

### B. Specialized Compiled Shaders
Each material graph compiles into its own unique shader module.
- **Pros**: Optimal performance for that specific material.
- **Cons**: Massive "Pipeline State Object" (PSO) explosion. Every new material creates a new Vulkan pipeline.

## 3. How Modern Vulkan (Bindless) Changes the Game
Your current work on **Bindless Textures** and **Descriptor Indexing** is a massive advantage for a material system:

1.  **Uniform Resource Access**: Instead of binding textures to slots 0, 1, 2, the shader graph only needs to know an **Index**. 
2.  **Material Data Buffers**: You can store all material parameters (roughness multiplier, albedo tint, texture IDs) in one massive `StructuredBuffer`.
3.  **Speed**: Switching materials becomes a single `uint` push-constant change (the offset into the material buffer) instead of rebinding descriptor sets.

## 4. Composed Material Systems
Composition refers to the ability to build complex materials from smaller, reusable blocks.

### A. Material Functions (Reusable Sub-graphs)
Common logic is encapsulated into "Functions" (e.g., Triplanar Mapping, Water Ripples).
- **Workflow**: One graph can "Call" another graph, effectively inlining logic.

### B. Material Layering (The "Stack" Approach)
Blending multiple PBR materials (e.g., Rusty Iron + Old Wood) using vertex colors or mask textures.
- **Optimization**: Attributes (Albedo, Normal, Roughness) are blended **before** the lighting pass.

### C. Static Switches
Toggling features (e.g., "Dirt Layer") off at compile-time to prune the GLSL code tree.

## 5. Multi-Model Shading (PBR vs. Blinn-Phong)
A compositional system can combine different lighting models by treating the **Shading Model** as material metadata.

- **Mapped Properties**: Albedo maps to Diffuse, Specular (from Metallic) maps to Specular Tint, Roughness maps to Glossiness.
- **Real-time Handling**: 
    - **Forward**: Specialized shaders for each model.
    - **Deferred**: Shading Model ID stored in the G-Buffer for branchy bitwise evaluation in the lighting pass.

## 6. Architectural Shader Organization (The 3-Layer Hierarchy)
1.  **Layer 1: The Library (`/shaders/lib/*.glsl`)**: Static math, PBR, and lighting logic.
2.  **Layer 2: The Templates (`/shaders/templates/*.glsl`)**: Skeletons defining the rendering "Pipe" but leaving holes for material logic.
3.  **Layer 3: Generated Material Code**: Code fragments emitted by the C++ Graph Compiler implementing the `shs_evaluate_material()` hook.

## 7. The Shader Assembler Workflow
The C++ **ShaderAssembler** stitches these layers at runtime:
1.  Inject Header, Extensions, and Resource Layouts.
2.  Include the Core Library.
3.  Inject the **Generated Material Logic**.
4.  Inject the **Pass Template** code.
5.  Call the Vulkan compiler (glslang/shaderc) to produce SPIR-V.

## 9. Implementation Roadmap
The detailed implementation roadmap for this system can be found in the dedicated roadmap documentation:
👉 **[Material System Roadmap](../roadmap/material_system_roadmap.md)**
