# Experimental Rendering Techniques

The `shs-renderer-lib` was built to construct and evaluate various rendering paradigms by combining diverse 3D light volumes with advanced spatial light culling.

To benchmark, profile, and validate these features, we are implementing a dedicated suite of demos in the `exp-rendering-techniques` folder. These demos will use a standardized scene containing varying culling complexities (rotating geometry against a static floor, filled with densely packed point/sphere lights) to accurately measure performance and correctness.

## Evaluated Rendering Techniques

The following is a comprehensive list of rendering and shading techniques to be implemented and evaluated as separate demos:

1. **Classic Forward Rendering (`demo_forward_classic`)**
   - **Concept**: Iterates over all active lights in the scene per-fragment. Opaque geometry is rendered single-pass.
   - **Purpose**: Establishes a baseline performance and visual benchmark. Used to demonstrate the O(Geometry * Lights) shading failure point.

2. **Forward with Z-Prepass (`demo_forward_zprepass`)**
   - **Concept**: Renders all opaque geometry to a depth buffer first. A second pass renders geometry shading using depth-testing `EQUAL`.
   - **Purpose**: Eliminates shading overdraw. Useful for comparing vertex-bound overhead vs fragment-bound overhead in dense geometry scenes.

3. **Classic Deferred Rendering (`demo_deferred_classic`)**
   - **Concept**: Renders geometry attributes (Albedo, Normal, Material, Depth) to a thick G-Buffer. Lighting is accumulated in a secondary full-screen pass over the G-Buffer by rendering light volumes.
   - **Purpose**: Demonstrates perfect decoupling of geometry and lighting. Baseline for measuring high memory bandwidth cost.

4. **Tiled Forward Rendering / Forward+ (`demo_forward_tiled`)**
   - **Concept**: A compute/setup pass divides the screen into a 2D grid of 16x16 or 32x32 tiles. Lights are frustrated culled into per-tile lists. The forward pass iterates only on lights within the current screen tile.
   - **Purpose**: Showcases effective handling of massive light counts while retaining hardware MSAA and transparent object support.

5. **Tiled Deferred Rendering (`demo_deferred_tiled`)**
   - **Concept**: Instead of rendering light geometry to evaluate lighting on the G-Buffer, a compute shader processes the G-Buffer using Tiled / Depth-Bounds (2.5D) cell grids, dispatching tailored light loops.
   - **Purpose**: Highly efficient deferred lighting strategy minimizing wasted fragment shading and overdraw on light volumes. Also addresses depth discontinuity in standard 2D tiled setups.

6. **Clustered Forward Rendering (`demo_forward_clustered`)**
   - **Concept**: Slices the camera frustum into a 3D volumetric grid (Clusters). Lights are bucketed into these 3D cells. Fragments query their precise 3D space cluster and iterate over a small, highly accurate light list.
   - **Purpose**: The industry standard for massive dynamic lighting. Seamlessly handles variable depth discontinuities and dense transparency cleanly. 

7. **Visibility Buffer / Deferred Texturing (`demo_visibility_buffer`)** *(Aspirational/Future)*
   - **Concept**: Renders only triangle IDs and Instance IDs to a minimalist buffer. Compute shaders rebuild vertex/material data dynamically on shading execution.
   - **Purpose**: Evaluates absolute minimal rasterization memory bandwidth. Overcomes geometry culling bottlenecks present in traditional deferred setups. 

## Secondary / Post-Processing Passes

To evaluate realistic production workloads, these technique demos will also incorporate industry-standard secondary passes:

*   **SSAO (Screen Space Ambient Occlusion)**: Evaluated directly from Depth/Normals.
*   **Anti-Aliasing**: Generic post-process AA (FXAA) or Temporal AA (TAA) depending on motion vector availability.

---

*This document was generated as part of the `exp-rendering-techniques` phase to guide the rollout of discrete rendering paradigms.*
