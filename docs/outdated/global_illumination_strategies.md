# Global Illumination Strategies for SHS Renderer

Given your existing **multi-pass rendering system** and the move toward **modern Vulkan (Bindless)**, here is a list of Global Illumination (GI) techniques ranked by implementation complexity and performance impact.

## 1. Reflective Shadow Maps (RSM) & Virtual Point Lights (VPLs)
VPLs are a foundational concept in real-time GI. Instead of tracing paths, we "fake" indirect light by placing thousands of tiny lights in the scene.

### What are VPLs?
When a primary light (like the Sun) hits a surface, that surface becomes a "secondary emitter." Each pixel in the **Reflective Shadow Map** represents one of these emitters.
- **Generation**: Render the scene from the light's POV, but instead of just depth, output **Flux** (Color * Intensity) and **World Space Normals**.
- **Distribution**: Every pixel in the resulting textures is effectively a **Virtual Point Light**.

### How RSM Uses Them
In the final shading pass, for every pixel on screen, the shader samples a neighborhood of these VPLs from the RSM.
- **Contribution**: Each VPL contributes its flux to the scene using a standard distance-falloff lighting equation.
- **Benefits**: Very fast for dynamic geometry and lights. Fits perfectly with your bindless/indexed light culling (you can treat VPLs as standard local lights if the count is manageable).

### The Challenge: "Many Lights Problem"
If you have a 512x512 RSM, you have **262,144 VPLs**.
- **Sampling**: You can't sample all of them per pixel. Techniques like **Importance Sampling** or **Interleaved Sampling** are used to pick which VPLs affect which pixel.
- **Artifacts**: Misplaced VPLs or low sample counts can cause "splotchiness." Clamping is often used to prevent "singularities" where a VPL is too close to a surface.

## 2. Screen Space Global Illumination (SSGI)
Extends SSAO concepts to provide indirect diffuse and specular reflections using only screen-space data.
- **How it works**: Samples the G-Buffer (Depth, Normal, Albedo) to trace ray-marched paths for neighboring light contribution.
- **Potential**: Handles complex dynamic objects perfectly, but limited to "what is on screen."
- **Project fit**: Fits perfectly into your post-processing/deferred pipe. Great for local light bounces.

## 3. Light Propagation Volumes (LPV)
A grid-based technique that propagates light through a 3D volume.
- **How it works**: Primary lights inject radiance into a 3D texture (often using Spherical Harmonics). This radiance is then "diffused" through the grid in several compute passes.
- **Potential**: Fast, supports dynamic geometry and lights. Quality is limited by grid resolution (often blocky).
- **Project fit**: Requires Compute shaders and 3D textures. Good match for your mid-tier GI needs.

## 4. Voxel Cone Tracing (VCT)
A more advanced volumetric technique that provides both indirect diffuse and soft specular reflections.
- **How it works**: The scene is "voxelized" into a 3D sparse texture. Shaders then "cone-trace" through this voxel grid. 
- **Potential**: High quality, but very expensive (especially the voxelization step).
- **Project fit**: High implementation complexity. Requires significant GPU memory and robust voxelization logic.

## 5. Radiance Cascades (Modern & Experimental)
A state-of-the-art technique for high-quality, real-time GI at low cost.
- **How it works**: Represents the radiance field using a series of hierarchical "cascades" (spatial and angular). 
- **Potential**: Extremely efficient, resolves "infinite bounces" well, and scales across hardware.
- **Project fit**: This would put your renderer at the cutting edge. Requires Compute shaders and a good understanding of radiance hierarchy.

## 6. SDFGI (Signed Distance Field Global Illumination)
Uses global Signed Distance Fields to trace rays for indirect lighting.
- **How it works**: Combines coarse voxel/SDF data with probes or radiance caching.
- **Potential**: Very stable for large-scale outdoor/indoor transitions (similar to Godot 4).
- **Project fit**: Requires generating and maintaining a global SDF of the scene.

---

## Comparison Matrix

| Technique | Bounce Type | Dynamic Geometry | Memory Cost | implementation Difficulty |
| :--- | :--- | :--- | :--- | :--- |
| **RSM** | 1-Bounce Diffuse | High | Low | Easy |
| **SSGI** | Multi-Bounce (SS) | High | Low | Medium |
| **LPV** | Multi-Bounce | High | Medium | Medium |
| **VCT** | Multi-Bounce + Spec | High | High | Hard |
| **Rad. Cascades**| Multi-Bounce | High | Low/Medium | Hard |
| **SDFGI** | Multi-Bounce | Low (Mostly static) | Medium | Hard |

## Recommendation for SHS
1.  **Short Term**: Implement **RSM** for the sun and **SSGI** for local bounces. This leverages your existing Multi-pass and G-Buffer maturity.
2.  **Long Term**: Explore **Radiance Cascades** or **DDGI (Dynamic Diffuse GI)** as the "Final" solution for high-end fidelity.
