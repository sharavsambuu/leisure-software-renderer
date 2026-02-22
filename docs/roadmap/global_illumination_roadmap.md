# Roadmap: Global Illumination (GI)

This roadmap outlines the implementation strategy for adding real-time dynamic global illumination to the SHS renderer.

## Phase 1: Direct Extensions (The "Quick Wins")
Leverage existing passes to create the first bounce of indirect light.
- **Action: Reflective Shadow Maps (RSM)**
    - Enhance sun shadow pass to output Flux and Normals.
    - Implement a "Many Lights" sampling pass to add 1-bounce sunlight.
- **Action: Enhance SSAO to SSGI**
    - Modify the SSAO compute shader to sample the G-Buffer for colored indirect light.

## Phase 2: Volumetric Stability
Move beyond screen-space and shadow-perspective limits.
- **Action: Light Propagation Volumes (LPV)**
    - Implement light injection into 3D SH (Spherical Harmonic) textures.
    - Implement the compute-based propagation step.
    - Add LPV sampling to the deferred lighting pass.

## Phase 3: High-End Volumetrics & Reflections
Provide both diffuse and specular indirect lighting with high fidelity.
- **Action: Voxelization Pipeline**
    - Implement a real-time scene voxelizer (likely using hardware rasterization into 3D textures).
- **Action: Voxel Cone Tracing (VCT)**
    - Implement cone-tracing logic for indirect diffuse and soft specular.

## Phase 4: Modern Radiance Fields (Future Proofing)
Reach the state-of-the-art for dynamic scenes.
- **Action: Radiance Cascades**
    - Implement hierarchical radiance field storage and sampling.
    - Transition from LPV/VCT to a unified radiance cascade system for superior quality/performance ratio.

---

> [!TIP]
> **Priority Path**: We recommend completing Phase 1 (**RSM + SSGI**) first, as they provide high visual impact with the least architectural friction.
