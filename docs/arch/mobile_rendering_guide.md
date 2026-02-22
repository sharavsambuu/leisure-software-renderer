# Mobile Rendering Guide

Optimization reference for TBDR (Tile-Based Deferred Rendering) architectures found on Android and iOS.

## 1. The TBDR Principle
**Rule**: Keep as much data as possible on-chip. Any image read within the same pass (via `input_attachment`) costs zero DRAM bandwidth. **Avoid flushes.**

### Key Strategies
- **Subpass Merging**: Merge Depth-Prepass + GBuffer + Lighting into one `VkRenderPass`.
- **Discard Depth**: Use `STORE_OP_DONT_CARE` for depth if not needed next frame to avoid a DRAM write.
- **Compressed Formats**: Use `R11G11B10` for color and `RGB10A2` (Oct-Encoded) for normals.

## 2. Mobile Feature Checklist
- [ ] Prefer **Clustered Forward** over broad Deferred shading.
- [ ] Pack all shadows into a single **Shadow Atlas**.
- [ ] Use **Half-Precision (FP16)** in fragment shaders.
- [ ] Implement aggressive **LOD bias**.

## 3. Thermal & Power
- Cap FPS to 30/60.
- Implement Dynamic Resolution Scaling.
- Monitor `VK_EXT_memory_budget`.
