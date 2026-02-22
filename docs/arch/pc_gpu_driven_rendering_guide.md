# PC & High-End GPU Guide

Optimization reference for discrete PC GPUs (RTX, RDNA, Arc) where CPU overhead and wave occupancy dominate performance.

## 1. GPU-Driven Pipeline
The ultimate goal is zero CPU per-draw overhead:
- **Indirect Drawing**: Use `vkCmdDrawIndexedIndirectCount` to dispatch all objects in one call.
- **GPU Culling**: Cull instances in a Compute Shader; update the count buffer on GPU.
- **Mesh Shaders**: Phase out Vertex/Geometry shaders for Meshlet-based culling.

## 2. Visibility Buffer
For maximum throughput, store only `(TriangleID, InstanceID)` in a 64-bit buffer, then shade in a fullscreen compute pass.

## 3. Bindless Resources
Eliminate descriptor churn by binding one global set of all texture and mesh pointers. Requires `VK_EXT_descriptor_indexing`.

## 4. Async Compute
Overlap Post-Process and Shadow/Cluster generation on separate compute queues.

## 5. PC Implementation Priority
1. **Render Graph** (Task #17) - Automatic barriers.
2. **Bindless Arrays** - Global resource access.
3. **GPU Culling** - Move `parallel_for` culling to GPU.
