# Future Vulkan Features

This document tracks major Vulkan features that are currently missing or underutilized in the `shs-renderer-lib` backend.

## 1. Async Compute
- **Status:** Backend detects queues, but workloads (Light Culling) still run on the Graphics queue.
- **Goal**: Form the **Compute Pillar** using dedicated compute queues and cross-queue sync (Timeline Semaphores).

## 2. Multi-threaded Command Recording
- **Status**: Infrastructure (`WorkerPool`) exists in demos but isn't fully wired into the library core.
- **Goal**: Implement job system integration where secondary command buffers are recorded in parallel.

## 3. Frames in Flight
- **Status**: Currently `kMaxFramesInFlight = 2`.
- **Goal**: Expand to 3 where beneficial, requiring robust resource ring management.

## 4. Hardware Ray Tracing / Ray Queries
- **Status**: `VK_KHR_ray_query` enabled; probing active.
- **Goal**: Build TLAS/BLAS from Jolt shapes; integrate inline ray queries for shadows and AO.

## 5. Mesh Shading
- **Status**: No implementation of `VK_EXT_mesh_shader`.
- **Goal**: Form the **Culling Pillar** by porting geometry to meshlets and utilizing Task/Mesh shaders.

## 6. Swapchain Recreations
- **Status**: Uses `vkDeviceWaitIdle` on resize.
- **Goal**: Implement seamless recreation using `oldSwapchain` to minimize stutter.
