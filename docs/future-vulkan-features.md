# Future Vulkan Features

This document tracks major Vulkan features and extensions that are currently missing or underutilized in the `shs-renderer-lib` Vulkan backend, yielding a roadmap for future engine capabilities.

## 1. Async Compute
- **Status:** The backend detects dedicated compute queues (`has_dedicated_compute`), but true asynchronous compute submission is not active. Compute workloads (like light culling) currently execute on the main graphics queue.
- **Goal:** Expose a dedicated compute queue, an independent compute command pool, and cross-queue synchronization (semaphores/events) to parallelize compute and graphics workloads on the GPU.

## 2. Multi-threaded Command Recording
- **Status:** The backend exposes `multithread_command_recording = true` in capabilities, but command buffer generation is mostly single-threaded sequentially per-frame.
- **Goal:** Implement a job system integration where secondary command buffers are recorded in parallel across multiple CPU threads and then executed on the primary command buffer.

## 3. Frames in Flight
- **Status:** The engine currently enforces `kMaxFramesInFlight = 1` intentionally to avoid inter-frame data hazards during rapid prototyping.
- **Goal:** Expand this to 2 or 3 frames in flight. This requires adopting robust per-frame resource rings and dynamic buffer sub-allocation strategies so CPU and GPU can work in tandem.

## 4. Hardware Ray Tracing / Ray Queries
- **Status:** The device initialization probes and selectively enables `VK_KHR_ray_query`, `VK_KHR_acceleration_structure`, and `VK_KHR_deferred_host_operations`.
- **Goal:** Introduce bottom-level (BLAS) and top-level (TLAS) acceleration structure builds from Jolt physics shapes. Write Inline Ray Query shaders (via `ext_ray_query`) to implement ray-traced shadows or ambient occlusion, mixing with the existing rasterizer paths.

## 5. Mesh Shading
- **Status:** No support for `VK_EXT_mesh_shader`.
- **Goal:** Completely bypass the traditional Vertex/Geometry pipeline. Pre-process meshes into meshlets and utilize Task and Mesh shaders for incredibly fast, fine-grained geometry culling (e.g., frustum, occlusion, and sub-pixel culling) on the GPU.

## 6. Graceful Swapchain Recreations
- **Status:** Currently relies on a heavy-handed `vkDeviceWaitIdle` upon window resizes.
- **Goal:** Implement a seamless swapchain recreation flow (using the `oldSwapchain` parameter) to retain in-flight frames and reduce resize stuttering.
