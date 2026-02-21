# Vulkan Backend Status and Reader Guide

Last updated: 2026-02-21

## 1) Purpose

This note captures the current Vulkan backend status in `shs-renderer-lib`, with a practical map for future humans/LLMs who need to continue backend work.

## 2) Review Scope

Primary files reviewed:

- `cpp-folders/src/shs-renderer-lib/include/shs/rhi/drivers/vulkan/vk_backend.hpp`
- `cpp-folders/src/shs-renderer-lib/include/shs/rhi/backend/backend_factory.hpp`
- `cpp-folders/src/shs-renderer-lib/include/shs/rhi/drivers/vulkan/vk_cmd_utils.hpp`
- `cpp-folders/src/shs-renderer-lib/include/shs/rhi/drivers/vulkan/vk_memory_utils.hpp`
- `cpp-folders/src/shs-renderer-lib/include/shs/rhi/drivers/vulkan/vk_shader_utils.hpp`
- `cpp-folders/src/shs-renderer-lib/include/shs/rhi/drivers/vulkan/vk_swapchain_uploader.hpp`
- `cpp-folders/src/shs-renderer-lib/include/shs/rhi/drivers/vulkan/vk_frame_ownership.hpp`
- `cpp-folders/src/shs-renderer-lib/include/shs/rhi/sync/vk_runtime.hpp`
- `cpp-folders/src/shs-renderer-lib/docs/vulkan-modernization-roadmap.md`

Integration examples checked:

- `cpp-folders/src/exp-plumbing/hello_vulkan_triangle.cpp`
- `cpp-folders/src/exp-plumbing/hello_pass_basics_vulkan.cpp`
- `cpp-folders/src/exp-plumbing/hello_rendering_paths.cpp`
- `cpp-folders/src/exp-plumbing/hello_soft_shadow_culling_vk.cpp`
- `cpp-folders/src/exp-plumbing/hello_culling_vk.cpp`
- `cpp-folders/src/exp-plumbing/hello_occlusion_culling_vk.cpp`

## 3) Current Backend State

### 3.1) Foundation and lifecycle

- Vulkan backend is implemented as a concrete `IRenderBackend` (`VulkanRenderBackend`) in `vk_backend.hpp`.
- Initialization sequence is complete for windowed rendering:
  - instance
  - surface
  - physical device
  - logical device + queues
  - swapchain
  - render pass
  - depth resources
  - framebuffers
  - command pool/buffers
  - sync objects
- Device-loss and surface-out-of-date cases are handled in frame begin/end and swapchain recreation paths.

### 3.2) Frame model

- Backend now runs with `kMaxFramesInFlight = 2`.
- This allows the CPU to parallelize command buffer recording while the GPU executes the previous frame.
- It maximizes GPU utilization compared to the previous single-frame model.
- Frame sync model uses binary semaphores + fences and per-image in-flight fence tracking.

### 3.3) Render path shape

- Main present path is classic render pass + framebuffer based.
- Depth attachment is created if a supported depth format is found (`D32`, `D32_S8`, `D24_S8` fallback chain).
- Swapchain usage enables transfer destination when surface capabilities allow it.

### 3.4) Capability probing and fallback strategy

- Backend probes optional feature bundles:
  - timeline semaphore
  - descriptor indexing (Bindless)
  - dynamic rendering
  - synchronization2
  - ray query bundle (hardware ray tracing)
  - mesh shader extension
- Device creation uses a robust fallback ladder:
  - try all optional bundles (Ray Query + Mesh Shader + Modern Sync/Bindless)
  - retry without mesh/ray bundles
  - fallback to baseline required extensions (Vulkan 1.1)
- `BackendCapabilities` is refreshed with probed features and hardware limits.

### 3.5) Synchronization2 status

- Submit path uses `vkQueueSubmit2`/KHR variant when available and valid.
- Backend exposes `supports_synchronization2()` and `cmd_pipeline_barrier2(...)` helper.
- Legacy submit/barrier path remains as fallback.
- Example demos already consume this helper path in barrier code.

### 3.6) Backend selection behavior

- `create_render_backend(...)` supports software, OpenGL, and Vulkan selection.
- For Vulkan/OpenGL requests, software backend is also registered as an auxiliary fallback for unported passes.
- If Vulkan is requested but not compiled in (`SHS_HAS_VULKAN` not available), backend creation falls back to software with an explanatory note.

## 4) What Is Stable and Worth Reusing

- Robust swapchain lifecycle with resize and out-of-date handling.
- Clean backend contract for begin/end frame and resize integration.
- Capability negotiation with graceful fallback.
- Reusable helper headers for viewport/scissor, buffer allocation, shader module loading.
- Reusable frame-ring utility (`VkFrameRing`) for future multi-frame resource ownership.
- `VulkanLikeRuntime` in `vk_runtime.hpp` provides a backend-agnostic Vulkan-like sync/submission simulation for pipeline testing.

## 5) Current Limits and Technical Debt

- Single frame-in-flight is conservative and limits CPU/GPU overlap.
- Dynamic rendering is detected but not yet used as the main path (`vkCmdBeginRendering` path is not active here).
- Async compute is only reported as a capability signal; backend submission still runs through the graphics/present flow.
- Swapchain recreation uses `vkDeviceWaitIdle`, which is simple but can create visible hitches during resize/rebuild.
- Memory management is manual Vulkan allocation; no allocator layer yet.
- Swapchain uploader currently uses legacy `vkCmdPipelineBarrier` transitions (not sync2-based helper path yet).
- Ray query capability is fully integrated. Demos can build and traverse acceleration structures on the GPU.

## 6) Shader Organization Status (2026-02-16)

### 6.1) Shared Vulkan shader modules added

Common reusable GLSL blocks now exist under:

- `cpp-folders/src/shs-renderer-lib/shaders/vulkan/common/math.glsl`
- `cpp-folders/src/shs-renderer-lib/shaders/vulkan/common/light_constants.glsl`
- `cpp-folders/src/shs-renderer-lib/shaders/vulkan/common/light_math.glsl`
- `cpp-folders/src/shs-renderer-lib/shaders/vulkan/common/culling_light_struct.glsl`

These are intended to reduce drift in light constants, attenuation logic, and core struct contracts.

### 6.2) Refactored shader consumers

Current consumers include:

- `cpp-folders/src/exp-plumbing/shaders/light_types_culling_vk.frag`
- `cpp-folders/src/shs-renderer-lib/shaders/vulkan/fp_stress_scene.frag`
- `cpp-folders/src/shs-renderer-lib/shaders/vulkan/fp_stress_light_cull.comp`
- `cpp-folders/src/shs-renderer-lib/shaders/vulkan/shape_cell_cull.comp`
- `cpp-folders/src/shs-renderer-lib/shaders/vulkan/shape_volume_cull.comp`

### 6.3) Build-system note (important)

`exp-plumbing/CMakeLists.txt` now explicitly tracks shared include files in shader compile `DEPENDS` for affected SPIR-V targets.  
This prevents stale shader binaries when common GLSL modules change.

### 6.4) Remaining shader debt

Still worth modularizing in future:

- shared BRDF utility blocks
- shared shadow sampling/filter helpers
- optional unified local-light evaluation helpers

## 7) Runtime Knobs

- `SHS_VK_PRESENT_MODE=mailbox` requests mailbox present mode when supported.
- Default behavior prefers FIFO for portability/stability.

## 8) Reading Order For Future Contributors

Recommended order:

1. `cpp-folders/src/shs-renderer-lib/include/shs/rhi/drivers/vulkan/vk_backend.hpp`
2. `cpp-folders/src/shs-renderer-lib/include/shs/rhi/backend/backend_factory.hpp`
3. `cpp-folders/src/shs-renderer-lib/include/shs/rhi/drivers/vulkan/vk_swapchain_uploader.hpp`
4. `cpp-folders/src/shs-renderer-lib/include/shs/rhi/drivers/vulkan/vk_memory_utils.hpp`
5. `cpp-folders/src/shs-renderer-lib/include/shs/rhi/drivers/vulkan/vk_cmd_utils.hpp`
6. `cpp-folders/src/shs-renderer-lib/docs/vulkan-modernization-roadmap.md`

Then inspect demo usage patterns:

- `cpp-folders/src/exp-plumbing/hello_pass_basics_vulkan.cpp`
- `cpp-folders/src/exp-plumbing/hello_rendering_paths.cpp`
- `cpp-folders/src/exp-plumbing/hello_soft_shadow_culling_vk.cpp`

## 9) Suggested Next Steps

Priority order for the next backend phase:

1. Move core paths toward 2+ frames in flight by adopting per-frame resource rings consistently.
2. Add an opt-in dynamic rendering path while keeping legacy render pass fallback.
3. Migrate remaining barrier-heavy utilities to sync2 helper paths.
4. Continue shader modularization to cover BRDF + shadow helper reuse.
5. Add clearer perf/debug instrumentation (timestamps, pass markers, GPU timing summary).
6. Introduce a lightweight allocation strategy to reduce manual allocation churn.

Render-path composition alignment (now that light/culling stability is stronger):

7. Introduce a reusable render-path interface contract and wrap current Forward/Forward+ path as baseline.
   - Status: baseline shared preset + runtime selector exists in
     `render_path_presets.hpp` + `render_path_executor.hpp` and is consumed by `HelloRenderingPaths`.
   - Status update: shared resource-plan compiler exists in
     `render_path_resource_plan.hpp`, and `HelloRenderingPaths` consumes recipe-driven tile/cluster sizing.
   - Status update: shared standard pass-contract lookup/registry exists in
     `pass_contract_registry.hpp`, and `HelloRenderingPaths` now compiles recipe plans against it.
   - Status update: shared runtime light-grid allocation layout exists in
     `render_path_runtime_layout.hpp`, and `HelloRenderingPaths` now allocates tile/light-grid buffers from it.
   - Status update: shared Vulkan render-path global descriptor contract exists in
     `vk_render_path_descriptors.hpp`, and `HelloRenderingPaths` now reuses it for set layout/pool/update wiring.
   - Status update: shared pass-chain dispatcher exists in
     `render_path_pass_dispatch.hpp`, and `HelloRenderingPaths` now executes per-frame pass order from active plan.
8. Add a second algorithm path (Deferred + light accumulation) reusing the same scene/light/culling inputs.
9. Add runtime path switching and path-aware telemetry for consistent A/B comparison.
10. Refactor common passes (depth/shadow/light-cull/debug) into reusable path modules.

Related plan:

- `docs/dynamic-render-path-composition-plan.md`
- `docs/render-path-core-draft-and-state.md`
- `docs/modern-rendering-maturity-roadmap.md`
## 10) Major Feature Usage Guide

### 10.1) Hardware Ray Tracing (Ray Query)

To use Ray Query in your demo:
1.  **Request the bundle**: Set `InitDesc::request_ray_bundle = true` during backend initialization.
2.  **Check support**: Verify `vk->capabilities().features.ray_query` is true.
3.  **Create AS**: Use `vk->create_acceleration_structure(...)` for both BLAS and TLAS.
4.  **Shader logic**: Enable `GL_EXT_ray_query` in your GLSL and use `rayQueryEXT` for intersection tests.
5.  **Buffer Usage**: Ensure vertex/index buffers used for AS building have `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT`.

### 10.2) Modern Synchronization (Sync2)

Avoid legacy barriers; use the backend helpers:
- **Layout Transitions**: Use `vk->transition_image_layout_sync2(...)`. This handles `VkImageMemoryBarrier2` and `VkDependencyInfo` internally.
- **Submission**: The backend automatically uses `vkQueueSubmit2` if available.

### 10.3) Bindless / Descriptor Indexing

To use large descriptor arrays:
1.  **Standard Layout**: Use `shs/pipeline/vk_render_path_descriptors.hpp` for a global bindless texture array.
2.  **Update after bind**: The backend enables `DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT` by default in its internal layouts.
3.  **Indexing**: Access textures in shaders via `sampler2D myTextures[]` indexed by push constants or UBO data.

### 10.4) Mesh Shaders

1.  **Check support**: Verify `vk->capabilities().features.mesh_shader` is true.
2.  **Draw calls**: Use `vk->cmd_draw_mesh_tasks(cmd, groupX, groupY, groupZ)` helper to launch mesh/task shader pipelines.

## 11) Software Renderer Counterpart Analysis

While the Vulkan backend targets modern GPU hardware, `shs-renderer-lib` maintains a CPU-based **Software Rasterizer** (`shs::SoftwareRasterizer`) for educational purposes and baseline validation.

### 11.1) Current Software Capabilities
- **Perspective-Correct Rasterization**: Full 1/w interpolation for varyings, UVs, and depth.
- **Parallel Execution**: Uses an `IJobSystem` to scale across multiple CPU cores.
- **Classic Pipeline**: Supports custom C++ Vertex and Fragment shaders (as `std::function`).
- **Standard Features**: Back/Front face culling, Frustum clipping, Depth testing.

### 11.2) Missing Counterparts (Gaps)
| Vulkan Feature | Software Counterpart | Status |
| :--- | :--- | :--- |
| **Hardware Ray Query** | CPU BVH / Ray Tracer | **Missing** (Demos use manual shadow maps) |
| **Compute Shaders** | `parallel_for` lambda | **Lacks unified Pipeline abstraction** |
| **Mesh/Task Shaders** | N/A | **Missing** |
| **Bindless Textures** | Pointer arrays | **Manual implementation only** |
| **Explicit Sync (2)** | Thread primitives | **Implicit only** |

### 11.3) Library Generalness and Cross-Backend Demos
The library is **very general** on the Vulkan side, allowing for standard modern rendering techniques (PBR, Clustered, Ray Tracing). However, there is currently a "fork" in how demos are written:
- **Vulkan Demos**: Use SPIR-V, descriptors, and command buffers.
- **Software Demos**: Use C++ lambdas and direct target-buffer access.

**Recommendation**: To reach "Full Generalness", the library could benefit from a higher-level abstraction that maps common concepts (like "Material Parameters" or "Passes") to both C++ functions and SPIR-V/Descriptors. Currently, developers must essentially write two different rendering paths if they want to support both backends.
