# Vulkan Backend Status and Reader Guide

Last updated: 2026-02-16

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
- `cpp-folders/src/exp-plumbing/hello_forward_plus_stress_vulkan.cpp`
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

- Backend currently runs with `kMaxFramesInFlight = 1`.
- This is intentional and documented in code to avoid inter-frame hazards while demos still update many resources in-place.
- Frame sync model uses binary semaphores + fences and per-image in-flight fence tracking.

### 3.3) Render path shape

- Main present path is classic render pass + framebuffer based.
- Depth attachment is created if a supported depth format is found (`D32`, `D32_S8`, `D24_S8` fallback chain).
- Swapchain usage enables transfer destination when surface capabilities allow it.

### 3.4) Capability probing and fallback strategy

- Backend probes optional feature bundles:
  - timeline semaphore
  - descriptor indexing
  - dynamic rendering
  - synchronization2
  - ray query bundle
- Device creation uses a safe fallback ladder:
  - try all optional bundles
  - retry without ray bundle
  - fallback to baseline required extensions
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
- Ray query capability is probed and surfaced, but no full ray-query rendering workflow is integrated in this backend layer.

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
- `cpp-folders/src/exp-plumbing/hello_forward_plus_stress_vulkan.cpp`
- `cpp-folders/src/exp-plumbing/hello_soft_shadow_culling_vk.cpp`

## 9) Suggested Next Steps

Priority order for the next backend phase:

1. Move core paths toward 2+ frames in flight by adopting per-frame resource rings consistently.
2. Add an opt-in dynamic rendering path while keeping legacy render pass fallback.
3. Migrate remaining barrier-heavy utilities to sync2 helper paths.
4. Continue shader modularization to cover BRDF + shadow helper reuse.
5. Add clearer perf/debug instrumentation (timestamps, pass markers, GPU timing summary).
6. Introduce a lightweight allocation strategy to reduce manual allocation churn.
