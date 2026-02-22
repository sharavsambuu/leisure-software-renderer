# Vulkan Modernization Roadmap

This roadmap outlines the transition of `shs-renderer-lib` from legacy Vulkan 1.0 patterns to a modern, high-performance Vulkan 1.3+ architecture.

## 1. Feature Support Matrix (Current Status)

| Feature | Status | Note |
| :--- | :--- | :--- |
| **Vulkan 1.3 Core** | ✅ | Native support negotiated in `vk_backend.hpp`. |
| **Swapchain** | ✅ Stable | Handles resizes via `vkDeviceWaitIdle` and ring buffers. |
| **Descriptors** | ✅ Bindless | Fully indexed texture arrays (Set 1) with high slot counts. |
| **Sync2** | ✅ Supported | Native/Ext. Preferred for all new barrier and submission logic. |
| **Timeline Semaphores** | ✅ Supported | Native/Ext. Probed and tracked in `capabilities_`. |
| **Dynamic Rendering** | ⚠️ Partial | Backend support 100%; used in `hello_modern_vulkan`. |
| **Ray Query** | ✅ Supported | `VK_KHR_ray_query` enabled; TLAS/BLAS helpers active. |
| **Mesh Shaders** | 🏗️ Probed | `VK_EXT_mesh_shader` detected; drawing helper implemented. |
| **Async Compute** | ⚠️ Shared | Dedicated compute queues detected; support is in progress. |

---

## 2. Modernization Phases

### Phase 1: Sync Modernization (COMPLETE ✅)
- [x] Adopt `VK_KHR_synchronization2` path in backend and demos.
- [x] Add `vkQueueSubmit2` submit path behind capability checks.
- [x] Replace demo-local `vkCmdPipelineBarrier` blocks with `vkCmdPipelineBarrier2` wrappers.
- [x] Centralize image layout transitions into `transition_image_layout_sync2` helper.

### Phase 2: Descriptor Model Upgrade (COMPLETE ✅)
- [x] Introduce descriptor indexing path for texture/material arrays (Bindless).
- [x] Support `UPDATE_AFTER_BIND` for high-frequency updates.
- [x] Integrate bindless texture access in `HelloRenderingPaths` and `HelloPassBasicsVulkan`.

### Phase 3: Queue Model and Async Compute (IN PROGRESS 🏗️)
Goal: overlap compute and graphics safely.
1. Extend backend queue selection to track dedicated compute queue.
2. Split compute (culling/reduction) submission from graphics when async queue exists.
3. Add timeline semaphore graph for inter-queue dependencies.
4. Keep single-queue fallback path as default-safe mode.

### Phase 4: Dynamic Rendering and Pass Graph Cleanup (IN PROGRESS 🏗️)
Goal: simplify render-pass compatibility constraints and ease feature iteration.
- [x] Implement dynamic rendering path (`vkCmdBeginRendering`) in `VulkanRenderBackend`.
- [x] Create `begin_rendering` / `end_rendering` RHI wrappers.
- [x] Validate Dynamic Rendering in modern-focused demos (`hello_modern_vulkan`).
- [ ] Migrate high-level `RenderPathCompiler` to generate `VkRenderingInfo` instead of `VkRenderPass`.
- [ ] Consolidate framebuffer/render-pass object lifetime rules.

### Phase 5: Advanced Rendering Options (IN PROGRESS 🏗️)
- [x] Evaluate and implement ray-query-only shadows/reflections path (`hello_ray_query`).
- [x] Add BLAS/TLAS creation and GPU building wrappers.
- [/] Prototype mesh/task shader path (Probed and helper added, full demo pending).
- [ ] Add robust capability matrix and runtime feature toggles.

---

## 3. Developer Reading Order
1. `shs/rhi/drivers/vulkan/vk_backend.hpp` (Core Lifecycle & Feature Probing)
2. `shs/rhi/drivers/vulkan/vk_cmd_utils.hpp` (Recording Helpers & Barriers)
3. `shs/rhi/drivers/vulkan/vk_render_path_descriptors.hpp` (Bindless Setup)
4. `docs/backlog/optimization_backlog.md` (Performance Tickets)

---

## 4. Test/Verification Checklist
1. Build targets: `HelloPassBasicsVulkan`, `HelloRenderingPaths`, `HelloVulkanTriangle`, `hello_modern_vulkan`.
2. Run with validation layer enabled and capture warnings/errors.
3. Visual A/B against known-good screenshots.
4. Record frame time + CPU submission time before/after.
