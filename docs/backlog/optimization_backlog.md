# Optimization & Culling Backlog

This document tracks the active backlog of performance and quality improvements for `shs-renderer-lib`. It combines strategic roadmaps with granular implementation tasks.

---

## ⚡ Culling Status & Stable Defaults

The Jolt-backed culling system is functionally complete. To avoid visual popping and query-related instability, use these **Stable Defaults**:

*   **View Occlusion**: ON (Software Depth or Query-based)
*   **Shadow Occlusion**: OFF (Avoids shadow-map flickering)
*   **Shadow Frustum Culling**: ON
*   **View Frustum Culling**: ON (Always)

---

## 🚀 The Modern Capability Roadmap (Strategic Focus)

1.  **Sync Pillar (#11)**: Timeline Semaphores for fine-grained CPU-GPU overlap.
2.  **Automation Pillar (#17)**: Render Graph for automatic barriers and resource aliasing.
3.  **Compute Pillar (#14, #24, #26)**: Shift culling, lighting, and post-processing to Compute Shaders.
4.  **Culling Pillar (#24)**: GPU-driven culling and Meshlet pipelines.

---

## 🏎️ Parallelization Quick Wins

| # | Task | Location | Effort |
| :--- | :--- | :--- | :--- |
| 1 | **Wire MT Command Recording** | `hello_soft_shadow_culling_vk` | ~30 lines |
| 2 | **Parallel Light Motion** | `update_light_motion()` | ~10 lines |
| 3 | **Parallel Batch Culling** | `jolt_culling.hpp` loops | ~15 lines |
| 4 | **Parallel Instance UBO Upload** | Render path UBO loops | ~10 lines |
| 5 | **Parallel Light-Object Filter** | `collect_object_lights()` | ~15 lines |

---

## 🛠️ Detailed Backlog Tasks

### 1-7: CPU Parallelism (via `shs::parallel_for_1d`)
- **Task #1**: Parallelize batch frustum/cell classification in `jolt_culling.hpp`.
- **Task #2**: Tile-based Software Occlusion Rasterization in `culling_software.hpp`.
- **Task #4**: Parallelize per-object light list collection (Light-Object Pre-filter).
- **Task #6**: Parallelize Light Bin/Cluster assignment in `light_culling_runtime.hpp`.

### 8-15: Modern RHI / Job System Extensions
- **Task #11**: Timeline Semaphores in `vk_backend.hpp` for Async Compute sync.
- **Task #12**: Work-Stealing Deque for the ThreadPool.
- **Task #14**: Lightweight Task Graph for stage-level parallelism (Cull -> Record -> Submit).

### 16-24: Mobile & Compact Path Optimizations
- **Task #18**: Vulkan Subpass Merging to keep G-Buffer data on-tile (TBDR).
- **Task #20**: Shadow Atlas implementation to reduce render pass overhead.
- **Task #21**: Half-Precision G-Buffer formats (R11G11B10 / RGB10A2).
- **Task #23**: Object-level LOD selection based on screen size.

### 25-26: Advanced GPU Features
- **Task #25**: GPU-Driven Pipeline (Mesh Shaders + Indirect Draw).
- **Task #26**: Compute-based Post-Processing (Tiled Bloom, Unified Post Kernel).

---

## 📚 Reference Prototypes
Foundational experiments in `src/hello-parallelization/`:
- `hello_job_system_graph.cpp` (Node-based dependencies)
- `hello_thread_shader_job.cpp` (Software parallelism)
- `hello_xsimd_threads.cpp` (SIMD + MT)
