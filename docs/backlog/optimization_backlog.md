# Optimization & Culling Backlog

> Status: **active (2026-09-18)** — live strategic/implementation backlog for
> `shs-renderer-lib` performance & quality work. Nature: schedule, not law —
> items land per the law-budget norm (Constitution II §2.2(5): each named
> gate met before ticking). Loud-header banner added during the 2026-09-18
> docs sweep; the file previously shipped without a status banner. The
> rasterizer-specific hot-path track (R1–R5) is closed and archived at
> [`docs/outdated/rasterizer_hot_path_todo.md`](../outdated/rasterizer_hot_path_todo.md).
>
> **Tracking (2026-09-18).** The tasks below are now `- [ ]` checkboxes so they
> participate in completion counts like every other backlog. **Tick rule**
> (law-budget norm, Constitution II §2.2(5)): an item closes only with (a) a
> named benchmark or measurement showing it pays for itself — this register is
> demand/measure-gated, never speculative — and (b) full `build/` CTest +
> `check_kdba_boundaries.sh` + `check_include_graph.py` green in the same
> commit. Numbers are the original tracker IDs and are stable (Rule N5); no
> item is renumbered here.

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

Cheap, localized wins — expected effort in parentheses.

- [ ] **QW1** Wire MT command recording — `hello_soft_shadow_culling_vk` (~30 lines).
- [ ] **QW2** Parallel light motion — `update_light_motion()` (~10 lines).
- [ ] **QW3** Parallel batch culling — `jolt_culling.hpp` loops (~15 lines). *Same work as #1 below — do not double-count.*
- [ ] **QW4** Parallel instance UBO upload — render-path UBO loops (~10 lines).
- [ ] **QW5** Parallel light-object filter — `collect_object_lights()` (~15 lines). *Same work as #4 below — do not double-count.*

---

## 🛠️ Detailed Backlog Tasks

### 1-7: CPU Parallelism (via `shs::parallel_for_1d`)

- [ ] **#1** Parallelize batch frustum/cell classification — `jolt_culling.hpp`.
- [ ] **#2** Tile-based software occlusion rasterization — `culling_software.hpp`.
- [ ] **#4** Parallelize per-object light list collection (light-object pre-filter).
- [ ] **#6** Parallelize light bin/cluster assignment — `light_culling_runtime.hpp`.

### 8-15: Modern RHI / Job System Extensions

- [ ] **#11** Timeline semaphores in `vk_backend.hpp` for async-compute sync.
- [ ] **#12** Work-stealing deque for the thread pool.
- [ ] **#14** Lightweight task graph for stage-level parallelism (cull → record → submit).

### 16-24: Mobile & Compact Path Optimizations

- [ ] **#18** Vulkan subpass merging to keep G-buffer data on-tile (TBDR).
- [ ] **#20** Shadow atlas to reduce render-pass overhead.
- [ ] **#21** Half-precision G-buffer formats (R11G11B10 / RGB10A2).
- [ ] **#23** Object-level LOD selection based on screen size.

### 25-26: Advanced GPU Features

- [ ] **#25** GPU-driven pipeline (mesh shaders + indirect draw).
- [ ] **#26** Compute-based post-processing (tiled bloom, unified post kernel).

**Open count (2026-09-18):** 18 checkboxes total — QW1–QW5, plus #1, #2, #4,
#6, #11, #12, #14, #18, #20, #21, #23, #25, #26 — of which QW3≡#1 and QW5≡#4,
so **16 distinct tasks**. The section ranges (1–7, 8–15, 16–24, 25–26) come from
the original external tracker numbering; numbers not itemized above have no
recorded task text in this repository and are deliberately neither counted nor
invented. Strategic pillars (sync / automation / compute / culling) are realized
by the items in this list, not by separate tasks.

---

## 📚 Reference Prototypes
Foundational experiments in `src/hello-parallelization/`:
- `hello_job_system_graph.cpp` (Node-based dependencies)
- `hello_thread_shader_job.cpp` (Software parallelism)
- `hello_xsimd_threads.cpp` (SIMD + MT)
