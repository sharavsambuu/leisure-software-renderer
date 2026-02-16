# Vulkan Modernization Roadmap

This roadmap targets incremental upgrades from the current Vulkan 1.1-style core path toward modern Vulkan capabilities while keeping existing demos stable.

## Baseline (current)

- Core render-pass pipeline + explicit barriers works (`hello_pass_basics_vulkan`, `hello_rendering_paths`).
- Compute passes exist for light culling/depth reduce.
- Secondary command buffer recording exists for stress demo.
- Backend now probes and exposes optional feature availability:
  - timeline semaphores
  - descriptor indexing
  - dynamic rendering
  - synchronization2 extension support
  - ray query capability signal
- Phase-1 bootstrap landed:
  - backend can submit via `vkQueueSubmit2` when Sync2 is available (with legacy fallback)
  - `HelloPassBasicsVulkan` barriers use Sync2 path when available (with legacy fallback)
  - `HelloRenderingPaths` memory barriers use Sync2 path when available (with legacy fallback)

## Phase 1: Sync modernization (high priority)

Goal: remove brittle stage-mask/barrier assumptions and make frame orchestration deterministic.

1. Adopt `VK_KHR_synchronization2` path in backend and demos when available.
2. Add `vkQueueSubmit2` submit path behind capability checks, fallback to legacy `vkQueueSubmit`.
3. Replace demo-local `vkCmdPipelineBarrier` blocks with `vkCmdPipelineBarrier2` wrappers.
4. Centralize image layout transitions into small helpers with per-resource state tracking.

Success criteria:
- No regression in `HelloPassBasicsVulkan` and `HelloRenderingPaths`.
- Validation layer clean in normal runtime path.

## Phase 2: Descriptor model upgrade (high priority)

Goal: reduce descriptor churn and prepare for scalable light/material data.

1. Introduce descriptor indexing path for texture/material arrays.
2. Replace per-object descriptor updates with ringed/pooled descriptor strategy.
3. Add optional bindless material/texture table for stress demo.
4. Keep fallback path for GPUs lacking descriptor indexing.

Success criteria:
- Lower CPU time in descriptor update hot paths.
- No visual mismatch between legacy and indexed paths.

## Phase 3: Queue model and async compute (medium priority)

Goal: overlap compute and graphics safely.

1. Extend backend queue selection to track dedicated compute queue.
2. Split compute (culling/reduction) submission from graphics when async queue exists.
3. Add timeline semaphore graph for inter-queue dependencies.
4. Keep single-queue fallback path as default-safe mode.

Success criteria:
- Stable frame pacing and no queue hazard validation errors.
- Measurable GPU overlap on hardware with dedicated compute queue.

## Phase 4: Dynamic rendering and pass graph cleanup (medium priority)

Goal: simplify render-pass compatibility constraints and ease feature iteration.

1. Add dynamic rendering path (`vkCmdBeginRendering`) when supported.
2. Incrementally port post-process passes first, then scene/shadow passes.
3. Consolidate framebuffer/render-pass object lifetime rules.
4. Keep legacy render-pass path for portability/testing.

Success criteria:
- Reduced pipeline/render-pass coupling complexity.
- Same output between dynamic and legacy modes.

## Phase 5: Advanced rendering options (optional, long-term)

Goal: selectively leverage modern hardware features for quality/perf.

1. Evaluate ray-query-only shadows/reflections path (not full RT pipeline first).
2. Prototype mesh/task shader path only after Phase 1-4 stabilization.
3. Add robust capability matrix and runtime feature toggles.

Success criteria:
- Optional features degrade gracefully with clear fallbacks.
- No hard dependency on advanced extensions for baseline demos.

## Test/verification checklist per phase

1. Build targets:
   - `HelloPassBasicsVulkan`
   - `HelloRenderingPaths`
   - `HelloVulkanTriangle`
2. Run with validation layer enabled and capture warnings/errors.
3. Visual A/B against known-good screenshots.
4. Record frame time + CPU submission time before/after.
