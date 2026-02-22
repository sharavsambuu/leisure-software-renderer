# Modern Rendering Maturity Roadmap

Last updated: 2026-02-22

## 1) Purpose

This roadmap defines how to move from current dynamic render-path composition into production-style modern rendering maturity.

Refer to the high-level architecture guide for implementation details:
- [Render Path Architecture](file:///wsl.localhost/Ubuntu-24.04/home/sharavsambuu/src/dev/leisure-software-renderer/docs/arch/render_path_architecture.md)

Target outcome:
- Reusable and configurable render architecture presets (not demo-specific code paths).
- Stable runtime switching between major rendering paths.
- Clean integration path for modern effects (`SSAO`, `TAA`, `DoF`, `Motion Blur`, `AA`).

## 2) Current Baseline (L4 Maturity)

The project is already strong on composition foundation:
- Shared render-path presets + executor
- Shared path+technique composition presets
- Runtime path/composition controls (`F2`, `F3`, `F4`)
- Temporal core with shared history ownership.

### Recent Progress (Phase I Complete)
- **Phase I**: Achieved contract parity between Vulkan and Software backends.
- **Phase J**: Active focus on Execution Efficiency (Pillars: Sync, Automation, Compute, Culling).

---

## 3) Implementation Phases

### Phase A-D: Core Features & Post-Stack
- **A**: Deferred Baseline.
- **B**: Module Parity (Forward+ and Deferred).
- **C**: Temporal Core (History + TAA).
- **D**: Post-Process Stack (`SSAO`, `Motion Blur`, `DoF`).

### Phase E-I: Robustness & Parity
- **E**: Backend Robustness (Timing + Telemetry).
- **F**: Baseline Metrics & Regressions (JSONL metrics).
- **G**: Soak & Rebuild Reliability (HB heartbeats).
- **H**: Graph-Owned Barriers (Automatic layout transitions).
- **I**: Cross-Backend Parity (Vulkan + Software).

### Phase J: Execution Efficiency (2026 Focus)
- **Sync Pillar**: Timeline Semaphores.
- **Automation Pillar**: Full Render Graph.
- **Compute Pillar**: Shift everything to Compute Kernels.
- **Culling Pillar**: GPU-driven / Meshlet culling.

---

## 4) Definition of "Modern Maturity Reached"
- Major architectures (`Forward+` and `Deferred`) are fully recipe-driven.
- Temporal and Post effects are recipe-composed.
- Demo hosts no longer own architecture-specific pass sequencing.
- Path swaps are validated and benchmarkable with shared telemetry.
