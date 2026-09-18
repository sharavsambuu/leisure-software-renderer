# Comprehensive Technical & Architectural Review: `shs-renderer-lib`

> **Document Status**: Official Architectural Review  
> **Target Subsystem**: [`cpp-folders/src/shs-renderer-lib`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib)  
> **Review Baseline**: C++23, Kleisli Domain Boundary Architecture (KDBA), Contract Guardrails (Rule 17)  
> **Author**: Antigravity AI (Google DeepMind)  
> **Review Date**: 2026-09-18  

---

## 1. Executive Summary

The `shs-renderer-lib` library serves as the core engine and rendering foundation for the `leisure-software-renderer` project. It represents a mature and mathematically disciplined convergence of three paradigms:
1. **Data-Oriented Design (DOD / Constitution III)**: Contiguous memory layouts, Structure-of-Arrays (`SoaTable`), $\mathcal{O}(1)$ transient frame arenas (`FrameMemoryResource`), and generational handles over raw pointers.
2. **Domain-Driven Design via Domain Value Objects (DDD / DVO Backbone, Constitution II §2.3)**: Strict bounded contexts where plain, identity-free, always-valid data structures undergo state transitions exclusively through owning gateways.
3. **Functional Programming (Kleisli Domain Boundary Architecture / KDBA)**: Atomic transitions modeled as Kleisli arrows (`A -> std::expected<B, Error>`), eliminating monolithic `switch-case` mutation blocks and ensuring zero-signal-loss event reporting.

### Key Metrics & Verification Status
* **Test Suite**: **43 / 43 tests passing** (100% pass rate in CTest, 78.69s execution time).
* **Boundary Integrity**: 100% green across all 9 boundary and architecture gates in [`check_kdba_boundaries.sh`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/tools/check_kdba_boundaries.sh).
* **Header Self-Containment**: **231 / 231 public headers** compile standalone with zero external macro injection and zero build-tree leaks ([`header_self_containment_test.sh`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/tests/header_self_containment_test.sh)).
* **Dual-Backend Parity**: Validated bounded parity between CPU software rasterization and GPU Vulkan offscreen rendering ([`sw_vk_parity_tests.cpp`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/tests/sw_vk_parity_tests.cpp)).

### Overall Assessment
| Dimension | Rating | Summary |
| :--- | :---: | :--- |
| **Architectural Rigor** | **Exceptional (9.5/10)** | Uncompromising adherence to KDBA, Core 4 per DVO, and pure/edge isolation. |
| **Safety & Contracts** | **Exceptional (9.5/10)** | Strict contract placement (Gate 8), railway error handling (Gate 9), and replay parity. |
| **Code Hygiene & Style** | **Excellent (9.0/10)** | NASA/JPL vertical column alignment, strong header inventory, zero-leak IWYU DAG. |
| **Rendering Architecture** | **Excellent (8.8/10)** | Declarative compile-then-execute pipeline with capability gating and hot-swap resilience. |
| **Software Rasterizer Hot-Path** | **Moderate (6.5/10)** | Significant performance bottlenecks: per-triangle heap allocations, `std::function` inside pixel loops, and per-triangle job sync barriers. |

---

## 2. Constitutional & Specification Compliance

### 2.1 Constitution I — Conventions & Spatial Laws ([`docs/spec/conventions.md`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/docs/spec/conventions.md))

* **Coordinate System**: Render space is strictly **Left-Handed (LH)** with **+Y up** and **+Z forward** ([`shs/camera/convention.hpp`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs/camera/convention.hpp#L21-L37)). NDC Z-range is OpenGL-style `[-1.0, 1.0]` using `glm::perspectiveLH_NO`.
* **Screen vs. Canvas Space**: Screen space (top-left origin) and `shs::Canvas` (bottom-left origin) coordinate differences are properly isolated. Rasterizer presentation correctly converts integer coordinates via `row_canvas = (H - 1) - row_screen`.
* **Physics Seam & Jolt Bridge**: Jolt uses a Right-Handed (+Y up, -Z forward) system. The bridge in [`shs/geometry/adapters/jolt/jolt_adapter.hpp`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs/geometry/adapters/jolt/jolt_adapter.hpp) correctly implements:
  - Vector reflection: $(x, y, -z)$
  - Matrix conjugation: $M_{jolt} = S \cdot M_{shs} \cdot S$ where $S = \text{diag}(1, 1, -1, 1)$
  - Involution quaternion conjugation: $(-x, -y, z, w)$
* **Lighting Semantics**: `sun_dir_to_scene_ws` is universally treated as the vector pointing **from the sun toward the scene**, preventing direction inversions across shading passes.
* **NASA/JPL Vertical Alignment Style**: Structural tables, assignment blocks, and member listings in [`include/shs/`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs) follow strict column alignment (e.g. [`renderpath.gateway.hpp`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs/renderpath/renderpath.gateway.hpp#L90-L96)), drastically enhancing human scanning speed.
* **Pluggability / No User Lock-In (§7)**: In [`shs/app/backend/backend_factory.hpp`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs/app/backend/backend_factory.hpp#L84-L95), when Vulkan is requested on a GPU-less or headless build, the factory gracefully falls back to the software backend with an explicit diagnostic note instead of crashing or aborting CMake configuration.

### 2.2 Constitution II — KDBA & Domain Value Objects ([`docs/spec/value_oriented_programming.md`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/docs/spec/value_oriented_programming.md))

* **Core 4 Structural Law (§6.1, §6.2)**: Every domain in `shs-renderer-lib` defines its mandatory 4 components in separate files:
  1. `*.contract.hpp`: Plain DVO schemas, zero mutation logic.
  2. `*.command.hpp`: Closed intent vocabulary (`<Pod>Command = std::variant<...Intent>`).
  3. `*.event.hpp`: Closed discrete fact vocabulary (`<Pod>Event = std::variant<...Event>`).
  4. `*.gateway.hpp`: Pure Kleisli state transition entry points.
  - Slices with intentionally empty command/event vocabularies (`geometry`, `lighting`, `sky`, `scene`, `resources`, `gfx`, `frame`) explicitly define `std::variant<std::monostate>` and lock it with compile-time assertions (`static_assert(std::variant_size_v<X> == 1)`).
* **Pure Value Center vs. Execution Edge (§2, §3 Rule 2)**: All planners and gateways are free of globals, singletons, and dynamic allocation. Side effects (GPU command buffer recording, swapchain presentation, OS events) are strictly confined to execution edges (`renderpath/execution/`, `rhi/vulkan/runtime/`, `platform/`).
* **Contract Guardrails (Rule 17, Gate 8)**: Guardrail placement strictly follows the single-source law ([`check_contract_placement.sh`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/tools/check_contract_placement.sh)):
  - `SHS_CONTRACT_ASSERT`: Used exclusively for value invariants in `*.contract.hpp` and allowlisted pure leaves (such as compiler transition tables in [`render_path_compiler.hpp:L124`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs/renderpath/planning/render_path_compiler.hpp#L124)).
  - `SHS_PRE` / `SHS_POST`: Used exclusively at gateway boundaries (e.g. non-aliasing preconditions in [`renderpath.gateway.hpp:L364`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs/renderpath/renderpath.gateway.hpp#L364) and committed plan validity postconditions in [`renderpath.gateway.hpp:L199`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs/renderpath/renderpath.gateway.hpp#L199)).
* **Gateway Rails (Gate 9, [`check_gateway_rails.sh`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/tools/check_gateway_rails.sh))**:
  - Zero `throw` statements on gateway rims.
  - Every `std::visit` over a closed intent variant ends with a compile-time exhaustiveness assertion (`static_assert(sizeof(T) == 0, ...)`), preventing silent `default:` drops.
  - Gateways return explicit value step records (`RenderPathStep`, `InputStep`, `FrameStep`) with zero-signal-loss accounting.

### 2.3 Constitution III — Data Layout & Execution ([`docs/spec/dod_ecs_architecture.md`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/docs/spec/dod_ecs_architecture.md))

* **Transient Frame Bump Allocator**: [`shs/memory/frame_memory_resource.hpp`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs/memory/frame_memory_resource.hpp) implements `FrameMemoryResource` with $\mathcal{O}(1)$ reset, real buffer pointer alignment, high-water tracking, and strict `bad_alloc` on overflow (rejecting illegal fallback into persistent heap).
* **SoA Contiguous Storage**: [`shs/containers/soa_table.hpp`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs/containers/soa_table.hpp) implements `SoaTable<Ts...>`:
  - 64-byte alignment per column base (cache-line optimized).
  - Generational uint32 handle (`SoaHandle`) preventing ABA issues.
  - Swap-and-pop removal maintaining packed array density.
* **Elimination of Node Containers**: Cold registries (`ResourceRegistry` in [`resources/storage/resource_registry.hpp`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs/resources/storage/resource_registry.hpp) and `RTRegistry` in [`render/targets/storage/rt_registry.hpp`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs/render/targets/storage/rt_registry.hpp)) have been successfully migrated to open-addressing / flat contiguous lookups via [`shs/containers/flat_map.hpp`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs/containers/flat_map.hpp).

---

## 3. Deep Subsystem Analysis

### 3.1 Render Path & Pipeline Planning (`shs/renderpath/`)

#### Architectural Flow
```
┌──────────────────┐      ┌─────────────────────────┐      ┌──────────────────────────┐
│ RenderPathRecipe │ ───► │   RenderPathCompiler    │ ───► │ RenderPathExecutionPlan  │
│ (Declared Intent)│      │  (Capabilities + Rules) │      │ (Resolved Pass Sequence) │
└──────────────────┘      └─────────────────────────┘      └────────────┬─────────────┘
                                                                        │
                                   ┌────────────────────────────────────┴───────────────────────────────────┐
                                   ▼                                                                        ▼
                    ┌──────────────────────────────┐                                         ┌─────────────────────────────┐
                    │    RenderPathResourcePlan    │                                         │    RenderPathBarrierPlan    │
                    │  (Semantic Binding Resolution│                                         │ (Image Layout Transitions)  │
                    └──────────────┬───────────────┘                                         └──────────────┬──────────────┘
                                   │                                                                        │
                                   └────────────────────────────────────┬───────────────────────────────────┘
                                                                        ▼
                                                       ┌──────────────────────────────────┐
                                                       │      PipelineRuntimeExecutor     │
                                                       │   (Dispatches execute_resolved)  │
                                                       └──────────────────────────────────┘
```

#### Strengths
1. **Dynamic Hot-Swapping with Atomic Rollback**: The gateway [`renderpath_gateway`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs/renderpath/renderpath.gateway.hpp#L352-L403) cleanly decouples intent submission from compilation. If a candidate recipe is incompatible with hardware capabilities (e.g. requesting occlusion culling without depth attachments), compilation returns `std::unexpected(reason)`, the gateway retains the existing execution plan, bumps no generation, and logs a `PathSwapRejectedEvent`.
2. **Capability Decoupling**: Planners query [`RenderPathCapabilitySet`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs/renderpath/planning/render_path_capabilities.hpp), which is a plain data snapshot. No driver initialization or live Vulkan device is needed during planning.
3. **Semantic Resource Mapping**: Passes request resources by semantic role (`Albedo`, `Depth`, `LightGrid`, `Normal`) rather than hardcoded target IDs, enabling flexible G-buffer layout restructuring.

#### Deficiencies & Technical Debt
* **Closed `PassId` Enumeration**: [`PassId`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs/renderpath/planning/pass_contract.hpp) is limited to a closed enum of 16 built-in passes. For third-party or demo-authored passes to be first-class without core library modifications (Constitution I §7 Pluggability Law), `PassId` must transition to open 64-bit string-hash IDs or a dynamic pass registry.

---

### 3.2 Software Rasterizer (`shs/render/software/rasterizer.hpp`)

The software rasterizer in [`rasterizer.hpp`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs/render/software/rasterizer.hpp) provides an exact mathematical reference implementation for CPU-based rendering. However, it exhibits several critical performance and design flaws that directly conflict with the project's performance laws.

#### 1. Per-Triangle Heap Allocations (Severe Violation of Constitution II §4.3)
In [`rasterizer.hpp:L245`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs/render/software/rasterizer.hpp#L245):
```cpp
std::vector<detail::RasterVertex> poly = {
    rv0, rv1, rv2
};
if (!(fully_inside_clip(rv0) && fully_inside_clip(rv1) && fully_inside_clip(rv2)))
{
    poly = detail::clip_polygon_frustum(poly);
}
```
Furthermore, inside [`detail::clip_polygon_frustum`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs/render/software/rasterizer.hpp#L157-L167), clipping against the 6 frustum planes calls [`clip_polygon_plane`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs/render/software/rasterizer.hpp#L115) 6 consecutive times, where each invocation constructs a brand new `std::vector<RasterVertex> out;` on the global heap!
* **Impact**: A mesh with 10,000 triangles invokes `malloc`/`free` **at least 10,000 to 70,000 times per frame** inside the hot render loop.
* **Remediation**: Sutherland-Hodgman polygon clipping on a triangle against 6 half-spaces produces a convex polygon with at most $3 + 6 = 9$ vertices. Using a double-buffered fixed-size array (`std::array<RasterVertex, 16>` or a PMR scratch buffer) completely eliminates all per-triangle dynamic heap allocations.

#### 2. Fragment Shader Invocation via `std::function`
In [`shs/render/shader/program.hpp:L23`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs/render/shader/program.hpp#L23) and [`rasterizer.hpp:L419`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs/render/software/rasterizer.hpp#L419):
```cpp
using FragmentShaderFn = std::function<FragmentOut(const FragmentIn&, const ShaderUniforms&)>;
...
const FragmentOut fout = program.fs(fin, uniforms);
```
* **Impact**: Invoking `std::function::operator()` for every covered pixel incurs virtual-table-like indirect branch dispatch, prevents compiler inlining, and prohibits SIMD vectorization across pixel chunks. At 1080p, this amounts to millions of indirect function calls per frame.
* **Remediation**: Parameterize `rasterize_mesh` on the shader functor type (`template <typename Program>` or function pointer with direct inlining), or provide a specialized batch fragment shader kernel that processes $2 \times 2$ pixel quads.

#### 3. Inner-Loop Barycentric Recalculation
In [`rasterizer.hpp:L340`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs/render/software/rasterizer.hpp#L340):
```cpp
for (int y = yb; y < ye; ++y) {
    for (int x = minx; x <= maxx; ++x) {
        const glm::vec2 p{(float)x + 0.5f, (float)y + 0.5f};
        const glm::vec3 bc = barycentric_2d(p, s0, s1, s2);
        ...
```
* **Impact**: Re-evaluating 2D cross products and floating-point divisions for every $(x, y)$ coordinate in the bounding box is highly redundant.
* **Remediation**: Standard edge equations ($E(x, y) = A \cdot x + B \cdot y + C$) can be updated incrementally with single additions ($\Delta x = A$, $\Delta y = B$), replacing all per-pixel multiplication and division with cheap additions.

#### 4. Threading Granularity & Barrier Contention
In [`rasterizer.hpp:L436`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs/render/software/rasterizer.hpp#L436):
```cpp
if (use_parallel) {
    parallel_for_1d(config.job_system, miny, maxy + 1, ..., raster_rows);
}
```
* **Impact**: Triangles are iterated sequentially. When a large triangle is encountered, `parallel_for_1d` pushes row slices to `IJobSystem` and blocks on a `WaitGroup::wait()`. This creates a hard synchronization barrier per large triangle. Small triangles do not parallelize, while large ones suffer thread launch and join overhead.
* **Remediation**: Adopt screen-space tile binning (e.g. $32 \times 32$ or $64 \times 64$ tiles). Bin triangles into tiles during a coarse setup pass, then dispatch one job per tile where workers rasterize into disjoint memory without any synchronization barriers.

---

### 3.3 RHI & Vulkan Edge Integration (`shs/rhi/`)

#### Pure Value Translation Layer
The value-oriented Vulkan translation in [`include/shs/rhi/vulkan/value/`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs/rhi/vulkan/value) (e.g. [`vk_device.hpp`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs/rhi/vulkan/value/vk_device.hpp)) is an exemplary implementation of Constitution II:
* Maps `RHIFormat`, `RHIBufferUsage`, `RHIImageUsage`, `RHIMemoryClass`, and `RHIPipelineStage` to `Vk*` flags using pure `constexpr` / `inline` functions.
* Zero `VkDevice`, `VkInstance`, or GPU driver dependencies are required to test the translation logic, as demonstrated by the fast-running [`vk_driver_tests.cpp`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/tests/vk_driver_tests.cpp).

#### Dual-Backend Parity
[`tests/sw_vk_parity_tests.cpp`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/tests/sw_vk_parity_tests.cpp) verifies that a canonical test triangle produces identical color outputs between CPU software rasterization and GPU Vulkan offscreen rendering within a 1-quantum 8-bit color tolerance. This provides a rock-solid regression guard against coordinate drift.

#### Lingering Forwarder Debt
Several files in [`include/shs/rhi/drivers/vulkan/`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs/rhi/drivers/vulkan/) are 5-line forwarders redirecting to `shs/rhi/vulkan/runtime/`. While harmless, these forwarders represent legacy migration residue that should be cleaned up once external consumer code is fully repointed.

---

### 3.4 Application & Session Orchestration (`shs/app/`)

#### Authoritative State Ownership
In [`shs/app/session_orchestrator.gateway.hpp`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs/app/session_orchestrator.gateway.hpp), the library addresses historical state fragmentation:
* `SessionState` is the single authoritative owner of session-level settings (camera pose, FOV, near/far clipping planes, light shaft toggles, and quit flags).
* Render projections ([`shs::Camera`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs/camera/camera.contract.hpp) and [`FrameParams`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs/render/frame/frame_params.hpp)) are updated strictly via pure synchronization funnels in [`session_settings_sync.hpp`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs/app/session_settings_sync.hpp).
* Zero-signal-loss is guaranteed: every input command produces an explicit discrete event (`CameraTranslatedEvent`, `CameraRotatedEvent`, `RuntimeFlagToggledEvent`, `QuitRequestedEvent`), and `InputStep` counts are checked against input span length via `SHS_POST`.

#### Headless Replay Engine: `VerticalSliceHost`
[`shs/app/vertical_slice_host.hpp`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs/app/vertical_slice_host.hpp) provides a reference integration host that runs without SDL, windowing systems, or GPU drivers:
* Takes recorded input streams and renderpath commands.
* Produces a deterministic 64-bit FNV-1a pixel digest (`digest_render_target`).
* This enables bit-for-bit replay parity verification across CI runs and platforms.

---

## 4. Code Quality, Safety & Ergonomics

### 4.1 Modern C++23 Idioms
* **`std::expected` / `std::unexpected`**: Used as the sole mechanism for domain rejections across compilers and gateways.
* **`std::span` & `std::string_view`**: Used consistently for non-owning, zero-allocation input views.
* **Monadic Chains**: Idiomatic `.transform()`, `.or_else()`, and `.transform_error()` pipelines throughout gateway assemblies.
* **Concepts & Type Traits**: Exhaustive variant visitation patterns enforced via `std::is_same_v` and `static_assert(sizeof(T) == 0)`.

### 4.2 Error Handling & Logging
* The engine completely avoids throwing exceptions in the core rendering and simulation path.
* Failure states are explicit, typed, and represented as closed enumerations (e.g. `PathSwapRejectionReason`, `RenderPathCompileRejection`).
* Invariant violations are channeled through the centralized violation handler in [`contract_guardrails.hpp`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/shs-renderer-lib/include/shs/core/contract_guardrails.hpp), which cleanly isolates debug assertions from release `[[assume]]` optimizations.

---

## 5. Prioritized Recommendations & Action Items

### Priority 1: Performance & Hot-Path Hardening (Short-Term)

| Item | Subsystem | Issue | Actionable Solution |
| :--- | :--- | :--- | :--- |
| **P1.1** | `rasterizer.hpp` | Per-triangle heap allocations via `std::vector<RasterVertex>` | Replace `std::vector` with fixed-capacity `std::array<RasterVertex, 16>` buffer ping-ponging during Sutherland-Hodgman clipping. |
| **P1.2** | `rasterizer.hpp` | Per-pixel `std::function` fragment shader invocation | Provide template-specialized rasterization loops or raw function pointers to allow full compiler inlining and auto-vectorization. |
| **P1.3** | `rasterizer.hpp` | Redundant per-pixel barycentric calculations | Implement standard incremental edge-function stepping ($\Delta x$, $\Delta y$) across triangle bounding boxes. |

### Priority 2: Architecture & Extensibility (Medium-Term)

| Item | Subsystem | Issue | Actionable Solution |
| :--- | :--- | :--- | :--- |
| **P2.1** | `rasterizer.hpp` | Per-triangle multithreading sync barrier | Transition software rasterization from per-triangle `parallel_for_1d` to screen-space tile binning ($32 \times 32$ tiles) for wait-free parallel dispatch. |
| **P2.2** | `renderpath/planning` | Closed `PassId` enum restricts user passes | Expand `PassId` to support open 64-bit string-hash keys or an extensible registry per Constitution I §7. |
| **P2.3** | `rhi/drivers` | Stale forwarder files in `rhi/drivers/vulkan` | Remove legacy forwarding headers once downstream demos and tooling are verified canonical. |

### Priority 3: Long-Term Modernization (Post-C++26)

| Item | Subsystem | Issue | Actionable Solution |
| :--- | :--- | :--- | :--- |
| **P3.1** | `core/` | C++23 contract emulation bridge | Execute [`cpp26_native_switch_runbook.md`](file:///home/sharavsambuu/src/dev/leisure-software-renderer/docs/backlog/cpp26_native_switch_runbook.md) once GCC 16 / Clang contracts support lands. |
| **P3.2** | `rasterizer.hpp` | Scalar pixel math | Exploit `xsimd` to vectorize 4 or 8 pixels simultaneously during fragment evaluation and depth testing. |
| **P3.3** | `task/` | Classic mutex/CV thread pool | Explore C++26 / `stdexec` (P2300) sender-receiver execution graphs for compute and tile dispatch. |

---

## 6. Conclusion

`shs-renderer-lib` is an exceptionally well-architected, disciplined modern C++23 library. It demonstrates rare adherence to formal architectural laws, mathematical rigor, and deterministic value transitions. Its boundary enforcement tooling (`check_kdba_boundaries.sh`, `check_contract_placement.sh`, `check_gateway_rails.sh`) sets an industry-grade standard for architectural preservation in large-scale codebases.

Addressing the highlighted software rasterizer hot-path bottlenecks (eliminating per-triangle heap allocations and dynamic fragment calls) will elevate the library's software rendering performance to match the exceptional quality of its architecture.
