# Parallelization & Optimization Tasks

Tracked future work items for CPU/GPU performance improvements across `shs-renderer-lib` and the `exp-rendering-techniques` / `exp-plumbing` demos.

The job system infrastructure (`shs::ThreadPoolJobSystem`, `shs::parallel_for_1d`, `shs::WaitGroup`) is already implemented in `shs/job/`. All CPU tasks below can leverage it directly.

---

## 🚀 Quick Wins — Suggested Starting Points

Ordered by effort-to-payoff ratio. Start here before tackling medium/high complexity items.

| # | What | Where | Effort | Why it's easy |
|---|------|--------|--------|---------------|
| 1 | **Wire MT recording in `hello_soft_shadow_culling_vk`** | `draw_frame()` — flip `use_multithread_recording_` | **~30 lines** | `WorkerPool`, range functions, and `record_main_secondary_batch()` are already written — just needs dispatching via `ThreadPoolJobSystem` + `WaitGroup` |
| 2 | **Parallel light motion update** | Caller of `update_light_motion()` | **~10 lines** | Pure per-light transform, zero shared state — drop-in `parallel_for_1d` |
| 3 | **Parallel batch frustum cull** | `cull_vs_cell()` / `cull_vs_frustum()` inner loops | **~15 lines** | Each element writes only to its own `out.classes[i]` — zero contention |
| 4 | **Parallel instance UBO upload** | Per-frame UBO fill loop in renderpath demos | **~10 lines** | Each instance maps to its own buffer slot — pure `parallel_for_1d` |
| 5 | **Add range API to `hello_occlusion_culling_vk`** | Extract `record_main_draws_range()` | **~20 lines** | Mirror the existing `hello_soft_shadow_culling_vk` pattern |

> **Rule of thumb**: if the for-loop iterates over instances and each element only writes to `output[i]`, it's a drop-in `parallel_for_1d`. No design work needed.

For detailed implementation notes on each item, see the numbered sections below.

## Core Library — `shs-renderer-lib`

### 1. Parallel Batch Frustum Culling
**File**: `shs/geometry/jolt_culling.hpp` — `cull_vs_cell()`, `cull_vs_frustum()`

Both batch cull functions are sequential loops over all shapes. Each element is read-only and writes only to its own index in `out.classes[]`, making this a **zero-contention parallel_for** candidate.

```cpp
// Current: sequential loop at jolt_culling.hpp:342, 309
for (size_t i = 0; i < n; ++i) { out.classes[i] = classify_vs_cell(objects[i], cell); }

// Target: shs::parallel_for_1d over [0, n)
```

- **Complexity**: Low
- **Gain**: Linear with core count for 1000+ instance scenes
- **Used by**: All demos, `SceneCullingContext::run_frustum()`

---

### 2. Parallel Software Occlusion Rasterization *(Partial)*
**File**: `shs/geometry/culling_software.hpp` — `run_software_occlusion_pass()`, `rasterize_mesh_depth_transformed()`

The inner triangle rasterization loop in `run_software_occlusion_pass()` is sequential. The front-to-back sort must remain serial, but the **depth rasterization phase** can be tiled:

- Split the depth buffer into screen-space tiles (e.g. 4–8 horizontal strips)
- Each worker thread rasterizes only triangles whose bounding rect falls in its tile(s)
- Merge (no merge needed — each tile owns disjoint pixels)
- Per-instance AABB occlusion query runs serially after all rasters complete

- **Complexity**: Medium — requires tile-aware triangle dispatch
- **Gain**: Significant for scenes with many large mesh occluders
- **Used by**: `SceneCullingContext::run_software_occlusion()`, `demo_forward_classic_renderpath`

---

### 3. Parallel Light Motion Update
**File**: `shs/lighting/light_runtime.hpp` — `update_light_motion()`

Called once per light per frame. Each `LightInstance` is updated independently — no cross-light dependencies.

```cpp
// Current: caller iterates lights sequentially
for (auto& light : lights_) update_light_motion(light, time);

// Target: parallel_for_1d over light indices
```

- **Complexity**: Low
- **Gain**: Scales with light count (stress scenes use 200–500 lights)
- **Used by**: All demos with animated lights

---

### 4. Parallel Light-Object AABB Pre-filter
**File**: `shs/lighting/light_runtime.hpp` — `collect_object_lights()`

Iterates all visible lights for every visible object. Inner loop `light_affects_object()` is a pure AABB/sphere test per light — no shared state between objects.

- Parallelize over **objects** (outer loop), each independently building its `LightSelection`
- Output is per-object → no write conflicts

- **Complexity**: Low
- **Gain**: High in Forward+ / Clustered with many lights × many objects
- **Used by**: All forward rendering demos, UBO upload loop

---

### 5. Parallel Tile View-Depth Range Building
**File**: `shs/lighting/light_culling_runtime.hpp` — `build_tile_view_depth_range_from_scene()`

Iterates visible scene elements, projecting each AABB to find which screen tiles it covers. Tile writes (`min_view_depth[tile_idx]`, `max_view_depth[tile_idx]`) conflict between elements → needs **atomic float min/max** or per-thread tile buffers merged after.

- Best approach: each thread accumulates its own `TileViewDepthRange` scratch, merge with `std::atomic` or post-join reduction
- **Complexity**: Medium — atomic float min/max not in std; use `std::atomic<uint32_t>` with bit_cast or a manual spinlock tile reduce
- **Gain**: Meaningful for TiledDepthRange light culling mode
- **Used by**: `build_light_bin_culling()`, `demo_forward_classic_renderpath` compute culler

---

### 6. Parallel Light Bin Assignment
**File**: `shs/lighting/light_culling_runtime.hpp` — `build_light_bin_culling()`

The tile/cluster bin assignment loop (which light falls in which bin) iterates `visible_light_scene_indices` sequentially and writes to `bin_local_light_lists[]`. Bins are disjoint per-light assignment — **parallelize over lights**, each writing only to bins it touches using a per-bin mutex or lock-free append.

- **Complexity**: Medium — requires per-bin concurrent append (e.g. `std::vector` + mutex per bin, or pre-sort + prefix sum)
- **Gain**: High for clustered mode with many lights
- **Used by**: All tiled/clustered demos (future demos)

---

### 7. Parallel Logic System Tick
**File**: `shs/scene/system_processors.hpp` — `LogicSystemProcessor::tick()`

Currently ticks all `ILogicSystem` implementations serially. If systems are side-effect-free with respect to each other (no shared mutable state), they can be dispatched concurrently.

- Requires audit of per-system dependencies before enabling
- **Complexity**: Low (if systems are independent) / High (if interdependent)
- **Applies to**: `exp-plumbing` demos using the `LogicSystemProcessor` pattern

---

## Demo-Level — `exp-rendering-techniques` & `exp-plumbing`

### Multi-Threaded Command Recording — Per-Demo Status

All demos record draw commands on the main thread only. The strategy is:
1. Split `render_visible_indices_` or equivalent into N ranges
2. Each worker records a **secondary** command buffer for its range
3. Main thread calls `vkCmdExecuteCommands` to replay all secondaries

Secondary cmd buffers require `VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS` in `vkCmdBeginRenderPass` / `VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT` in Dynamic Rendering.

---

#### `hello_soft_shadow_culling_vk.cpp` — ⚡ MT infrastructure mostly done, not wired

**Status**: The most advanced demo. MT scaffolding is largely complete:
- `constexpr uint32_t kMaxRecordingWorkers = 8u` — constant defined
- `struct WorkerPool` — per-worker `VkCommandPool[kFrameRing]`
- `configure_recording_workers()`, `create_worker_pools()`, `destroy_worker_pools()` — pool lifecycle done
- `record_main_draws_range(cmd, set, begin, end)` — range-parameterized, ready to dispatch
- `record_depth_prepass_range(cmd, set, begin, end)` — same
- `record_aabb_overlay_range(cmd, set, begin, end)` — same
- `record_view_occlusion_queries_range(cmd, set, ring, begin, end)` — same
- `record_main_secondary_batch(...)` — secondary cmd buffer recording function **already written**

**Remaining work**:
- Wire `use_multithread_recording_` toggle into `draw_frame()` — currently it always calls `record_main_draws()` → `record_main_draws_range(0, n)` single-threaded
- Dispatch worker threads via `shs::ThreadPoolJobSystem` + `WaitGroup`: each worker calls `record_main_secondary_batch(...)` for its range
- Change `vkCmdBeginRenderPass` to use `VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS`
- Collect all secondary buffers, call `vkCmdExecuteCommands`
- Shadow pass `record_shadow_*_range()` functions need same treatment for shadow cmd buffers

**Passes suitable for MT**: main draw, depth prepass, occlusion queries, shadow depth

---

#### `hello_occlusion_culling_vk.cpp` — 🔴 Fully single-threaded, no MT structure

**Status**: All recording is in a single primary cmd buffer:
- `record_depth_prepass(cmd, set)` — iterates `render_visible_indices_` sequentially (lines 996–1021)
- `record_main_draws(cmd, set)` — iterates `render_visible_indices_` sequentially (lines 1077–1109)
- `record_occlusion_queries(cmd, set, ring)` — queries are inherently serial (query ordering matters)

**Recommended approach**:
- Extract range variants `record_depth_prepass_range` / `record_main_draws_range` (mirror the soft_shadow pattern)
- Occlusion queries must remain serial on one thread (query index ordering constraint)
- Add `WorkerPool`, dispatch via `ThreadPoolJobSystem`

**Passes suitable for MT**: depth prepass, main draw
**Passes NOT suitable**: occlusion query recording (ordering constraint)

---

#### `hello_light_types_culling_vk.cpp` — 🔴 Fully single-threaded

**Status**: Single primary cmd buffer, no range API. Has complex per-object light selection in the draw loop — each draw call requires `LightSelection` data to fill push constants.

**Special consideration**: The light selection per object (`collect_object_lights`) is CPU-side and could be pre-computed in a parallel pass before recording, storing results in a per-instance array. Recording then just reads the pre-computed selection — making the recording loop trivially parallelizable.

**Recommended approach**:
- Pre-compute `LightSelection` for each visible instance using `parallel_for_1d` (relates to task #4)
- Store in `std::vector<LightSelection> per_instance_light_selections_`
- Recording loops just read the cached result — safe to split across workers

**Passes suitable for MT**: main draw (after light pre-computation)

---

#### `hello_soft_shadow_culling_sw.cpp` / `hello_culling_sw.cpp` / other SW demos — ⚪ Not applicable

Software demos don't issue Vulkan draw calls in loops — they draw debug overlays only. MT recording not applicable.

---

### 8. Parallel Instance UBO Upload
**File**: `demo_forward_classic_renderpath.cpp` — UBO fill loop

The per-frame loop that fills `InstanceUBO` / `DrawPush` data for each visible instance is sequential. Since each instance maps to its own buffer slot:

```cpp
// Target: parallel_for_1d over visible instance indices
// Each thread writes to its own slot — no conflicts
```

- **Complexity**: Low
- **Gain**: Reduces CPU upload time on scenes with 1000+ instances

---

### 9. Parallel Shadow Map Cull (Future — Shadow Pass)
**File**: Will apply when shadow map pass is expanded

When a `ShadowMap` pass is added, culling scene instances against the shadow camera frustum is a separate sequential pass that could run **concurrently** with the main camera frustum cull using separate `SceneCullingContext` instances on separate threads.

- **Complexity**: Low (separate contexts, no shared mutable state)
- **Gain**: Hides shadow cull cost behind main cull on multi-core CPU

---

### 10. Multi-Threaded Secondary Command Buffer Recording
**File**: `demo_forward_classic_renderpath.cpp` — `record_inline_scene()`, all technique demos

The main draw call recording loop is fully single-threaded. The `VulkanRenderBackend` already has `get_secondary_command_buffer()` per-thread pools:

- Split instance batch into N ranges (N = worker count)
- Each worker records a secondary command buffer for its range
- Main thread executes all via `vkCmdExecuteCommands`

- **Complexity**: Medium
- **Gain**: Directly reduces frame time on CPU-bound scenes
- **Applies to**: All technique demos (universal goal)


---

## GPU / Vulkan

### 11. Timeline Semaphores for Compute/Graphics Sync
**File**: `shs/rhi/drivers/vulkan/vk_backend.hpp`

Replace binary semaphores for `compute_finished_[]` (compute → graphics dependency) with **Vulkan timeline semaphores** (`VK_KHR_timeline_semaphore`). Enables more expressive multi-queue dependency chains and simplifies frame overlap logic.

- **Complexity**: Medium — requires VkSemaphoreTypeCreateInfo + wait/signal value management
- **Impact**: Enables future async compute overlap (e.g. light cull compute running while previous frame's graphics completes)

---

*Items are grouped by scope (core lib → demo → GPU) and not by priority. Complexity ratings: Low = drop-in parallel_for, Medium = needs design, High = architectural change.*

---

## Job System — `shs/job/`

The current `ThreadPoolJobSystem` + `parallel_for_1d` + `WaitGroup` stack covers all 11 tasks above. The following are improvement ideas for more advanced scenarios:

### 12. Work-Stealing Queue
**Current**: Single `std::mutex`-guarded `std::queue<>` shared by all workers → lock contention under very fine-grained dispatch.  
**Improvement**: Replace with a **per-thread work-stealing deque** (Chase-Lev style). Idle workers steal from busy workers' tails. Eliminates the shared queue bottleneck entirely.  
- **When needed**: Only if job granularity drops to per-triangle or per-pixel level. Current chunk-based `parallel_for_1d` is not bottlenecked here yet.
- **Applicable to**: Task #2 (tiled occlusion rasterization) if triangles become the dispatch unit; Task #10 (secondary cmd recording) if instance batches become very small.
- **Complexity**: High

---

### 13. `parallel_for_2d` Tile Variant
**Current**: `parallel_for_1d` only — 2D tile work must be manually flattened.  
**Improvement**: Add a `parallel_for_2d(js, rows, cols, min_grain, fn)` that tiles a 2D domain and dispatches rectangular chunks.  
- **When needed**: Whenever the work domain is naturally 2D (screen tiles, depth buffer rows).
- **Applicable to**: Task #2 (software occlusion depth rasterization — rows × cols of the depth buffer); Task #5 (tile view-depth range building — tile grid).
- **Complexity**: Low — direct extension of `parallel_for_1d`

---

### 14. Task Graph / Dependency Ordering
**Current**: All jobs are independent fire-and-forget. `wait()` must be called manually to chain stages.  
**Improvement**: Add a lightweight `TaskGraph` that lets you express: *"run A and B in parallel, then C after both complete."* Enables pipelining of **cull → upload → record** as a declared graph rather than sequential `wait()` barriers.  
- **When needed**: When multiple independent CPU stages should run concurrently but have data dependencies between them.
- **Applicable to**: Tasks #1 + #3 (frustum cull + light motion) running concurrently → feeds Task #4 (light-object pre-filter) → feeds Task #8 (UBO upload) → feeds Task #10 (cmd recording). A task graph would express this naturally.
- **Complexity**: High

---

### 15. Job Priority Levels
**Current**: All jobs share one FIFO queue — no way to prioritize time-critical work.  
**Improvement**: Add `JobPriority` (High / Normal / Low) with separate queues or a priority heap. High-priority jobs preempt lower ones.  
- **When needed**: When background work (e.g. asset streaming, BVH rebuild) shares the job system with per-frame critical path work.
- **Applicable to**: Task #1 (frustum cull — High), Task #9 (shadow cull — High), Task #3 (light motion — Normal), any future async asset loading (Low).
- **Complexity**: Medium

---

*Job system improvements should only be pursued once the existing `parallel_for_1d` bandwidth is actually measured as a bottleneck. Premature upgrade has high cost with uncertain gain.*

