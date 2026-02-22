# Multi-Threaded Demo Code — Best Practices

Guidelines for writing performant, correct multi-threaded and multi-threaded rendering code in `exp-plumbing` and `exp-rendering-techniques` demos using `shs-renderer-lib`.

---

## 1. Frame Structure — The Two-Phase Model

Every demo frame must separate CPU work into two explicit phases:

```
Frame N:
┌─────────────────────────────────────────────────────┐
│  PHASE 1 — SIMULATE + CULL  (parallel CPU work)     │
│   • Animate instances                               │
│   • Frustum cull (parallel_for_1d)                  │
│   • Light motion update (parallel_for_1d)           │
│   • Software occlusion (tiled parallel raster)      │
│   • Light-object pre-filter (parallel_for_1d)       │
│   • Pre-compute per-instance push data              │
├─────────────────────────────────────────────────────┤
│  PHASE 2 — RECORD + SUBMIT  (parallel recording)    │
│   • Workers record secondary cmd buffers in ranges  │
│   • Main thread calls vkCmdExecuteCommands          │
│   • end_frame / present                             │
└─────────────────────────────────────────────────────┘
```

**Rule**: All mutable scene state must be fully resolved before Phase 2 begins. Workers in Phase 2 are read-only with respect to scene/instance data.

---

## 2. Ownership Rules — The Golden Thread Safety Contract

These rules eliminate data races without explicit locking in the hot path:

| Data | Owner | Workers can |
|------|-------|-------------|
| `instances_[]` | Main thread (Phase 1) | Read-only in Phase 2 |
| `render_visible_indices_[]` | Main thread (Phase 1) | Read-only in Phase 2 |
| `per_instance_light_sel_[]` | Main thread (Phase 1, pre-computed) | Read-only in Phase 2 |
| Secondary `VkCommandBuffer` | One worker per buffer | Write exclusively |
| `VkCommandPool` | One pool per worker per frame-ring | No sharing |
| Camera/light UBOs | Main thread (mapped, written before recording) | Read via GPU |

**Never** pass a mutable reference into a job. If a worker needs per-instance output (e.g. a computed transform), pre-allocate a flat output array indexed by instance, then fill it in parallel.

---

## 3. Parallel CPU Work — Using `shs::parallel_for_1d`

### Pattern A — Zero-Contention Array Fill
Use when each element writes to its own index only:

```cpp
// Pre-allocate output (same size as visible set)
std::vector<PerInstanceData> push_data(render_visible_indices_.size());

shs::parallel_for_1d(job_system_, 0, (int)render_visible_indices_.size(), /*min_grain=*/64,
    [&](int begin, int end) {
        for (int i = begin; i < end; ++i) {
            const uint32_t scene_idx = render_visible_indices_[i];
            push_data[i] = compute_push(instances_[scene_idx]); // pure read → pure write
        }
    });
// Phase 2 workers read push_data[i] safely
```

**Min grain guideline**: Use ≥ 64 items per chunk for culling/math, ≥ 4 items for heavier work (mesh transforms). Avoid grains smaller than 16 — task overhead dominates.

### Pattern B — Pre-Sort Then Parallel
When ordering matters for correctness (e.g. front-to-back for occlusion), sort on the main thread first, then dispatch parallel work over the sorted output:

```cpp
// Sort is serial (fast, acceptable)
std::sort(sorted_indices.begin(), sorted_indices.end(), front_to_back_cmp);

// Rasterization is parallel by tile (disjoint pixel ownership)
shs::parallel_for_1d(job_system_, 0, tile_count, 1,
    [&](int begin, int end) {
        for (int t = begin; t < end; ++t)
            rasterize_tile(sorted_indices, tile_depth_bufs[t], t);
    });
```

### Pattern C — Parallel with `WaitGroup` for Custom Sync
When you need to wait for a subset of jobs before proceeding:

```cpp
shs::WaitGroup wg{};
for (int i = 0; i < n_cull_jobs; ++i) {
    wg.add(1);
    job_system_->enqueue([i, &wg, &results]() {
        results[i] = run_cull_for_range(i);
        wg.done();
    });
}
wg.wait(); // block until all cull jobs complete before recording starts
```

---

## 4. Vulkan Multi-Threaded Recording — Secondary Command Buffers

### Setup Requirements

Every demo targeting MT recording must have:

```cpp
// 1. Per-worker command pools (one per frame-ring slot)
struct WorkerPool {
    std::array<VkCommandPool, kFrameRing> pools{};
};
std::vector<WorkerPool> worker_pools_; // size = worker_count_

// 2. Worker count from job system
uint32_t worker_count_ = 0;

// 3. In configure_recording_workers():
worker_count_ = std::min(
    static_cast<uint32_t>(job_system_->worker_count()),
    kMaxRecordingWorkers);
```

### Recording Pattern — Single Pass

```cpp
void record_draws_mt(VkRenderPass rp, VkFramebuffer fb, VkExtent2D ext,
                     VkDescriptorSet camera_set, uint32_t ring)
{
    const uint32_t n = (uint32_t)render_visible_indices_.size();
    const uint32_t chunk = (n + worker_count_ - 1) / worker_count_;

    // Reset worker pools for this ring
    for (uint32_t w = 0; w < worker_count_; ++w)
        vkResetCommandPool(device_, worker_pools_[w].pools[ring], 0);

    std::vector<VkCommandBuffer> secondary_cmds(worker_count_, VK_NULL_HANDLE);
    shs::WaitGroup wg{};

    for (uint32_t w = 0; w < worker_count_; ++w) {
        const uint32_t begin = w * chunk;
        const uint32_t end   = std::min(n, begin + chunk);
        if (begin >= end) continue;

        wg.add(1);
        job_system_->enqueue([=, &secondary_cmds, &wg]() {
            secondary_cmds[w] = record_secondary_batch(
                rp, fb, ext, camera_set, ring,
                worker_pools_[w].pools[ring],
                begin, end);
            wg.done();
        });
    }
    wg.wait();

    // Execute secondaries on main thread
    std::vector<VkCommandBuffer> valid{};
    for (auto cb : secondary_cmds)
        if (cb != VK_NULL_HANDLE) valid.push_back(cb);
    if (!valid.empty())
        vkCmdExecuteCommands(primary_cmd_, (uint32_t)valid.size(), valid.data());
}
```

### Secondary Buffer Allocation

```cpp
VkCommandBuffer record_secondary_batch(
    VkRenderPass rp, VkFramebuffer fb, VkExtent2D ext,
    VkDescriptorSet camera_set, uint32_t ring,
    VkCommandPool pool, uint32_t begin, uint32_t end)
{
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = pool;                          // worker-exclusive pool
    ai.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
    ai.commandBufferCount = 1;

    VkCommandBuffer cb = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(device_, &ai, &cb);

    VkCommandBufferInheritanceInfo inh{};
    inh.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
    inh.renderPass  = rp;
    inh.framebuffer = fb;
    inh.subpass     = 0;

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT |
               VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
    bi.pInheritanceInfo = &inh;
    vkBeginCommandBuffer(cb, &bi);

    // bind pipeline + descriptor sets (each secondary must bind its own)
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_tri_);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline_layout_, 0, 1, &camera_set, 0, nullptr);
    vk_cmd_set_viewport_scissor(cb, ext.width, ext.height, true);

    // draw range
    record_main_draws_range(cb, camera_set, begin, end);

    vkEndCommandBuffer(cb);
    return cb;
}
```

### Primary Command Buffer — Use `SECONDARY_COMMAND_BUFFERS`

```cpp
// When beginning the render pass that will use secondary buffers:
vkCmdBeginRenderPass(primary_cmd_, &rp_info,
    VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS); // ← required
record_draws_mt(fi.render_pass, fi.framebuffer, fi.extent, camera_set, ring);
vkCmdEndRenderPass(primary_cmd_);
```

> **Dynamic Rendering**: Use `VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT_KHR` in `VkRenderingInfo::flags` instead.

---

## 5. What Must Stay Single-Threaded

Not everything should be parallelized. Keep the following on the main thread:

| Operation | Why |
|-----------|-----|
| Front-to-back sort (occlusion) | Order dependency between steps |
| GPU occlusion query recording | Query index ordering is sequential |
| `vkCmdExecuteCommands` | API requires primary cmd buffer context |
| `vkBeginRenderPass` / `vkEndRenderPass` | Primary cmd buffer API |
| `VisibilityHistory` updates | Shared mutable history map |
| Scene state mutation (animation) | Single writer, many readers |
| Swapchain acquire / present | Vulkan API constraint |

---

## 6. Per-Instance Pre-Computation Pattern

For demos where the draw loop must compute something expensive per object (e.g. `LightSelection` in `hello_light_types_culling_vk`), always **pre-compute and cache** before recording:

```cpp
// Phase 1 (CPU, parallel)
std::vector<LightSelection> light_sel(render_visible_indices_.size());
shs::parallel_for_1d(job_system_, 0, (int)render_visible_indices_.size(), 32,
    [&](int begin, int end) {
        for (int i = begin; i < end; ++i) {
            const AABB obj_aabb = get_world_aabb(render_visible_indices_[i]);
            light_sel[i] = collect_object_lights(
                obj_aabb, light_scene_candidates, light_scene_, lights_, cull_mode_);
        }
    });

// Phase 2 (recording, workers read light_sel[i] — no race)
```

This converts a data-dependent serial loop into a trivially parallelizable recording pass.

---

## 7. Demo Checklist — New Demo Template

When writing a new Vulkan demo targeting full parallel rendering, ensure:

- [ ] **Job system pointer** `IJobSystem* job_system_` initialized in `init_backend()`
- [ ] **Worker count** `uint32_t worker_count_` set from `job_system_->worker_count()`
- [ ] **`WorkerPool` array** `std::vector<WorkerPool> worker_pools_` sized to `worker_count_`
- [ ] **`create_worker_pools()` / `destroy_worker_pools()`** called in setup/cleanup
- [ ] **Range recording functions** `record_*_range(cmd, set, begin, end)` for every draw pass
- [ ] **Pre-computed per-instance arrays** for any per-draw CPU work (push data, light selection)
- [ ] **`reset_worker_pools_for_frame(ring)`** called at start of `draw_frame()`
- [ ] **`VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS`** in render pass begin
- [ ] **`WaitGroup`** used to join worker threads before `vkCmdExecuteCommands`
- [ ] **Culling phases complete** (`parallel_for_1d`) before recording begins

---

## 8. Debugging Multi-Threaded Recording

Common issues and how to spot them:

| Symptom | Likely Cause |
|---------|-------------|
| Objects flickering in/out | `render_visible_indices_` mutated during recording |
| Validation error: wrong render pass | Secondary buffer's inherited `renderPass` doesn't match primary's |
| Crash in worker thread | Command pool shared between workers in same frame-ring slot |
| Incorrect draw order | Assuming secondaries execute in submission order — they do, but only after `vkCmdExecuteCommands` order |
| Occlusion query results wrong | Query pool reset and recording on different threads without sync |
| `vkCmdExecuteCommands` crash | Null `VkCommandBuffer` in the array — guard with validity check |

---

*See `hello_soft_shadow_culling_vk.cpp` as the reference implementation — it has the most complete MT recording scaffolding in the codebase.*  
*See `optimization_tasks.md` for the full list of CPU and GPU parallelization opportunities.*

---

## Future Job System Upgrades — How Patterns Change

These sections document how the best practices above evolve as each new job system capability from `optimization_tasks.md` is implemented. **Until a feature is implemented, use the patterns in sections 1–8 above.**

---

### With `parallel_for_2d` (task #13)

**Changes section 3** — adds Pattern D for naturally 2D work domains.  
Replace manual tile-index flattening with the 2D variant:

```cpp
// Current (manual flatten):
shs::parallel_for_1d(js, 0, tiles_x * tiles_y, 1, [&](int begin, int end) {
    for (int t = begin; t < end; ++t) {
        int tx = t % tiles_x, ty = t / tiles_x;
        rasterize_tile(depth_buf, sorted_indices, tx, ty);
    }
});

// With parallel_for_2d:
shs::parallel_for_2d(js, 0, tiles_y, 0, tiles_x, 1, [&](int ty, int tx) {
    rasterize_tile(depth_buf, sorted_indices, tx, ty);
});
```

**Use for**: tiled software occlusion rasterization, tile view-depth range building.  
**Min grain**: 1 tile per thread is safe — tiles own disjoint depth buffer regions.  
**Checklist addition**: Replace `parallel_for_1d` with `parallel_for_2d` wherever the loop naturally iterates over a `(rows, cols)` grid.

---

### With Task Graph (task #14)

**Changes section 1** — the two-phase serial barrier model becomes a declared dependency graph. Manual `WaitGroup` barriers between phases disappear.

```
// Current: sequential phases with WaitGroup barriers
phase1_cull();   // wait internally
phase1_lights(); // wait internally
phase2_record(); // wait internally

// With Task Graph: declare edges, let the graph schedule
TaskGraph graph{};
auto t_cull   = graph.add(task_frustum_cull,   {});
auto t_motion = graph.add(task_light_motion,    {});          // parallel with cull
auto t_prefilter = graph.add(task_light_prefilter, {t_cull, t_motion}); // waits for both
auto t_upload = graph.add(task_ubo_upload,      {t_prefilter});
auto t_record = graph.add(task_record_draws,    {t_upload});
graph.run_and_wait();
```

**Changes section 3** — Pattern C (`WaitGroup` for custom sync) is replaced by graph edge declarations.  
**Changes section 4** — `record_draws_mt()` becomes a graph node rather than an inline blocking call.  
**Checklist addition**: New demos declare a per-frame `TaskGraph` in `draw_frame()`. No manual `WaitGroup` needed for inter-phase sync.

---

### With Work-Stealing Queue (task #12)

**Changes section 3** — min-grain guideline relaxes significantly for fine-grained work. The work-stealing scheduler handles load imbalance automatically, so overly conservative chunking is no longer needed.

```cpp
// Current: chunk to avoid queue contention
shs::parallel_for_1d(js, 0, n, /*min_grain=*/64, fn);

// With work-stealing: can go down to grain=1 for heavy per-item work
shs::parallel_for_1d(js, 0, n, /*min_grain=*/1, fn);
// Or enqueue directly per-item for maximum stealing opportunity:
for (int i = 0; i < n; ++i) {
    wg.add(1);
    js->enqueue([i, &wg]() { fn(i); wg.done(); });
}
wg.wait();
```

**When to use direct enqueue**: only worthwhile if per-item cost is high enough (> ~5µs) to amortize job overhead. For math-heavy loops (culling, transforms), stick with `parallel_for_1d` with a moderate grain.  
**Checklist addition**: Per-item enqueue is valid for mesh BVH traversal, heavy BRDF evaluation, or any work with unpredictable per-element cost.

---

### With Job Priority Levels (task #15)

**Changes sections 3 and 4** — critical-path jobs (culling, cmd recording) use `High` priority. Background work shares the same pool at `Low` priority without starving the frame.

```cpp
// Recording workers — critical path
for (uint32_t w = 0; w < worker_count_; ++w) {
    wg.add(1);
    js->enqueue(JobPriority::High, [=, &wg]() {
        secondary_cmds[w] = record_secondary_batch(...);
        wg.done();
    });
}

// Background BVH rebuild — non-critical
js->enqueue(JobPriority::Low, [&]() {
    rebuild_bvh_async(scene_);
});
```

**Priority guide for demos**:

| Work | Priority |
|------|----------|
| Frustum cull | High |
| Shadow cull | High |
| Command recording | High |
| Light motion update | Normal |
| Light-object pre-filter | Normal |
| UBO upload fill | Normal |
| BVH/spatial structure rebuild | Low |
| Debug overlay generation | Low |

**Checklist addition**: Pass `JobPriority` enum to every `enqueue()` call. Default to `Normal` if unsure — only escalate to `High` for frame-critical path.
