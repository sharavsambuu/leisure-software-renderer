# C++20 Coroutines in `shs-renderer-lib` — Opportunities

Where C++20 coroutines (`co_await`, `co_yield`, `co_return`) fit naturally in the renderer, based on a scan of all modules.

---

## Why Coroutines in a Renderer?

A renderer's main loop is full of **"do X, then wait for something, then continue"** patterns — GPU fences, asset loads, multi-frame animations, FSM transitions. These are traditionally written as manual state machines or callback chains. Coroutines let you write the same logic as straight-line sequential code that suspends and resumes automatically.

> Coroutines ≠ threads. A coroutine runs on one thread, suspends at `co_await`, yields control back to the caller, and resumes later when the awaited value is ready. No locks needed.

---

## 1. `StateMachine` → Coroutine Behavior Tree

**File**: `shs/logic/state_machine.hpp`  
**Current**: Callback-based FSM — `on_enter`, `on_update(dt)`, `on_exit` lambdas per state with transition predicates.

**The problem**: Multi-step behaviors require splitting logic across many callbacks. A "patrol → chase → attack → return" sequence reads like scattered fragments rather than a coherent script.

**Coroutine replacement**: each behavior is a linear coroutine that `co_await`s on conditions:

```cpp
// Current: spread across on_enter/on_update/on_exit callbacks
// Coroutine version: reads like a script
Task enemy_behavior(EnemyCtx& ctx) {
    // Patrol phase
    while (!ctx.player_in_range()) {
        ctx.move_to(ctx.next_patrol_point());
        co_await next_frame();         // resume next tick
    }
    // Chase phase
    while (ctx.player_in_range() && !ctx.player_in_attack_range()) {
        ctx.chase_player();
        co_await next_frame();
    }
    // Attack
    co_await ctx.play_attack_anim();   // suspends until animation done
    ctx.deal_damage();
    // Return
    co_await ctx.move_to_async(ctx.spawn_point());
}
```

The FSM's `tick(ctx, dt)` drives the coroutine scheduler each frame. States become `co_await` suspension points rather than explicit enum transitions.

**Benefit**: eliminates the `TransitionRule` predicate system for complex scripted behaviors. Keep `StateMachine` for simple reactive state (e.g. lamp on/off), use coroutines for scripted sequences.

---

## 2. Asset Loading — Async `co_await load_mesh()`

**File**: `shs/resources/asset_manager.hpp` — `load_mesh()`, `load_texture()`  
**Current**: Fully synchronous — `import_mesh_assimp()` blocks the main thread until complete.

**The problem**: loading a large mesh or texture stalls the frame loop. The only workaround is manually dispatching to a thread and polling for completion.

**Coroutine approach**: make `AssetManager::load_mesh_async()` return an awaitable:

```cpp
// Current — blocks main thread
MeshAssetHandle h = asset_mgr_.load_mesh("models/dragon.obj");

// Coroutine version — suspends caller, resumes when ready
MeshAssetHandle h = co_await asset_mgr_.load_mesh_async("models/dragon.obj");
// execution here resumes on main thread, asset is ready
```

Internally, `load_mesh_async` enqueues the I/O + parse work to `IJobSystem`, returns an awaitable that registers a continuation, and the job calls `resume()` on the coroutine handle when done.

**Benefit**: async asset streaming without callbacks or polling. Demo `init()` functions can `co_await` all assets in sequence and read naturally.

---

## 3. GPU Fence Awaiting — `co_await gpu_fence()`

**File**: `shs/rhi/drivers/vulkan/vk_backend.hpp`  
**Current**: `vkDeviceWaitIdle()` or per-frame fence polling — both either stall or require manual frame-ring bookkeeping.

**Coroutine approach**: wrap fence/semaphore waiting in an awaitable:

```cpp
// Current — blocks everything
vkDeviceWaitIdle(device_);

// Coroutine version — suspends this coroutine, CPU free for other work
co_await vk_->fence_awaitable(timeline_semaphore_, frame_signal_value_);
// resumes when GPU signals the semaphore value
```

A background thread polls/waits on the Vulkan fence and calls `handle.resume()` when signaled. The calling coroutine (e.g. a frame task) suspends without spinning.

**Benefit**: enables `cull → co_await gpu_compute_done → record draws` as a clean linear async pipeline instead of the current two-phase manual sync.

---

## 4. Frame Timeline Orchestration

**File**: `draw_frame()` in all demos  
**Current**: `draw_frame()` is a monolithic function that sequentially calls cull, upload, record, submit — with `WaitGroup` barriers between CPU stages.

**Coroutine approach**: express the frame pipeline as a coroutine, with each CPU stage co_awaiting completion of parallel work:

```cpp
Task frame_coroutine(FrameCtx& ctx) {
    // Fork parallel CPU work
    auto cull_task   = co_await spawn(task_frustum_cull(ctx));
    auto motion_task = co_await spawn(task_light_motion(ctx));
    co_await all(cull_task, motion_task);         // join both

    auto prefilter = co_await spawn(task_light_prefilter(ctx));
    co_await prefilter;

    co_await spawn(task_ubo_upload(ctx));          // upload
    co_await task_record_draws_mt(ctx);            // parallel recording
    co_await ctx.vk->end_frame_async(ctx.fi);      // submit + present
}
```

This replaces the explicit `WaitGroup` barrier model from `multithreaded_coding_best_practices.md` with structured concurrency — the **Task Graph** (optimization_tasks.md #14) implemented via coroutines.

**Benefit**: frame timeline is readable top-to-bottom in one function. Adding a new stage is one line, not a new `WaitGroup` + barrier pair.

---

## 5. Camera Tour / Scripted Sequences

**File**: Demo `main_loop()`, camera automation  
**Current**: Camera tours in stress demos use time-based `if (elapsed > threshold)` checks inside `update_scene_and_culling()` — fragile and hard to extend.

**Coroutine approach**: scripted sequences become linear:

```cpp
Task camera_tour(FreeCamera& cam) {
    cam.set_pos({0, 10, -30});
    co_await wait_seconds(2.0f);

    co_await lerp_camera_to(cam, {50, 8, 0}, 4.0f);   // smooth move
    co_await wait_seconds(1.5f);

    co_await lerp_camera_to(cam, {0, 40, 0}, 3.0f);   // overhead shot
    co_yield {};                                        // loop marker
}
```

`co_yield` at the end lets the tour loop by re-suspending at the start each time.

**Benefit**: camera scripts are readable, editable sequences — not arrays of waypoints with lerp state machines.

---

## 6. Multi-Frame GPU Operation Sequences

Any operation that spans multiple frames is a natural coroutine — e.g. generating IBL mipmaps, building a BVH, or uploading a large texture in chunks:

```cpp
Task upload_texture_streamed(TextureAssetHandle h, VkUploadCtx& up) {
    for (uint32_t mip = 0; mip < mip_count; ++mip) {
        upload_mip(h, mip, up);
        co_await next_frame();    // yield CPU, let frame run, resume next tick
    }
    finalize_texture_view(h);
}
```

---

## Summary — Where Coroutines Fit

| Area | Current Pattern | Coroutine Benefit | Effort |
|------|----------------|-------------------|--------|
| `StateMachine` complex behaviors | Callback FSM + transitions | Linear behavior scripts | Medium |
| Asset loading (`load_mesh`, `load_texture`) | Synchronous block | Async background load, no stall | Medium |
| GPU fence waiting | `vkDeviceWaitIdle` / polling | `co_await fence` — CPU free while GPU works | High |
| Frame pipeline (`draw_frame`) | Manual `WaitGroup` barriers | Structured async pipeline | High |
| Camera tour / scripted sequences | Time-based state polling | Linear readable scripts | Low |
| Multi-frame GPU operations (IBL, BVH, streaming) | Frame counter polling | `co_await next_frame()` loops | Low |

---

## What You Need First — `Task<T>` Coroutine Type

All of the above rely on a minimal coroutine infrastructure. The smallest usable foundation:

```cpp
// shs/job/task.hpp (new file)
struct Task {
    struct promise_type {
        Task get_return_object() { return {}; }
        std::suspend_never initial_suspend() { return {}; }
        std::suspend_never final_suspend() noexcept { return {}; }
        void return_void() {}
        void unhandled_exception() { std::terminate(); }
    };
};

// Awaitable for "resume next frame"
struct NextFrame {
    bool await_ready() { return false; }
    void await_suspend(std::coroutine_handle<> h) {
        get_frame_scheduler().push_next_frame(h);  // main loop resumes it
    }
    void await_resume() {}
};
inline NextFrame next_frame() { return {}; }
```

This is ~50 lines. Everything above builds on it.

---

## Integration with Job System — Coroutines as Task Graph

Coroutines and `IJobSystem` are **complementary, not competing**:

| Tool | Purpose |
|------|---------|
| `IJobSystem` + `parallel_for_1d` | Parallel work with no natural suspension point (cull, transform, recording) |
| Coroutines | Sequential async logic with suspension points (load, fence wait, scripted behavior) |

Together they implement the **Task Graph** (optimization_tasks.md #14) without building a separate graph data structure. The dependency graph is expressed directly as coroutine `co_await` call order:

```cpp
// Instead of: build graph, add edges, run graph
TaskGraph g;
auto n1 = g.add(cull,      {});
auto n2 = g.add(motion,    {});
auto n3 = g.add(prefilter, {n1, n2});
g.run_and_wait();

// With coroutines + spawn — the graph IS the code
co_await all(spawn(cull), spawn(motion)); // parallel, then join
co_await spawn(prefilter);                // runs after both
```

The `spawn(fn)` awaitable enqueues `fn` on `IJobSystem` and resumes the calling coroutine when the job completes. No graph type needed — just the `Task<T>` type and `spawn` + `all` awaitables (~100 lines total).

This means implementing the Task Graph upgrade and adopting coroutines are effectively the **same work** — do one, get the other free.

---

## Renderer Implementation Details — Three High-Value Targets

Beyond the demo layer, three specific renderer internals would benefit significantly:

### 7. Pipeline / PSO Compilation
**File**: `shs/pipeline/render_path_compiler.hpp`, demo `create_pipelines()`  
**Current**: `vkCreateGraphicsPipelines()` blocks the main thread at startup. With many PSO variants this is a noticeable stall.

```cpp
// Current — blocks startup
create_pipelines(); // may take 200ms+

// Coroutine version — runs in background, display loading frame meanwhile
co_await spawn(compile_all_pipelines_async());
// swap to ready state once done
```

Internally each `vkCreateGraphicsPipelines` call goes to a `IJobSystem` worker (Vulkan allows concurrent PSO creation from different threads). The coroutine resumes when all pipelines are ready.

### 8. IBL Map Generation
**File**: `shs/resources/ibl.hpp`  
**Current**: Irradiance + prefiltered env map generation is a sequential compute sequence typically blocking startup.

```cpp
Task generate_ibl_async(VkImage env_cube, IblMaps& out) {
    co_await dispatch_equirect_to_cube(env_cube, out.env_cube);
    co_await next_frame();                              // let GPU breathe
    co_await dispatch_irradiance_pass(out.env_cube, out.irradiance);
    co_await next_frame();
    for (uint32_t mip = 0; mip < kPrefilterMips; ++mip) {
        co_await dispatch_prefilter_pass(out.env_cube, out.prefilter, mip);
    }
    co_await next_frame();                              // BRDF LUT
    co_await dispatch_brdf_lut(out.brdf_lut);
}
```

The `co_await next_frame()` between passes keeps the frame loop responsive during IBL generation instead of freezing for 10s of frames.

### 9. `RenderPathCompiler` Initialization
**File**: `shs/pipeline/render_path_compiler.hpp`  
**Current**: Compile-then-use initialization. Multi-technique setup (Forward + SSAO + FXAA) compiles all passes sequentially.

```cpp
Task init_render_path_async(RenderPathCompiler& rpc) {
    auto forward = co_await spawn(rpc.compile_pass_async(PassType::Forward));
    auto ssao    = co_await spawn(rpc.compile_pass_async(PassType::SSAO));
    auto fxaa    = co_await spawn(rpc.compile_pass_async(PassType::FXAA));
    co_await all(forward, ssao, fxaa);   // all three compile in parallel
    rpc.link();                           // link after all pass shaders ready
}
```

---

## Updated Summary — All Opportunities

| Area | File | Coroutine Benefit | Effort |
|------|------|-------------------|--------|
| `StateMachine` behaviors | `logic/state_machine.hpp` | Linear behavior scripts | Medium |
| Asset loading | `resources/asset_manager.hpp` | Async, no main thread stall | Medium |
| GPU fence waiting | `rhi/.../vk_backend.hpp` | CPU free while GPU works | High |
| Frame pipeline | Demo `draw_frame()` | Structured async, replaces `WaitGroup` | High |
| Camera tour / scripts | Demo `main_loop()` | Readable sequential scripts | Low |
| Multi-frame GPU ops | IBL, BVH, streaming | `co_await next_frame()` loops | Low |
| PSO compilation | `pipeline/render_path_compiler.hpp` | Parallel background compilation | Medium |
| IBL generation | `resources/ibl.hpp` | Responsive multi-frame generation | Medium |
| RenderPath init | `pipeline/render_path_compiler.hpp` | Parallel pass compilation | Medium |
| Task Graph (job system) | `job/` | Replaces explicit graph structure | Low (once `Task<T>` exists) |

---

*See `optimization_tasks.md` Task Graph (#14) — coroutines implement it directly.*  
*See `multithreaded_coding_best_practices.md` Task Graph section — the `co_await all(spawn(...))` pattern supersedes the `WaitGroup` model when coroutines are available.*
