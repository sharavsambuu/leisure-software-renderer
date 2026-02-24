# C++20 Coroutines Opportunities

Research note on where `co_await`, `co_yield`, and `co_return` fit naturally in the renderer to replace complex state machines and callback chains.

VOP alignment note:
- Coroutine adoption is not a blocker for "full VOP-first" completion.
- Coroutines should stay on runtime/effect edges and must not introduce planner-side hidden mutation.

## 0. VOP Boundary Rules for Coroutines
1. Do use coroutines for runtime scheduling, GPU waits, and async I/O orchestration.
2. Do not use coroutines to mutate planning/reducer state machines in hidden or non-deterministic ways.
3. Keep planner/reducer layers as explicit value transforms; coroutine handles/promises should not leak into planning contracts.

## 1. Logic → Coroutine Scripts
Replace the callback-based `StateMachine` with sequential coroutine scripts for "Patrol -> Chase -> Attack" logic.

## 2. Asset Loading
Make `AssetManager::load_mesh_async()` return an awaitable to avoid stalling the main thread during I/O.

## 3. GPU Sync
`co_await` on Vulkan fences or timeline semaphores to free the CPU while waiting for the GPU.

## 4. Frame Timeline
Express the frame pipeline as a structured graph of awaited tasks:
`co_await all(spawn(cull), spawn(motion));`

## 5. Scripted Demos
Use coroutines for smooth camera tours and automated stress tests without time-accumulator math.

## 6. Multi-Frame GPU Ops
Simplify operations that span frames, like IBL mipmap generation or BVH rebuilds, using `co_await next_frame()`.
