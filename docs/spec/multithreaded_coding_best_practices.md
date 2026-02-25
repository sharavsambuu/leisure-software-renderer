# Multi-Threaded Coding Best Practices (Wait-Free Concurrency)

Guidelines for writing performant, thread-safe, and cache-friendly code using the `shs::ThreadPoolJobSystem`. Adherence to [Constitution III (DOD & ECS)](dod_ecs_architecture.md) is required.

## 1. The Wait-Free Two-Phase Model
Every frame must strictly separate mutation from recording, guaranteeing **100% lock-free and wait-free execution** during simulation:
- **Phase 1 (Simulate + Cull)**: Parallel compute (animation, frustum cull, light motion, physics). Jobs must run as **pure functions** without `std::mutex` or `std::atomic` locks.
  - Inputs must be passed as read-only spans (`std::span<const T>`).
  - Outputs must be passed as exclusive-write spans (`std::span<U>`).
- **Phase 2 (Record + Submit)**: Parallel recording of secondary command buffers based on Phase 1 output. Workers are **strictly read-only** with respect to scene state.

## 2. Ownership Contract & Data Layout
- **Never** pass a mutable reference into a job unless it is strictly scoped to that thread's exclusive output span.
- **Pre-allocate** output arrays indexed by instance and fill them in parallel (`output[i] = compute(input[i])`).
- **Struct of Arrays (SoA)**: Data must be laid out in contiguous arrays (SoA) to maximize cache-line efficiency during parallel traversal. Avoid passing Array of Structs (AoS) to high-volume jobs.
- **One Pool Per Worker**: Each recording thread must own its own `VkCommandPool`.

## 3. Patterns
- **Pattern A (Zero Contention)**: Use `parallel_for_1d` when items write to disjoint slots.
- **Pattern B (Sort then Parallel)**: Sort serially (e.g., front-to-back) then dispatch parallel processing.
- **Pattern C (WaitGroup)**: Use `shs::WaitGroup` to join parallel stages before proceeding.

## 4. Checklist for MT Recording
- [ ] Each worker has a dedicated command pool slot per frame-ring.
 - [ ] Pools are reset at the start of the frame.
 - [ ] `VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS` is set on the primary pass.
 - [ ] `vkCmdExecuteCommands` is called on the main thread after all workers sync.
 
 ## 5. Lock-Free Job System Evolution
 
 While `shs::ThreadPoolJobSystem` may currently utilize `std::mutex` and `std::condition_variable` under the hood for scheduling, the architectural target is a true **Wait-Free Job System**:
 
 - **MPMC Queues**: The internal scheduling must transition to a Multi-Producer/Multi-Consumer (MPMC) lock-free atomic queue or a work-stealing deque.
 - **Zero Context Switch Overhead**: Worker threads must never block on OS locks; they should actively pull jobs or spin-yield, ensuring 100% thread utilization during the `Simulate + Cull` and `Record` phases.
