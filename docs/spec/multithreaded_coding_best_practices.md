# Multi-Threaded Coding Best Practices

Guidelines for writing performant, thread-safe code using the `shs::ThreadPoolJobSystem`.

## 1. The Two-Phase Model
Every frame must strictly separate mutation from recording:
- **Phase 1 (Simulate + Cull)**: Parallel compute (animation, frustum cull, light motion).
- **Phase 2 (Record + Submit)**: Parallel recording of secondary command buffers. Workers are **read-only** with respect to scene state.

## 2. Ownership Contract
- **Never** pass a mutable reference into a job.
- **Pre-allocate** output arrays indexed by instance and fill them in parallel (`output[i] = compute(input[i])`).
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
