# Vulkan Roadmap: From "Hello Triangle" to Production

We're moving away from the "Manual Labor" of legacy Vulkan and towards an automated, performance-first backend.

## Phase 1: Automation (Making it usable)
The biggest pain in Vulkan is synchronization.
- **Render Graph**: This is Task #1. We need a system that automatically handles image layout transitions and pipeline barriers based on a high-level "Pass" description.
- **Timeline Semaphores**: Ditch the binary fences. Use timelines to unify the sync between Graphics, Compute, and Transfer queues.

## Phase 2: The Modern Pipeline
- **Bindless Evolution**: Move all resource management to a global index system. No more descriptor churn.
- **Dynamic Rendering**: Kill the legacy `VkRenderPass` and `VkFramebuffer` boilerplate. Use `VK_KHR_dynamic_rendering` to keep the code clean.

## Phase 3: Pushing the GPU
- **Compute Everything**: Shift tasks like light binning, particle sim, and culling to dedicated compute queues.
- **Ray Queries**: Don't build a full Ray Tracer yet. Just use Ray Queries for "Perfect" shadows and Ambient Occlusion. It's much faster and easier to integrate into our rasterizer.

## Definition of "Done"
The backend is mature when:
1. You can swap between Forward+ and Deferred rendering paths with a single config change.
2. The engine automatically handles resizing, windowing, and multi-GPU selection.
3. We have zero validation errors under the most stressful scenes.

---

> [!IMPORTANT]
> **Focus**: Don't get distracted by "Shiny features." Finish the **Render Graph** first. It's the foundation that makes every other phase possible.
