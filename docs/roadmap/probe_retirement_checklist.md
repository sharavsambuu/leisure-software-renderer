# Probe Retirement Checklist (exp-plumbing)

> Companion to `docs/arch/render_path_domain_pod_architecture.md` §5:
> "`hello_modern_vulkan.cpp` and the other probes are migrated incrementally or
> retired once the decomposed demo covers them."
>
> Coverage baseline: demo TU `exp-rendering-techniques/demo_forward_classic_renderpath.cpp`
> after tranche 1 (commit `38f78ae`) — GPU-free build, forward-classic + deferred
> paths, light-cull + depth-reduce compute, layered shadows, SSAO/motion-blur/DoF,
> software-backend parity (10/10).
>
> Rule: **retire a probe only after** (a) its unique feature is demonstrably in
> the demo (grep evidence + golden visual run), (b) main-tree ctest stays 14/14,
> and (c) the golden run (`xvfb` + lavapipe + `SHS_DEMO_FRAME_LIMIT=5`) exits 0
> with no new VUIDs vs. baseline `970fae3`.

## Inventory (18 TUs, ~28.7k lines)

| Probe | Lines | Unique feature | Demo coverage (grep evidence) | Verdict |
| :--- | ---: | :--- | :--- | :--- |
| `hello_vulkan_triangle.cpp` | 407 | minimal Vulkan init + draw | full (demo backend init + swapchain present) | **Retire now** |
| `hello_software_triangle.cpp` | 229 | minimal SW rasterizer draw | full (demo SW backend parity run) | **Retire now** (verify golden SW run first) |
| `hello_pass_plumbing.cpp` | 440 | pass plumbing teaching TU | full (demo has 10 `vkCmdBeginRenderPass` sites) | **Retire now** |
| `hello_pass_basics_vulkan.cpp` | 4008 | VK render pass/framebuffer teaching | full (no feature absent from demo) | **Retire now** |
| `hello_pass_basics.cpp` | 1255 | SW pass basics; 1× `RenderPathCompiler` ref | near-full | Retire after confirming its single compiler reference is exercised by `hello_rendering_paths` |
| `hello_jolt_integration.cpp` | 217 | JPH physics bridge (`JPH::Mat`, `JoltRenderable`) | full (24 Jolt refs; `RenderPathLightVolumeProvider::JoltShapeVolumes` recipe lives in demo) | **Retire now** |
| `hello_culling_sw.cpp` / `hello_culling_vk.cpp` | 393 / 1238 | light-cull compute pair | full pending task-16 merge (demo has light-cull comp + culling-debug stats) | Merge → retire |
| `hello_light_types_culling_sw.cpp` / `_vk.cpp` | 1136 / 2016 | light-type variety + culling | full pending task-16 merge (demo: spot ×39, directional, point) | Merge → retire |
| `hello_occlusion_culling_sw.cpp` / `_vk.cpp` | 478 / 1574 | **query-pool** occlusion (`vkCreateQueryPool` ×2, `vkCmdBeginQuery`) | partial — demo's occlusion is depth-reduce compute; demo has 1 `vkCreateQueryPool` (line 3153, culling-debug stats) | Merge → **verify query-path parity** before retire |
| `hello_soft_shadow_culling_sw.cpp` / `_vk.cpp` | 1257 / 3182 | soft shadows + **secondary command buffers** (`vkCmdExecuteCommands` ×3) | partial — demo has soft-shadow features but **no `vkCmdExecuteCommands`** | Merge → note secondary-cmd-buffer gap; decide port-or-drop |
| `hello_modern_vulkan.cpp` | 476 | `VK_KHR_dynamic_rendering` | **none** (demo: 0 dynamic-rendering refs, classic render passes only) | Keep until demo adopts dynamic rendering or feature is de-scoped |
| `hello_mesh_shader.cpp` | 410 | `VK_EXT_mesh_shader` | **none** | Keep (specialized extension probe) |
| `hello_ray_query.cpp` | 562 | acceleration structures + ray query | **none** | Keep (specialized extension probe) |
| `hello_rendering_paths.cpp` | 9385 | 5 path presets via `RenderPath*` pods: `Forward`, `Deferred`, `ClusteredForward`, `ForwardPlus`, `TiledDeferred` | partial — demo covers Forward + Deferred only | **Keep** — this is the L4 value-vocab showcase; decompose per `domain_pod_engine_rollout_roadmap.md` (DoD: demo < ~1.5k lines, all 5 presets hot-swappable) |

## Retirement order

1. **Wave 1 — no-merge retirements** (feature already in demo): `hello_vulkan_triangle`,
   `hello_software_triangle`, `hello_pass_plumbing`, `hello_pass_basics_vulkan`,
   `hello_jolt_integration` (≈6.6k lines with `hello_pass_basics`). Also
   `hello_pass_basics` after the compiler-reference check. Each: `git rm` (recoverable from history), drop its
   CMake block, full build + ctest 14/14 + golden run.
2. **Wave 2 — after task-16 merges**: the four `_sw`/`_vk` culling pairs
   (≈11.3k lines). Retire each pair only when its merge tranche lands with
   golden-run parity.
3. **Wave 3 — feature-gated**: `hello_modern_vulkan`, `hello_mesh_shader`,
   `hello_ray_query` — keep until the demo (or the pod decompositions) covers
   their extension features, or they are explicitly de-scoped in the rollout
   roadmap.
4. **Last**: `hello_rendering_paths.cpp` — retires only when the decomposed demo
   exposes all 5 presets through the RenderPath pods.

## Non-goals

- No changes to `shs-gpu-renderer-lib` (drivers stay; only demo/probe TUs retire).
- No deletion of anything referenced by main-tree ctest.
- No deletion of the `SHS_HAS_VULKAN` guard structure.
