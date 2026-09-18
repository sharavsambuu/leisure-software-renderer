# K-G3 factory-facing execution — closure evidence (2026-09-18)

**Status: CLOSED 2026-09-18.** Owner items: `kdba_conformance_backlog.md` **G3**
(remaining slice) and `remaining_todos_2026-09-18.md` **K-G3**.

## 1. What was actually missing

The factory already constructed a real Vulkan backend
(`include/shs/app/backend/backend_factory.hpp` → `std::make_unique<VulkanRenderBackend>()`
inside the `SHS_HAS_VULKAN` branch), and `app::Context` already resolved backends
by type (`register_backend` / `backend(type)` / `active_backend`). What did **not**
exist was any way to *use* it generically:

| Element | State before this change |
| :--- | :--- |
| `IRenderBackend` surface | `type`, `name`, `capabilities`, `on_resize`, `begin_frame`, `end_frame` — **no execution, no device-open** |
| Offscreen execution | Only on the concrete `VulkanRenderBackend` value class (`prepare_offscreen` / `execute_offscreen` / `reset_offscreen`, `std::expected<..., VulkanExecutionFailure>` returns) |
| Device opening | Only the concrete `initialize_device()` / `init(InitDesc{...})` |
| Consumer idiom | Factory → `register_backend` → **`dynamic_cast<VulkanRenderBackend*>`** → concrete init/execute. Verified in `exp-plumbing/hello_soft_shadow_culling_vk.cpp:563,584`, `hello_rendering_paths.cpp:2322,2344`, `hello_light_types_culling_vk.cpp`, `hello_occlusion_culling_vk.cpp`, `hello_modern_vulkan.cpp`, `hello_mesh_shader.cpp`, `hello_ray_query.cpp` |

So "factory-facing execution" was nominal: the consumer immediately downcast back
to the concrete type. That downcast is the defect this slice removes, and it is
why the slice could not be a test-only change.

## 2. Ruling and shape

Option A (owner-approved): one optional hook plus a minimal, vendor-free contract.

* **Hook:** `IRenderBackend::offscreen_execution()` returning `IOffscreenExecution*`,
  defaulting to `nullptr`. Null means "fall back or skip", never "passed".
* **Contract:** `include/shs/rhi/core/offscreen_execution.hpp` — no `Vk*` type, no
  SDK include, no `VulkanExecutionFailure`. Methods: `initialize_device`,
  `prepare_offscreen`, `offscreen_target`, `execute_offscreen`, `reset_offscreen`.
* **Composition, not inheritance:** `prepare_offscreen`/`execute_offscreen` already
  exist on `VulkanRenderBackend` with `std::expected` return types, and C++ cannot
  overload on return type, so the generic names cannot be additional overrides on
  the same class. The generic surface is therefore implemented by a thin adapter
  (`VulkanOffscreenExecution`) that holds a non-owning backend pointer and forwards,
  collapsing failures to `false`. Failure detail stays on the concrete API.
* **`initialize_device()` is part of the contract** because the factory only
  *constructs* the backend. Without a portable open step the approved test cannot
  run without a downcast, which would have defeated the slice.
* **Windowed runtime backend stays `nullptr`:** `runtime/vk_backend.hpp` uses a
  Vulkan-specific `InitDesc` (window, drawable size, validation) and owns no
  self-owned offscreen target. Declining is recorded, not hidden.

## 3. Files

| File | Change |
| :--- | :--- |
| `include/shs/rhi/core/offscreen_execution.hpp` | **new** — vendor-free `IOffscreenExecution` contract |
| `include/shs/rhi/core/backend.hpp` | forward declaration + `offscreen_execution()` default `nullptr` |
| `include/shs/rhi/core/rhi_stack.hpp` | aggregation include |
| `include/shs/rhi/vulkan/value/vk_backend.hpp` | `VulkanOffscreenExecution` adapter + override |
| `tests/vk_factory_offscreen_tests.cpp` | **new** — the portable gate |
| `CMakeLists.txt` | `shs_renderer_vk_factory_offscreen_tests`, `SKIP_RETURN_CODE 77` |
| `docs/backlog/engine_header_inventory.json` | regenerated (224 → 225 headers) |

## 4. Known-answer independence (not parity)

The expected pixels are derived from the **authored** scene, not from a run of the
software rasterizer or any other backend. `tests/shaders/offscreen_pipeline.slang`:

* `vs_main` emits the triangle at NDC `(-0.5,-0.5) (0.5,-0.5) (0.0,0.5)` from
  `SV_VulkanVertexID % 3`, driven by a single non-indexed `draw({3})`;
* `fs_main` returns the constant `float4(1.0, 0.25, 0.0, 1.0)`.

On the 32×32 target that is the pixel triangle `(8,24) (24,24) (16,8)`, so the
interior sample `(16,12)` must read RGBA8 `(255,64,0,255)` and `(16,28)`, below the
base edge, must keep the pass clear value `(0,0,0,0)`. `(1,1)` additionally proves
the readback overwrote the buffer (it is pre-filled with `0xCD`, not with the clear
value). **No parity agreement is accepted as evidence here.**

## 5. Claims → artifacts

| Claim | Artifact / assertion |
| :--- | :--- |
| The backend comes from the factory, not a local instance | `create_render_backend(RenderBackendType::Vulkan)`; `requested == Vulkan` |
| The consumer path resolves it generically | `ctx.register_backend` + `set_primary_backend` + `ctx.backend(Vulkan)`; `type()` matches |
| The generic surface is optional and null-safe | `offscreen_execution() == nullptr` → explicit `SKIP` (77) with the factory note |
| Device opening is generic | `offscreen->initialize_device()`; false → `SKIP` (77) |
| Unsupported descriptors are rejected generically | target without `RHIImageUsage_TransferSrc` → `prepare_offscreen(...) == 0`, `offscreen_target() == 0` |
| The minimal scene executes and writes known pixels | `execute_offscreen(stream, pixels)` + the three samples in §4 |
| Repeated execution is stable | second `execute_offscreen` on the same stream, same known pixel |
| Failures are reported, not swallowed | missing-binding stream, empty stream, undersized output → all `false` |
| Reset invalidates, re-prepare recovers | `reset_offscreen()` → `offscreen_target() == 0`, execution `false`, re-prepare, known pixel again |
| The test never names the concrete backend | the TU includes no Vulkan header and no `value/vk_backend.hpp`; it includes `backend_factory.hpp`, `app/context.hpp`, `core/offscreen_execution.hpp` only |

## 6. Skip-not-pass discipline

`SKIP_RETURN_CODE 77` with an explicit stderr reason for: no generic surface
(no-Vulkan build, windowed/runtime backend, software backend), no minimal-scene
SPIR-V, and device/pipeline preparation failure. A skip is never counted as a pass
and the reason string names the declining backend.

## 7. Verification (all green)

| Gate | Result |
| :--- | :--- |
| Full build (`cpp-folders/build`, Debug) | **0 errors, 0 warnings** |
| Full CTest | **68/68 passed, 0 failed** (67/67 before; +1 new target) |
| `shs_renderer_vk_factory_offscreen_tests` | **Passed, 0.31 s** (real device in the default environment) |
| `tools/check_kdba_boundaries.sh` | `[gateway-rails] all checks passed` / `[kdba-boundary] all checks passed` |
| `tools/check_include_graph.py` | `OK: acyclic; SDK placement and value-tier purity rules hold` |
| `tests/header_self_containment_test.sh` | Passed (the new header compiles standalone) |
| `shs_renderer_boundary_check`, `package_consumer_test` | Passed |
| `tools/inventory_headers.py` | regenerated; diff is exactly 1 new header + its edges |

## 8. Remaining limitations (recorded, not claimed away)

1. The contract covers only the minimal **single-target RGBA8 offscreen slice**:
   no buffers, no depth/stencil, no indexed-vertex upload, no multi-pass, no
   presentation.
2. `execute_offscreen` collapses failure detail to `bool`. The rich
   `VulkanExecutionFailure` remains owned by the concrete API.
3. **Asynchronous retirement is still unclaimed.** Execution is synchronous.
4. `SoftwareRenderBackend` returns `nullptr` — the software rasterizer keeps its
   own path; giving software a generic offscreen surface is future work and would
   require updating this gate's expectation deliberately.
5. The windowed runtime backend returns `nullptr`; windowed/surface opening is
   explicitly outside this contract.
6. **This is G3 scope only.** It does not satisfy **G4** (library SW/Vulkan
   equivalence with documented per-output tolerances), and it is not
   demo-consumer evidence for AD4.

