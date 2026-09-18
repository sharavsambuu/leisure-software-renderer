# K-G4 — Library SW/Vulkan equivalence: evidence (2026-09-18)

Scope: the **library** G4 gate in
[`kdba_conformance_backlog.md`](kdba_conformance_backlog.md) — the same minimal
scene and policy through the library's actual execution paths for **both**
realizations, with documented per-output tolerances and independent
known-answer checks, wired as a portable CTest gate with backend diagnostics
retained. Adventure AD1/AD4 evidence is related work and never substitutes for
this gate.

## 1. What was actually missing

Prior state: `shs_renderer_sw_vk_parity_tests` already compared the CPU
rasterizer against the GPU offscreen path and recorded 15 differing pixels on
the 32×32 fixture. It was explicitly logged as *not* G4 (backlog, "PARTIAL
2026-09-17 (triangle parity)"), and all three stated reasons were real:

1. It called `shs::render::rasterize_mesh` **directly**. The comparison lived in
   the test body; the *library* had no CPU execution path to compare.
2. It named the concrete Vulkan backend and read back through the concrete API.
   No factory, no generic contract.
3. Its tolerances were bare numbers inside the test, and they disagreed with
   each other: one comment said "edge budget 8 pixels" while the assertion
   allowed 16.

Second gap, created by G3's own record: G3 left the software side **declining**.
`SoftwareRenderBackend::offscreen_execution()` returned `nullptr`, so "the same
minimal scene through actual library execution paths" had no CPU realization at
all. G4's dependency text ("a verified software realization of the selected
recipe") named exactly that hole.

## 2. The ruling: what a software realization of a SPIR-V pipeline means

`RHIGraphicsPipelineDesc` carries shader modules as **SPIR-V bytecode plus an
entry-point name** — that is the vendor-free contract's shader carrier. A CPU
rasterizer cannot execute arbitrary SPIR-V. Three options were considered:

| Option | Verdict |
| :--- | :--- |
| Interpret SPIR-V on the CPU | out of scope by an order of magnitude; would become a second compiler |
| Drop the module requirement for the CPU path | rejected — a caller could pass garbage bytecode and get CPU success with GPU failure, which destroys the equivalence claim it was meant to strengthen |
| **Bind the module by entry name to a built-in CPU realization, behind the same descriptor envelope** | **chosen** |

So the CPU realization validates the same descriptor envelope (including the
SPIR-V module header) and then resolves `vs_main`/`fs_main` to a built-in CPU
program pair. An unknown entry name is a **rejection** (`prepare_offscreen`
returns 0), never a silent approximation.

One authored detail had to be decomposed rather than copied. `vs_main` fetches
its positions *inside* the shader from `SV_VulkanVertexID`:

```slang
float2 positions[3] = { float2(-0.5, -0.5), float2(0.5, -0.5), float2(0.0, 0.5) };
return float4(positions[vertex % 3], 0.0, 1.0);
```

`ShaderVertex` has no vertex-ID input, so the CPU realization performs the
identical fetch as an explicit triangle-list mesh (one vertex per drawn vertex,
`positions[i % 3]`) and the vertex program only passes the fetched NDC position
through. This is behaviourally equivalent for a triangle list — the same
decomposition the earlier parity test used — and it is recorded here as the
reason the CPU realization is bound to an **authored recipe** rather than to
general shader portability.

## 3. Shape

- **One shared vendor-free descriptor gate.** `rhi_shader_module_supported()`
  and `rhi_graphics_pipeline_desc_supported()` now live in
  `rhi/pipeline/pipeline_desc.hpp`. `VulkanOffscreenPipeline::supports()` and the
  CPU realization both call it, so acceptance/rejection cannot drift between the
  two realizations. This was a mechanical extraction of the previously inline
  body (identical semantics); the dead
  `d.bytecode_size > std::numeric_limits<size_t>::max()` tautology in the old
  copy was dropped, and the two now-unused includes went with it.
- **The CPU realization is self-contained**: `SoftwareOffscreenExecution` owns
  its target and state, needs no device, and never forwards to the backend.
  Unlike the Vulkan adapter (composition because of the `std::expected`
  return-type collision) there was nothing to forward to.
- **Ids are descriptor-derived and stable**, as the GPU path's hash-keyed
  registries are: re-preparing the same descriptors after `reset_offscreen()`
  resolves the *same* target/pipeline ids, so a recorded stream stays valid
  across a reset/re-prepare cycle. This is asserted, not assumed.


## 4. Files

| File | Change |
| :--- | :--- |
| `shs/rhi/software/sw_offscreen.hpp` | **new** — `sw_offscreen_recipe` (authored positions, entry names, CPU program pair, mesh realization, RGBA8 store, descriptor-derived ids) + `SoftwareOffscreenExecution` |
| `shs/rhi/software/sw_backend.hpp` | includes the realization, overrides `offscreen_execution()`, owns one instance |
| `shs/rhi/pipeline/pipeline_desc.hpp` | **new** shared vendor-free descriptor gate (module header + envelope) |
| `shs/rhi/vulkan/value/vk_offscreen_pipeline.hpp` | `supports()` delegates to the shared gate; dead `shader_supported` + unused includes removed |
| `tests/sw_vk_equivalence_tests.cpp` | **new** G4 gate (dual-backend, factory-driven, generic surface) |
| `CMakeLists.txt` | registers `shs_renderer_sw_vk_equivalence_tests` with `SKIP_RETURN_CODE 77` |
| `tests/sw_vk_parity_tests.cpp` | stale "edge budget 8" comment corrected to 16 (matches the assertion) |
| `tests/vk_factory_offscreen_tests.cpp` | stale known-answer geometry comment corrected: the viewport maps the authored NDC triangle to pixel triangle (8,8) (24,8) (16,24) — base edge row 8, apex row 24 |
| `docs/backlog/engine_header_inventory.json` | regenerated same commit (Rule 15): 225 → 226 headers, plus corrected edges/consumers |

## 5. Known-answer independence

Each half is checked against the **authored scene**, never against the other
half's output. `offscreen_pipeline.slang` puts the triangle at NDC
(-0.5,-0.5) (0.5,-0.5) (0.0,0.5) and `fs_main()` returns the constant
float4(1.0, 0.25, 0.0, 1.0). Both realizations map NDC y downward, so on a 32×32
target that is the pixel triangle (8,8) (24,8) (16,24) — base edge on row 8,
apex on row 24. Both halves independently assert:

| Sample | Expected | Meaning |
| :--- | :--- | :--- |
| (16,12) | `(255,64,0,255)` | interior; 0.25 × 255 = 63.75 → 64 |
| (16,28) | `(0,0,0,0)` | below the apex — pass clear value |
| (1,1) | `(0,0,0,0)` | outside both edges; proves the readback wrote |

The gate runs this check twice, labels each result on stderr, and only then
compares the two readbacks. A parity-only design would have proven nothing about
either backend on its own.

## 6. Documented per-output tolerances

Two tolerances, each stated in the gate with its cause, not just its value:

| Output | Tolerance | Why it is the honest bound |
| :--- | :--- | :--- |
| Stored colour where **both** rasterizers covered the pixel | ≤ 1 per channel (one 8-bit quantum) | both paths store RGBA8 from the same authored constant; `mm_both == 0` is asserted, so a beyond-quantum colour disagreement is a hard failure |
| **Coverage** disagreement | ≤ 16 pixels at 32×32, and the GPU may only *add* coverage | the CPU rasterizer maps NDC onto the `(W-1)/(H-1)` extent while the GPU viewport uses the `w/h` extent; the boundary band is a screen-mapping convention gap, bounded rather than hand-waved |

Measured on lavapipe at 32×32:

```
equivalence: cpu_covered=113 gpu_covered=128 mismatches=15
             (both=0 gpu_only=15 cpu_only=0) max_delta=255
```

`gpu_only=15` is exactly the 15-pixel band the earlier parity work recorded
independently (2026-09-17) — an independent confirmation of the same physical
cause. `cpu_only=0` is asserted because a CPU-covered pixel going blank on the
GPU would mean a *missing* draw, not a boundary convention. `max_delta=255`
comes only from those 15 boundary pixels (orange vs transparent); it is a
coverage statistic, not a colour discrepancy, which is why `mm_both` is the

## 7. Same policy on both sides, and what "rejection" means

The gate runs **one** consumer function for both halves: `create_render_backend()`
→ `app::Context` → generic `IRenderBackend` → generic `IOffscreenExecution`. The
translation unit names no concrete backend and includes no GPU header. Every
probe is policy both realizations must agree on:

| Probe | Expected behaviour |
| :--- | :--- |
| Target without `TransferSrc` | refused; `offscreen_target()` stays 0 |
| Fragment module declared as a vertex stage | refused (shared envelope) |
| Legal preparation | nonzero pipeline id, nonzero target id, `prepare` reports capability claims on stderr |
| Re-preparation without reset | refused |
| Canonical stream (begin/bind/draw 3/end) | executes; pixels captured |
| Draw without a bound pipeline | refused |
| Empty stream / stream with no end pass | refused |
| Undersized readback buffer | refused (the readback size contract is exact) |
| Stream naming a foreign target id | refused |
| Repeated execution | succeeds and is **byte-identical** to the first run |
| After `reset_offscreen()` | target id 0, execution refused |
| Re-preparation of the same descriptors | resolves the **same** ids and the previously recorded stream works again |
| Different in-contract pipeline descriptor (cull = Back) | resolves a **different** pipeline id |

## 8. Skip discipline and retained diagnostics

- The **software half is always asserted.** A CPU rasterizer needs no device, so
  any refusal there is a hard FAIL (exit 1), not a skip. It cannot hide behind
  "no GPU today".
- Only the Vulkan half may report itself unavailable (`active != Vulkan`, no
  generic surface, no device, no SPIR-V). The process then exits **77** —
  CTest records a *skip*, and the stderr line says explicitly that the
  equivalence claim was **NOT** established. The software half must have passed
  first, and a software failure exits 1 before the Vulkan half is even reached.
- **Diagnostics retained** through the generic surface: backend `name()`,
  `supports_offscreen` / `supports_present`, the factory's `requested` / `active`
  / `note`, and a per-step refusal reason for whichever probe failed. What the
  generic contract cannot retain is the concrete failure *code* —
  `IOffscreenExecution` collapses detailed failures to `bool` by design (G3
  limitation 2, unchanged).
- The gate is registered inside the Vulkan-enabled configuration block (it links
  the loader), so a GPU-free build does not compile it — the same scope as the
  other two offscreen gates. The GPU-free path of the *software* realization was
  verified separately (see §9, row 5).

## 9. Verification

| # | Claim | Artifact | Result |
| :--- | :--- | :--- | :--- |
| 1 | Full build, no regressions | `cmake --build cpp-folders/build -j` | 100%, `BUILD_EXIT=0`, 0 errors, 0 warnings |
| 2 | Full suite green | `ctest --output-on-failure` | **69/69** (was 68/68); the only initial failure was the stale header inventory, regenerated |
| 3 | The new gate passes end to end | `shs_renderer_sw_vk_equivalence_tests` | `PASS` — CPU and GPU known-answers match the authored scene; equivalence within tolerance |
| 4 | Structural gates | `ctest -R "header_inventory\|include_graph\|boundary\|seam_symbols\|header_self_containment\|package_consumer\|header_migration"` | 10/10 passed |
| 5 | The software realization is genuinely device-free | throwaway TU including `backend_factory.hpp` compiled with `-Wall -Wextra -Wpedantic` and **no** `SHS_HAS_VULKAN`, then run | compiles clean; `requested=vulkan` falls back to `active=software`; CPU render through the generic surface yields interior `(255,64,0,255)` and clear `(0,0,0,0)` |
| 6 | Inventory reflects reality | `inventory_headers.py --write` | 225 → 226 headers; new header entry + corrected edges/consumers, all explained by this slice |

## 10. Remaining limitations (recorded, not claimed away)

1. **The CPU realization is bound to an authored recipe, not to SPIR-V.** It
   resolves modules by entry name (`vs_main` / `fs_main`) and rejects anything
   else. This is bounded SW/Vulkan equivalence over the declared fixed ABI — not
   general shader portability. Any new shader needs a CPU counterpart before it
   can participate.
2. **The software accepted set is a strict subset of the Vulkan one.** The
   generic contract still exposes no buffer-creation surface, so the CPU
   realization rejects vertex/index binding, indexed draws, instancing,
   first-vertex offsets and dispatch — commands a GPU *could* run. Equivalence is
   asserted only over the slice both accept (attribute-less `Procedural`,
   `draw(3)`, single pass).
3. **`Position2F` is gated but not CPU-realizable** for the same reason: it
   passes the shared descriptor envelope, then `prepare_offscreen()` returns 0
   because there is no generic vertex-buffer entry point to bind.
4. **Tolerances are calibrated for this fixture.** The colour quantum is
   structural (RGBA8), but the 16-pixel coverage budget is a 32×32 measurement of
   the `(W-1)/(H-1)`-vs-`w/h` screen-mapping gap. A larger target or a longer
   edge list needs its own measured bound, not this constant.
5. **Failure detail still collapses to `bool`** at the generic seam
   (`VulkanExecutionFailure` / typed backend errors remain on the concrete APIs).
6. **Asynchronous retirement is still unclaimed**; execution on both sides is
   synchronous.
7. **The gate is Vulkan-configuration-only.** In a GPU-free build the binary is
   not compiled, so its software half is not asserted by CTest there; the
   GPU-free software path is currently proven by the separate probe in §9 row 5,
   not by a committed gate. A committed GPU-free software gate is a candidate
   follow-up, not a claim of this slice.
8. **This is library evidence only.** It is not demo-consumer evidence: AD2/AD3
   and the rest of the AD track remain open and are tracked in
   [`adventure_demo_conformance_backlog.md`](adventure_demo_conformance_backlog.md).

binding assertion.
