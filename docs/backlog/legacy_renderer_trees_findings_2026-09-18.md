# Legacy renderer experiment trees — findings (no reuse commitment)

> Status: **FINDINGS ONLY (2026-09-18).** This document records what is actually
> inside the legacy experiment trees under `cpp-folders/src/exps-*`, established
> by direct scan, so that later decisions rest on evidence rather than
> recollection. **Nothing here is a commitment to reuse any of it.** Reuse is
> optional by design — the plug points in §5 can be filled from the legacy code,
> from scratch, or not at all, and each would be its own proposal + owner ruling.
> Companions: `docs/arch/render_path_architecture.md` §4 (RP requirement table)
> and §5 (L4 maturity gap list); `docs/backlog/remaining_todos_2026-09-18.md`.

## 1. Scope and method

Scanned (2026-09-18, read-only):

| Tree | Code files | Disk |
| :--- | ---: | ---: |
| `cpp-folders/src/exps-gpu-renderer` | 17 | 1.4 MB |
| `cpp-folders/src/exps-software-renderer` | 117 | 2.6 MB |
| `cpp-folders/src/exps-rendering-adventures` | 28 | 436 KB |
| `cpp-folders/src/exps-other` | 16 | 204 KB |

Method: filename inventory (`find`), then content greps for `tiled`, `clustered`,
`deferred`, `forward`, `cull`, followed by targeted reads of every hit.
Re-verification commands are in §8.

## 2. Inventory: the advanced content is two monoliths, not a module set

The GPU tree carries only 17 code files but 1.4 MB, because the rendering-path
work is fused into two large, near-duplicate files:

| File | Lines | Bytes |
| :--- | ---: | ---: |
| `exps-gpu-renderer/exp-rendering-techniques/demo_forward_classic_renderpath.cpp` | 9,800 | 421,477 |
| `exps-gpu-renderer/exp-plumbing/hello_rendering_paths.cpp` | 9,385 | 402,164 |

They are forks of one another: the same feature set appears in both, roughly 400
lines apart (e.g. the F2 cycle hint at `:919` vs `:987`, the preset drive at
`:1787` vs `:1857`, the clustered recipe at `:5457` vs `:5560`). No tiled,
clustered, or deferred path exists as a separate translation unit anywhere in
any of the four trees.

## 3. Negative result: what is *not* there

| Searched for | Result |
| :--- | :--- |
| Files named `*tiled*`, `*clustered*`, `*deferred*` | **none** in any exps tree |
| Tiled/clustered/deferred lighting as a standalone module | **none** |
| A G-buffer fill implementation | **none** |

The techniques exist as *driving code inside the monoliths* and as *calls into
library enums* — they are not factored out. Any reuse therefore starts from
extraction, not from copying a module.

## 4. What is there (verified content)

Real per-mode culling work exists inside `hello_rendering_paths.cpp`:

| Site | Content |
| :--- | :--- |
| `:7741-7743` | mode dispatch across `Tiled` / `TiledDepthRange` / `Clustered` |
| `:7848-7849` | depth-range reduction; `dispatch_z` switched on `cluster_z_slices_` for `Clustered` |
| `:6268-6270` | a hand-tuned conservative bias for tiled-depth, commented "to avoid edge popping" |
| `:9250` | `LightCullingMode culling_mode_ = LightCullingMode::Tiled` (runtime-cycled) |
| `:8640` | `active_technique_ == shs::TechniqueMode::TiledDeferred` |
| `:1787`, `:2006`, `:5684-5687` | drives `shs::RenderPathPreset::ClusteredForward` / `::TiledDeferred` and looks compositions up by preset |
| `:5457` | `path_recipe.light_volume_provider = shs::RenderPathLightVolumeProvider::ClusteredGrid` |
| `:5497-5500` | asserts `PassId::PBRForwardClustered` and `PassId::DeferredLightingTiled` are in the profile |

`demo_forward_classic_renderpath.cpp` carries the same content (`:8151-8153`,
`:8239-8240`, `:9640`, `:6193`, `:6219`, `:6624-6626`, `:5601-5604`,
`:5701-5704`). `exp-plumbing/hello_light_types_culling_vk.cpp` adds
`LightCullingMode::TiledDepthRange` handling (`:1424`) and defaults to
`LightCullingMode::Clustered` (`:1982`).

Interpretation: these demos are already *consumers* of the library's
tiled/clustered vocabulary. They drive `TechniqueMode::TiledDeferred` /
`ClusteredForward` through `RenderPathPreset` and composition presets, not
through a recipe's `RenderPathRenderingTechnique`.


## 5. Where legacy code could plug in: three current-library holes

These are facts about the **current** library (not the legacy trees), recorded
here because they define the only places legacy technique code could land. All
line references are `shs-renderer-lib/include/shs/renderpath/execution/pass_adapters.hpp`
unless stated.

**The orchestration layer is already built.** Six passes exist with real ids,
contracts, IO, and semantics — `cluster_build` (`:616`), `cluster_light_assign`
(`:686`), `gbuffer` (`:746`), `deferred_lighting` (`:823`),
`deferred_lighting_tiled` (`:884`), `pbr_forward_clustered` (`:961`) — over
`TechniqueMode::TiledDeferred` / `ClusteredForward` / `Deferred`
(`shs/render/frame/technique_mode.hpp:19-25`) and
`LightCullingMode { None, Tiled, TiledDepthRange, Clustered }`
(`shs/lighting/light_culling_mode.hpp:28-34`). What is missing is the *content*:

| # | Hole | Evidence |
| :--: | :--- | :--- |
| 1 | **The G-buffer body is empty.** `PassGBufferAdapter` is registered with *no* render-target handles and its `execute_resolved` writes nothing, although its contract and IO declare three written targets. | `pass_adapters.hpp:746-780`, `:1566-1568` |
| 2 | **No technique-specific shading.** All three lighting passes execute the same `PassPBRForward` body, differing only in `preserve_existing_depth`; `deferred_lighting` reads `technique.albedo/normal/material/ao` that no software pass writes, then shades from the `Scene` directly. | `:866-873`, `:942-950`, `:1011-1019` |
| 3 | **Cluster math is only half library-owned.** `cluster_build` sizes the SoA only; the binning runs in `detail::execute_generic_light_culling`, and the standalone binning math sits behind a hard Jolt gate. | `:672-678`, `:727-741`, `light_culling_runtime.hpp:12-13` |

Supporting detail:

- Hole 1: class at `:746`; contract at `:752-765`; IO at `:766-773`; the whole
  body at `:775-780` is `(void)ctx; if (!request.valid) …; return
  executed_no_outputs();`. Registration passes **no handles**:
  `std::make_unique<PassGBufferAdapter>()` (`:1566-1568`), unlike every sibling
  pass which receives `rt_hdr, rt_motion, rt_shadow`. The pass therefore has no
  structural access to any target and cannot write the three it declares. The
  Vulkan side does have hooks (`vk_standard_pass_execution.hpp:229-237`,
  `begin_gbuffer_pass` / `record_inline_gbuffer`); the software side is a no-op.
  `GBuffer` is nonetheless a real `PassId` (`pass_id.hpp:29`, `= 6`) and appears
  in the `Deferred` and `TiledDeferred` profiles (`technique_profile.hpp:74`,
  `:87`).
- Hole 2: `deferred_lighting` `:866-873`; `deferred_lighting_tiled` `:942-950`;
  `pbr_forward_clustered` `:1011-1019`. In each, `pass_.execute(ctx, in)` is
  `PassPBRForward`. `PassPBRForwardAdapter` itself advertises
  `Forward | ForwardPlus | ClusteredForward` (`:1044-1047`).
- Hole 3: `cluster_build` body `:672-678` sets `tile_size`, `tile_count_x/y` and
  resizes `tile_light_counts`. `cluster_light_assign` `:727-741` calls
  `detail::execute_generic_light_culling(..., true)` and reports
  `produced_light_grid` / `produced_light_index_list`.
  `technique_uses_light_culling` (`:232-239`) covers `ForwardPlus`,
  `TiledDeferred`, `ClusteredForward`. `shs/lighting/light_culling_runtime.hpp`
  holds the real binning vocabulary (`LightBinCullingConfig` `:32`,
  `TileViewDepthRange` `:41`, `LightBinCullingData` `:54`, `project_aabb_bounds`
  `:96`) but the entire header is inside
  `#if defined(SHS_HAS_JOLT) && ((SHS_HAS_JOLT + 0) == 1)` (`:12-13`), so it does
  not exist in a Jolt-less build.

## 6. Reuse assessment

| Category | Verdict |
| :--- | :--- |
| The tiled/clustered **orchestration** | Already in the library; nothing to port. |
| Algorithm **content** for holes 1-3 | The only genuinely valuable part. Extract per algorithm, land as pass bodies. |
| The monolith **plumbing** | Do not port. Its ad-hoc wiring is a likely origin of the reported bugginess, and the pipeline now owns that layer (derived allocation/barriers, declared IO, named refusals). |
| A **second monolith** | Do not port. The two are forks; keeping both imports a drift hazard. |

Consequence: reuse is not "copy a render path in". It is "re-express one
algorithm as a pass body behind an existing contract". Porting a monolith
wholesale would relocate its bugs into a pass body with no structural gain.

## 7. Disposition (non-binding)

1. Keep `hello_rendering_paths.cpp` as the single reference artifact; treat
   `demo_forward_classic_renderpath.cpp` as a redundant fork and mark it as such
   so it is not mistaken for a second implementation.
2. Label both as reference / failure-catalog material, not as implementation, so
   they are never read as "the tiled/clustered renderer".
3. Take technique work one hole at a time, each behind its own proposal. Hole 2
   has the cleanest entry point: `deferred_lighting` already has its contracts,
   IO, and profile wiring — only its body is shared.
4. Ordering constraint: hole 2's *selection* is blocked on RP-5. Until the
   technique relation is total, a recipe cannot request `TiledDeferred` /
   `ClusteredForward` at all, and any ported body would be unreachable from the
   authoring path.

## 8. Re-verification

```sh
cd cpp-folders/src
find . -type f \( -name '*.cpp' -o -name '*.hpp' \) | grep -iE 'tile|cluster|defer'   # §3
grep -niE 'tiled|clustered' exps-gpu-renderer/exp-plumbing/hello_rendering_paths.cpp   # §4
sed -n '746,781p' shs-renderer-lib/include/shs/renderpath/execution/pass_adapters.hpp  # §5 hole 1
```
