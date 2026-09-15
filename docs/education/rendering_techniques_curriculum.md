# Rendering Techniques Curriculum

> Status: active educational curriculum (2026-09). Part of the new `docs/education/`
> sector: progressive technique ladder for sharpening shs-renderer-lib, from first
> principles to SOTA. Each rung is executable work (recipe preset + pass contracts +
> Slang module + software `cpp_impl` + validation gate), not a demo.
> Companion docs: `docs/roadmap/slang_utilization_plan.md` (Slang migration rides on
> this ladder), `docs/arch/render_path_architecture.md` (recipe/pass machinery).

## How to use this curriculum

- **A rung is "done"** when it lands as: `RenderPathRecipe` preset + pass contracts
  registered in `pass_contract_registry` + Slang module(s) in
  `shs/execution/shader/slang/` + software `cpp_impl` handler + cross-backend
  equivalence test (P2.5 harness). If it ships as a standalone demo with its own
  shader glue, it is a portfolio piece, not a library feature.
- Difficulty escalates along the axis the architecture is built for — **data flow
  between passes** — not shader fanciness: multi-pass → resource dependencies;
  deferred → attachment semantics; clustered → data-parallel backend parity.
- Status markers: ✅ exists end-to-end in the lib · 🔶 frozen GLSL reference exists
  (port at migration) · ⬜ planned (new work) · 💡 deep bet (recommended SOTA target).
- GLSL references live frozen under `shs-renderer-lib/shaders/vulkan/` — port
  semantics, cite file+lines, never include them (see Slang plan §0).

## Tier 0 — Rasterization foundations (✅ done in software + Vulkan edges)

> Post-mortem with lessons, pitfalls, and a parity-harness note:
> `docs/education/tier0_rasterization_lessons.md` — read before starting Tier 1.

1. ✅ Triangle rasterization with barycentric varying interpolation
2. ✅ Perspective & orthographic projection, view frustum
3. ✅ Depth test, alpha blending, render-order
4. ✅ Texture sampling (bilinear, repeat/linear), scissor/viewport
5. ⬜ Stencil buffer operations

## Tier 1 — Classic shading

6. ✅ Lambert / Gouraud diffuse *(GLSL ref: `default_scene`)*
7. ✅ Phong / Blinn-Phong specular *(exists in `builtin_shaders` + C++ impls)*
8. 🔶 Normal mapping — tangent space, TBN construction *(GLSL ref: `pb_scene`)*
9. ⬜ Parallax / parallax occlusion mapping
10. ⬜ Toon/cel shading + outline (inverted hull) passes
11. ⬜ Decals & projective texturing

## Tier 2 — Physically-based materials

12. ✅ PBR metallic-roughness (Cook-Torrance BRDF) *(GLSL ref: `pb_scene`)*
13. 🔶 Specular-glossiness workflow
14. ⬜ Image-based lighting (IBL): irradiance map, prefiltered env, split-sum BRDF LUT
15. ⬜ Anisotropic specular
16. ⬜ Subsurface scattering approximations (pre-integrated skin)
17. ⬜ Hair/fur shading (Kajiya-Kay → Marschner)
18. ✅ Area lights — rect/tube shapes already encoded in `CullingLightGPU`
19. ⬜ Volumetric/holographic material stylings
20. 🔶 Physically-based atmosphere sky *(Slang plan P1: `default_sky` module)*

## Tier 3 — Shadows

21. ✅ Shadow mapping *(PassId::ShadowMap; handlers + GLSL ref: `pb_shadow`)*
22. ✅ PCF with configurable radius/step/bias *(already in `ShaderUniforms`)*
23. ✅ Soft shadows with occlusion culling *(existing `_sw`/`_vk` demo pair)*
24. ⬜ Cascaded shadow maps (CSM) — depth-split cascade selection
25. ⬜ Variance / exponential variance shadow maps (VSM/EVSM)
26. ⬜ Point-light cube shadows (6-face atlas)
27. ⬜ PCSS (percentage-closer soft shadows)
28. ⬜ Contact shadows (screen-space ray march)

## Tier 4 — Multi-pass & post-processing

29. 🔶 Depth prepass *(PassId::DepthPrepass — near-trivial second pass; best early
    hardener for the resource/barrier plan)*
30. ✅ Tone mapping *(PassId::Tonemap; handler + GLSL ref: `pb_bright`/composite)*
31. ⬜ Bloom (bright-pass → downsample → upsample chain) *(ref: `pb_bright`)*
32. 🔶 FXAA *(GLSL ref: `pb_fxaa`)*
33. ⬜ SMAA (edge-detection + blending weights)
34. ✅ Motion blur *(PassId::MotionBlur; handler + GLSL ref)*
35. 🔶 Screen-space light shafts / god rays *(handler exists: `pass_light_shafts`,
    GLSL ref: `pb_shafts`)*
36. ✅ Lens flares *(GLSL ref: `pb_flare`)*
37. 🔶 SSAO *(PassId::SSAO; GLSL ref: `fp_stress_ssao` — variants HBAO, GTAO)*
38. ⬜ Chromatic aberration, vignette, film grain (cheap pass-chain practice)
39. ⬜ Auto-exposure (histogram-based, compute reduce)
40. ⬜ Depth of field (bokeh) *(PassId::DepthOfField; GLSL ref: `fp_stress_dof`)*
41. 🔶 Screen-space reflections (hierarchical ray march)

## Tier 5 — Transparency & volumetrics

42. ⬜ Weighted blended OIT (order-independent transparency)
43. ⬜ Depth peeling / per-pixel linked lists OIT
44. ⬜ Volumetric fog & volumetric lighting (clipmap volumes)
45. ⬜ GPU-simulated particle systems (compute dispatch)
46. ⬜ Atmospheric scattering (Bruneton / Hillaire precomputed)
47. ⬜ Volumetric clouds (raymarched, temporal reprojection)
48. ⬜ FFT ocean (compute + displacement shading)
49. ⬜ Refraction & water caustics

## Tier 6 — Pipeline architectures (recipe system's home turf)

50. 🔶 Forward rendering *(minimal_forward recipe — Slang plan P1.5)*
51. ✅ Forward+ / tiled forward *(PassId::PBRForwardPlus; light_cull GLSL ref;
    tile size is already a `RenderPathRecipe` knob)*
52. 🔶 Clustered forward *(PassId::PBRForwardClustered + ClusterBuild/
    ClusterLightAssign pass ids reserved; cluster_z_slices knob exists)*
53. 🔶 Deferred shading *(PassId::GBuffer/DeferredLighting; GLSL ref:
    `fp_stress_deferred` — software side has no GBuffer handler yet; exposes a
    real lib gap)*
54. ⬜ Tiled deferred *(PassId::DeferredLightingTiled)*
55. ⬜ Hybrid deferred/forward (deferred opaque + forward transparency)
56. ⬜ Visibility buffer (material ID + triangle ID, deferred material resolve)
57. ⬜ Bindless rendering *(descriptor-indexing cap already in BackendFeatureCaps)*

## Tier 7 — Global illumination

58. ⬜ SH light probes (L1/L2 irradiance)
59. ⬜ Irradiance volumes (grid probe interpolation)
60. 🔶 Light propagation volumes *(GLSL ref: `light_math` groundwork)*
61. ⬜ Dynamic Diffuse GI (DDGI) — **recommended SOTA deep bet**; probe update +
    sample passes map cleanly onto the pod/reducer pass model
62. ⬜ Voxel GI (VXGI / cone tracing)
63. ⬜ Screen-space GI (SSGI/SSDO)
64. ⬜ Radiance cascades (modern probe-free GI frontier)
65. ⬜ RT GI *(ray_query cap already declared; ref: `hello_ray_query` demo)*

## Tier 8 — Ray tracing (SOTA)

66. ⬜ RT shadows *(ray_query feature cap exists; hello_ray_query reference)*
67. ⬜ RT reflections
68. ⬜ ReSTIR DI — reservoir-sampled direct lighting (modern many-light standard;
    pairs with the clustered light data path)
69. ⬜ ReSTIR GI
70. ⬜ Path tracing + denoising (SVGF-style)
71. ⬜ RTXDI-style many-light importance sampling

## Tier 9 — Temporal & reconstruction (arrives after handle-based uniform pods)

72. ⬜ TAA *(PassId::TAA reserved)* — history rejection, jitter sequences; punishes
    any temporal-state weakness; schedule last of the AA family
73. ⬜ TAA upscaling (FSR/DLSS-class spatial+temporal upscaler)
74. ⬜ Frame generation (motion-vector extrapolation)
75. ⬜ SVGF/Edge-variance denoising for RT signals

## Tier 10 — GPU-driven rendering (signature tier for this architecture)

76. ✅ GPU-driven view/shadow culling *(existing soft-shadow/occlusion culling
    `_sw`/`_vk` demo pair, Jolt shape volumes)*
77. ⬜ HZB occlusion culling *(GLSL ref: `fp_stress_depth_reduce.comp`)*
78. ⬜ GPU-driven indirect draw (device-side command generation)
79. ⬜ Mesh shaders *(caps fields exist: `max_mesh_*` limits; ref:
    `hello_mesh_shader`)*
80. ⬜ Virtualized geometry / Nanite-style cluster LOD
81. 🔶 GPU compute light culling *(GLSL ref: `fp_stress_light_cull.comp` — the
    CullingLightGPU single-source milestone, Slang plan P2)*
82. ⬜ Software rasterization on GPU (compute rasterizer) — the library's
    signature domain: the same pure reducers run on CPU SIMD and GPU
83. ⬜ Visibility-buffer + meshlet hybrid (endgame composition)

## How the ladder maps onto the Slang plan

The technique ladder and the Slang migration (`docs/roadmap/slang_utilization_plan.md`)
are **one braided track, not two**: each tier rung migrates its passes to Slang as
part of landing, so nothing is ported twice.

| Ladder tier | Slang milestone | Recipe preset | Validation gate |
|---|---|---|---|
| T0–T1 | P0/P1 (toolchain, common modules, default_sky) | `minimal_forward` | ctest 5/5 + first equivalence test (tonemap near-exact) |
| T4 basics (prepass, tonemap, blur) | P1/P2 (manifest, reflection) | `minimal_forward` extended | equivalence: inter-pass state |
| T3 shadows | P2 (reflection → pod driver, CullingLightGPU single-source) | existing soft-shadow recipes | culling PODs compare (CPU vs GPU, order-insensitive) |
| T6 pipelines (fwd+, clustered, deferred) | P2 caps→specialization | `forward_plus`, `deferred` recipes | per-attachment format checks + equivalence |
| T10 GPU-driven | P3 completion | existing culling recipes | same as shadows |

**Definition of feature-complete for this curriculum:** solid Tiers 0–6 plus the two
SOTA deep bets — **DDGI** (Tier 7) and **GPU-driven culling** (Tier 10) — the two
techniques that best exploit the pure-reducer Domain POD architecture. Everything
between is a menu, not a checklist.

## Ordering rules (learned constraints, not preferences)

1. **Interleave DepthPrepass + SSAO before full deferred** — the first is a trivial
   second pass that hardens the resource/barrier plan; the second is the first
   technique whose correctness is statistical, a good early stress of the
   equivalence harness's tolerance philosophy.
2. **TAA comes last** of the temporal family and only after the handle-based
   uniform pod redesign — it punishes temporal-state weakness more than any other
   technique.
3. **No rung ships as a standalone demo.** The `PassId` → contract-registry →
   recipe-preset path is the definition of done; a technique that exists only as a
   demo binary is portfolio, not library.
4. **The frozen GLSL is reference, not source** (Slang plan §0): port semantics,
   cite, then delete at P3.

## Related docs
- Slang utilization plan: `docs/roadmap/slang_utilization_plan.md` (braided track)
- Render path recipe/pass machinery: `docs/arch/render_path_architecture.md`
- GI deep dives: `docs/arch/global_illumination_strategies.md`,
  `docs/roadmap/global_illumination_roadmap.md`
- Modern rendering strategies: `docs/arch/modern_rendering_strategies.md`
- GPU-driven rendering: `docs/arch/pc_gpu_driven_rendering_guide.md`
- Maturity tracking: `docs/roadmap/modern-rendering-maturity-roadmap.md`
