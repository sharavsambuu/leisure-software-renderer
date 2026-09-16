# Slang Utilization Plan

> Status: active plan (2026-09). Owner: shs-renderer-lib shader pipeline.
> Predecessor docs: `docs/dev/build_and_setup.md` (Slang install, pinned v2026.17.1),
> `docs/roadmap/domain_pod_engine_rollout_roadmap.md` (pod architecture).
> Technique curriculum (which passes we build, in what order):
> `docs/education/rendering_techniques_curriculum.md` — the two docs are one
> braided track: each curriculum tier migrates its passes to Slang on landing.

## 0. Policy: GLSL is frozen

- **No new `.glsl` files** anywhere in the repo, effective immediately.
- **No edits** to existing GLSL — they are read-only reference material now.
- Retained temporarily under `shs-renderer-lib/shaders/vulkan/` until Slang is
  feature-complete (P3); they serve as porting references and for diff-checking
  migrated shader math.
- Deletion (P3) removes: all 30 GLSL files, the `glslangValidator` CMake path in
  parked demo CMakeLists, and the GLSL-era build steps. `vk_shader_utils.hpp`
  stays a plain `.spv` loader — compiler-agnostic by design.
- Enforcement: the P0 `shs_slang_shaders` CMake target is the only sanctioned
  shader build path; a trivial guard test fails if any `.glsl` appears outside
  `shaders/vulkan/` (nothing outside it exists today).

## 1. Why Slang is the right substrate for this architecture

The renderer is pure-reducer Domain PODs: passes consume/produce identical value
state on every backend. Slang's structural features map onto that directly:

| Slang feature | Renderer leverage |
|---|---|
| `module` / `import` (real modules, not text includes) | Shader modules mirror domain pods (`shs.lighting.types` ↔ `domains/lighting/light_types.hpp`). Kills `#include` drift, header-order bugs, macro guards — the exact failure mode `CullingLightGPU` suffered. |
| `interface` + `specialize<T>` | Pluggable materials/techniques as interfaces (`ILightingModel`), specialized at pipeline creation per `RenderPathRenderingTechnique`. Zero-overhead dispatch — the shader-side analog of the pluggable recipe. |
| `ParameterBlock<T>` / `StructuredBuffer<T>` | Bind domain pods (CullingLightGPU, frame uniforms) as named typed blocks; reflection validates them against the C++ static_assert'd pods → single-source structs (P2). |
| `[push_constant]` | Per-draw small data in the push block, budget-checked against `BackendLimitCaps::max_push_constant_bytes`. |
| Specialization constants | Bake capability limits and recipe knobs (`light_tile_size`, `cluster_z_slices` from `RenderPathRecipe`) into pipelines — one source of truth, no `#define` permutations. |
| `[shader("vertex"/"fragment"/"compute")]` | One `.slang` per pass with multiple entry points replaces the `.vert`/`.frag` paired-file convention; each entry point compiles to its own `.spv`. |
| `-emit-spirv-directly` (2026 default) | No glslang dependency; `slangc` alone is the Vulkan edge toolchain, GPU-free. |

## 2. Conventions & coordinate decisions (pin in P0 — load-bearing)

Each decision is enforced by a shared helper or CMake flag, never per-pass
discipline.

1. **Matrix layout — pin `-matrix-layout-column-major`.** GLM stores column-major
   and the C++ side uploads without transpose; Slang's default is row-major.
   Compiling all shaders column-major makes shader `mul(M, v)` mean exactly
   GLM's `M * v`. Never write per-file matrix-layout pragmas.
2. **Depth range — zero-to-one everywhere.** Vulkan NDC depth is [0,1]; GL was
   [-1,1]. P1 audit status: the software rasterizer already remaps NDC z to
   `z01 = z_ndc * 0.5 + 0.5` (rasterizer.hpp), but the depth-motion path overwrites
   it with *linear* view-z — two meanings of "depth01" to unify. Then pin
   `GLM_FORCE_DEPTH_ZERO_TO_ONE` (or an
   explicit `shs_perspective_zo()` shared by both backends) so one projection
   source feeds both. No GL-style depth survives P1. Until then,
   `conventions.md` §2/§5 ([−1,1] projection + explicit remap) is the shipped truth.
3. **Framebuffer Y origin — negative viewport height.** Standard Vulkan
   negative-height viewport keeps world/clip-space math identical to the
   software path; no per-shader Y flips. Verify the existing monolith viewport
   setup and conform — don't fork a second convention.
4. **Winding — pass-authored, contract-checked.** `RHIFrontFace` already maps to
   `VK_FRONT_FACE_*` in `vk_pipelines.hpp`. Correction over the earlier draft:
   with negative viewport height, Vulkan evaluates facing in *framebuffer* space,
   so the screen winding is inverted — the VK `frontFace` must be the **opposite**
   enum from the software rasterizer's canvas-space sense. The current identity
   mapping (`vk_device.hpp`) is only safe while culling is disabled; resolve at P1
   with a dual-backend culling parity test (see `conventions.md` §5). Any pass
   flip must be stated in its pass contract, not fixed in shader code.
5. **GLSL → Slang type mapping:** `vec3`→`float3`, `mat4`→`float4x4`,
   `mix`→`lerp`, `fract`→`frac`, `texture()`→`SampleLevel` (explicit LOD),
   `textureLod`→`SampleLevel`. GLSL `mod` is positive-mod; HLSL `fmod` differs in
   sign — write the positive variant explicitly. `gl_FragCoord` → `SV_Position`
   (same +0.5 pixel-center convention on Vulkan).
6. **Buffer layout — vec4-only shared structs.** The pod convention (all-vec4
   members, `alignas(16)`, `static_assert(sizeof % 16 == 0)`) already dodges
   std140/std430/`cbuffer` packing traps (`float3` members, array strides).
   Rule: any C++↔Slang shared struct contains only `float4`/`uvec4`-style
   members. `StructuredBuffer<T>` gives std430-comparable rules for arrays.
7. **Bindings — explicit `[[vk::binding(set, binding)]]`.** No automatic
   assignment; the P2 pod driver emits descriptors from reflection JSON, so
   determinism beats convenience.

## 3. Phase plan

### P0 — Toolchain + gate (GPU-free-safe)
- `find_program(SLANGC slangc)` in root CMakeLists; `SHS_HAS_SLANG` gate
  symmetric to `SHS_HAS_VULKAN`.
- Custom target `shs_slang_shaders`: compiles `execution/shader/slang/**.slang`
  → `build/shaders/slang/<pass>_<stage>.spv`, one `.spv` per entry point;
  BYPRODUCTS + DEPENDS on all `.slang` files for incremental builds.
- Pinned flags in one CMake variable:
  `-matrix-layout-column-major -profile spirv_1_5` (SPIR-V 1.5 ↔ Vulkan 1.2).
- Validation: main config builds + ctest 5/5;
  `-DCMAKE_DISABLE_FIND_PACKAGE_Vulkan=TRUE` still configures and compiles
  shaders (slangc needs no SDK).

### P1 — Spike: common modules + minimal vertical slice
- Zone: `shs/execution/shader/slang/` (linter forbids `shs/shader/` as a domain).
- Modules (real Slang `module`s):
  - `shs.core.math` — ported from frozen `common/math.glsl` + `light_math.glsl`
  - `shs.lighting.types` — `CullingLightGPU` (all-vec4, mirrors
    `light_types.hpp`; C++ static_assert mirror kept until P2)
  - `shs.lighting.constants` — from frozen `common/light_constants.glsl`
- Pass modules with entry points: `default_sky` (vs+fs) — smallest full loop:
  slang → `.spv` → `vk_shader_utils` load → pipeline.
- **Shader manifest value desc** (dual-realization design): `ShaderId` →
  `{ slang_module, entry_points, cpp_impl /* ShaderProgram factory */ }`.
  Pass handlers resolve a `ShaderBinding` per backend: software resolves
  `cpp_impl` → C++ `ShaderProgram`; Vulkan resolves `.spv`. Recipes stay
  backend-blind.
- GLSL ref policy: port *semantics* from frozen `.glsl`, cite file+lines in
  comments, never include them.

### P1.5 — Simpler rendering pipelines start here
- New `RenderPathRecipe` preset `minimal_forward`, backend-parameterized like
  the existing factories:
  `pass_chain = { PBRForward (required), Tonemap (required) }` — no shadows, no
  culling, no light volume. Deliberately the simplest recipe in the tree; new
  recipes graduate from it.
- Slang modules: `minimal_scene` (vs+fs), `tonemap` (fs + fullscreen vs).
- Software-side `cpp_impl` for the same passes reuses the existing
  `pass_pbr_forward`/`pass_tonemap` handlers — first proof that the *same
  recipe* executes on both substrates with shader identity as data.

### P2 — Reflection → pod driver
- `slangc` reflection JSON → `build/shaders/slang/<pass>.json`.
- Generate/validate value descs for `vk_pipelines.hpp`/`vk_resources.hpp`:
  descriptor sets, push-constant ranges, struct layouts checked against C++
  pods at build time.
- `CullingLightGPU` single-source: slang module is GPU-layout truth; the C++
  static_assert mirror becomes a *verified* mirror (reflection diff in a test),
  not a hand-maintained hope.
- Caps → specialization constants: `BackendFeatureCaps`/`BackendLimitCaps` and
  recipe knobs flow into pipeline specialization.

### P2.5 — Cross-backend equivalence harness (the drift killer)
- Headless test: render one frame of `minimal_forward` through both backends;
  compare outputs with RMSE tolerance (GPU interpolation ≠ exact software
  sampling — tolerance, not equality; per-pass threshold table).
- Start with `tonemap` (pure per-pixel → near-exact) and `default_sky`, then
  `PBRForward`. Light-culling compute results compare as sets (CPU reducer vs
  GPU dispatch must produce the same culling PODs — order-insensitive compare).
- This test is what makes "pluggable, backend-blind" an invariant instead of
  an aspiration.

### P3 — Migrate & delete
- Port remaining passes (shadow_map, depth_prepass, light_culling, gbuffer,
  deferred, motion_blur, ssao, dof, fxaa/composite, sky variants) citing the
  frozen GLSL as reference.
- Delete: `shaders/vulkan/**.glsl`, `glslangValidator` CMake blocks, parked-demo
  shader targets. The glslang path dies with them.

## 4. Validation gates per phase
- ctest stays green (5/5 baseline; equivalence tests add to it).
- GPU-free configure always works (`slangc` has no SDK dependency).
- `SHS_HAS_SLANG=0` build stays fully green: shader assets are optional; the
  Vulkan edge degrades exactly as it does today without Vulkan.
- No new GLSL at any phase (policy §0).

## 5. Open decisions (resolve at P1, not before)
1. Entry-point naming: `vs_main`/`fs_main` per module, one file per pass —
   chosen unless reflection JSON shows tooling friction.
2. Reflection extraction: CLI JSON dump (P2 bootstrap) vs slang API (richer);
   start CLI, migrate only if the driver needs types the JSON lacks.
3. Push-constant budget: `ShaderUniforms` is a grab-bag; only the handle-based
   pod redesign (separate track) settles push constants vs parameter blocks.

