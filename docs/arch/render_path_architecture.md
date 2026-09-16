# Render Path Architecture

The SHS Renderer uses a data-driven, compositional architecture for its rendering pipelines. This allows the engine to switch between different architectures (Forward, Forward+, Deferred, Clustered) and techniques (PBR, Blinn-Phong) at runtime using **Recipes**.

---

## 1. Core Architecture Pattern

The system follows a "Compile-then-Execute" flow to decouple demo logic from backend recording:

1.  **Recipe (`RenderPathRecipe`)**: A declarative description of the desired pipeline (What passes? What resolution? What culling?).
2.  **Compiler (`RenderPathCompiler`)**: Validates the recipe against backend capabilities and semantic rules.
3.  **Execution Plan (`RenderPathExecutionPlan`)**: The resolved sequence of passes.
4.  **Resource Plan (`RenderPathResourcePlan`)**: Maps semantic requirements (e.g., "Albedo", "Normal") to concrete targets and bindings.
5.  **Barrier Plan (`RenderPathBarrierPlan`)**: Automates synchronization and image layout transitions based on the resource usage timeline.
6.  **Dispatcher (`RenderPathPassDispatcher`)**: Executes the plan by calling registered "Handlers" for each pass.

---

## 2. Resource Semantics & Validation

Passes do not request specific "Textures"; they request **Semantics**. This prevents hardcoding G-buffer layouts.

### Canonical Semantics
- `Albedo / Normal / Material`: Deferred shading channels.
- `Depth`: Depth/Stencil buffer.
- `LightGrid / ClusterIndices`: Culling data structures.
- `Velocity`: Motion vectors for temporal effects.
- `AmbientOcclusion`: Result of SSAO/HBAO passes.

### Validation Metadata
The compiler ensures that producers and consumers match on:
- **Space**: `Screen`, `View`, `Light`, or `Tile`.
- **Encoding**: `Linear`, `sRGB`, `Depth`, `Velocity`, etc.
- **Lifetime**: `Transient` (reused memory) or `Persistent`.
- **Temporal Role**: `Current` vs. `History`.

---

## 3. Extension Guide (How to add features)

### Adding a New Technique
Edit `shs/pipeline/render_technique_presets.hpp`:
1.  Add enum to `RenderTechniquePreset`.
2.  Map it to a shading model and shader variant in `render_technique_shader_variant(...)`.
3.  Define its default setup in `make_builtin_render_technique_recipe(...)`.

### Adding a New Render Path
Edit `shs/pipeline/render_path_presets.hpp`:
1.  Add enum to `RenderPathPreset`.
2.  Define its unique `TechniqueMode` and default culling strategies.
3.  Define the default pass chain in `make_builtin_render_path_recipe(...)`.

### Adding a New Pass
1.  **Register ID**: Add a value to `PassId` in `shs/pipeline/pass_id.hpp`.
2.  **Define Contract**: Add its input/output semantic requirements in `pass_contract_registry.hpp`.
3.  **Implement Handler**: Register a dispatch handler in the backend (e.g., `vk_render_path_pass_context.hpp` or `shs_renderer_lib.cpp`).

### Demo-Authored Techniques (Technology-Demo Story)
The core ships small, complete renderer cores (e.g., the builtin Blinn-Phong /
PBR technique recipes) plus the shared abstractions — culling, light volumes,
pass contracts, temporal core. The *showcase* techniques (Deferred, Forward+,
Tiled Deferred, Clustered Forward+) are demonstrated through demos, which are
first-class users of the same extension points — no core edits required:
1.  Register a demo-owned `PassId` and its semantic contract.
2.  Author the pass logic (GLSL/Slang shader or C++ software shading).
3.  Register the dispatch handler alongside the builtin ones and reference it
    from a demo recipe.

This keeps the maturity rule ("presets are not demo-specific code paths") true
for the *core*: demo experiments graduate into builtin presets only after they
prove out in a demo, via the same Recipe → Compiler → Plan pipeline.

## 4. Long-Term Extensibility Contract

> Constitutionalized as **Constitution I §7 — No User Lock-In (Pluggability)**
> (`docs/spec/conventions.md`). This section is the formal extension-point
> contract that law refers to.
## 4. Long-Term Extensibility Contract

Vision: the core renderer satisfies *every class* of complex rendering technique —
materials, lighting, light culling, compute-based effects up to UE5-class
complexity — by being a **small closed vocabulary + open registries**, with
consumer complexity added as *additive abstractions* built on the Domain POD
concepts (value descs, contracts, recipes — never by editing the core).

Already first-class in the value vocabulary:
- **Compute** — `RHICmdDispatchDesc` + compute pipeline descs + compute queue
  class + `async_compute` capability: compute-based effects (SSAO-class post,
  particle sims, light propagation, voxelization) are ordinary passes, not
  special cases.
- **Culling / light volumes** — culling strategies are recipe data; light-grid
  and cluster structures are canonical semantics, not hardcoded pass internals.

Graduation requirements (tracked, not yet built):
1. **Open pass IDs** — `PassId` is a closed 16-value enum; consumer/demo-owned
   passes need a builtin range + open registered range (or stable-string-hash
   contract keys). Blocking for the demo-authoring story above.
2. **Open light/light-volume registries** — `RenderPathLightVolumeProvider` and
   the builtin light structs in `shader/types.hpp` are closed enums today;
   custom light abstractions (area/IES/volumetric/custom game lights) require
   the same open-registry treatment, with light *types* additive on top of the
   shared lighting math library.
3. **Material graph compiler** — complex material authoring follows the
   material-system roadmap (lib → assembler → node graph → multi-target
   emission: GLSL/Slang/C++), not core enumeration.

UE5-class features (virtualized geometry, GI/VCT/Lumen-class, compute VFX) are
tracked in their own roadmaps (`global_illumination_roadmap.md`,
`angstrom_era_virtual_spu_roadmap.md`, `vulkan_modernization_roadmap.md`,
`future-vulkan-features.md`) — this contract is what lets them land as
*additions* to the core rather than rewrites of it.

---

## 4. Current Implementation Status (L4 Maturity)

The system is currently at **L4 Maturity**, meaning:
*   Pass orchestration is library-owned (managed by the Dispatcher).
*   Host demos (like `HelloRenderingPaths`) are thin wrappers that only register handles and set up the scene.
*   Resource allocation and barriers are derived from the graph plan.
*   **Gap**: Some pass internal implementations are still interleaving demo-specific logic; final maturity goal is to move all "Common" pass bodies into the shared library.

---

## 5. Key Files
- **Logic**: `shs/pipeline/render_path_executor.hpp`
- **Presets**: `shs/pipeline/render_composition_presets.hpp`
- **Compiler**: `shs/pipeline/render_path_compiler.hpp`
- **Vulkan Bindings**: `shs/rhi/drivers/vulkan/vk_render_path_descriptors.hpp` *(planned — driver does not exist yet; see below)*
- **Domain Pod rearchitecture**: `docs/arch/render_path_domain_pod_architecture.md` —
  wraps this pipeline in the Core 4 Domain Pod canon (`domains/renderpath/`: contract =
  recipe/plan types, action = `RenderPathCommand` intents, gateway = `renderpath_gateway`
  with keep-on-reject hot-swap invariant, event = `PATH_COMPILED` / `PATH_SWAP_REJECTED`
  log). Rollout phases in `docs/roadmap/domain_pod_engine_rollout_roadmap.md`.
