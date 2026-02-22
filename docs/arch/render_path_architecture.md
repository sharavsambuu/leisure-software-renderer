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
- **Vulkan Bindings**: `shs/rhi/drivers/vulkan/vk_render_path_descriptors.hpp`
