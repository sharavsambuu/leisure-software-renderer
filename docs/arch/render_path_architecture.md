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
Edit `shs/renderpath/planning/render_technique_presets.hpp`:
1.  Add enum to `RenderTechniquePreset`.
2.  Map it to a shading model and shader variant in `render_technique_shader_variant(...)`.
3.  Define its default setup in `make_builtin_render_technique_recipe(...)`.

### Adding a New Render Path
Edit `shs/renderpath/planning/render_path_presets.hpp`:
1.  Add enum to `RenderPathPreset`.
2.  Define its unique `TechniqueMode` and default culling strategies.
3.  Define the default pass chain in `make_builtin_render_path_recipe(...)`.

### Adding a New Pass

**Core pass** (ships in the library):
1.  **Register ID**: Add a value to `PassId` in `shs/renderpath/planning/pass_id.hpp`.
2.  **Define Contract**: Add its input/output semantic requirements in `shs/renderpath/planning/pass_contract_registry.hpp`.
3.  **Implement Handler**: Register a dispatch handler in the backend (e.g., `vk_render_path_pass_context.hpp` or `shs_renderer_lib.cpp`).

**Consumer / demo-owned pass** (no core edit — Constitution I §7, graduation
requirement 1, shipped 2026-09-18):

1.  **Mint the id**: `PassFactoryRegistry::intern_pass_id("demo_fog_pass")` →
    `std::optional<PassId>` in the open registered range. Builtin names resolve
    to their builtin id (a core pass can never be shadowed), an already-minted
    name is idempotent, and a name-hash collision is refused loudly (no
    overwrite) — `nullopt` means "pick another name".
2.  **Register factory + descriptor** under that same name:
    `register_factory(id, "demo_fog_pass", factory)` and
    `register_descriptor(id, contract, backend_mask)`. The descriptor hints are
    what make the pass planner-visible (VOP-first rule: no hints, no planner
    participation), and they are exactly what builtins register.
3.  **Reference it from a recipe**: `make_render_path_pass_entry("demo_fog_pass",
    id, /*required=*/true)`, then compile through the same Recipe → Compiler →
    Plan pipeline as a builtin. Execution keys on the registered *name*, so
    replay logs, barrier tables and saved recipes stay stable.
4.  **Verify**: `render_path_plan_has_pass(plan, id)` — with the registry
    overload when the plan was authored string-keyed — answers for a consumer
    pass exactly as it does for a builtin.

No core edit, no enum change, no planner fork; the proof is the
`shs_renderer_pass_id_open_tests` gate.

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

Vision: the core renderer satisfies *every class* of complex rendering technique —
materials, lighting, light culling, compute-based effects up to UE5-class
complexity — by being a **small closed vocabulary + open registries**, with
consumer complexity added as *additive abstractions* built on the Domain Value Object
concepts (value descs, contracts, recipes — never by editing the core).

Already first-class in the value vocabulary:
- **Compute** — `RHICmdDispatchDesc` + compute pipeline descs + compute queue
  class + `async_compute` capability: compute-based effects (SSAO-class post,
  particle sims, light propagation, voxelization) are ordinary passes, not
  special cases.
- **Culling / light volumes** — culling strategies are recipe data; light-grid
  and cluster structures are canonical semantics, not hardcoded pass internals.

Graduation requirements (tracked; reqs 1, 4, 5, 6 and 7 have since shipped; reqs
2, 3 and 8 are open, and this list was amended 2026-09-18 to add reqs 7–8 after
an audit found two consumer-authorable axes it had never named — see "Blind
spot" below):

1. **Open pass IDs** — ✅ **DONE 2026-09-18.** `PassId` keeps its builtin
   vocabulary and gains an open registered range: the range law lives in
   `shs/renderpath/planning/pass_id.hpp` (builtin 1–1023, open 1024–65534,
   65535 reserved, plus `pass_id_is_builtin` / `pass_id_is_open` /
   `pass_id_in_valid_range`; the legacy `pass_id_is_standard` stays
   behavior-identical for every value the enum can hold) and the registry in
   `shs/renderpath/execution/pass_id_registry.hpp`. A demo registers a pass
   *name*, gets a typed id that is a stable function of that name
   (content-addressed, so it is identical across registries, translation units
   and processes — and registration order can never leak into a plan),
   registers its factory/descriptor through the same API builtins use, and
   references it from a recipe: **zero core edits**. Builtin names always
   resolve to their builtin id, so a consumer can never shadow a core pass;
   collisions are refused loudly, and the verified `(id, name)` registration
   path makes a foreign or colliding id a hard miss rather than a wrong pass.
   Evidence + gates: `docs/backlog/open_pass_id_registry_evidence_2026-09-18.md`
   (`shs_renderer_pass_id_open_tests`, 12 GPU-free checks; full CTest 72/72).
   Residual, stated: the open range is 16-bit because `PassId` is — 64,511
   consumer slots, one pass per name-hash slot. Widening the id type is a
   separate, gated decision if a consumer ever needs past that.
2. **Open light/light-volume registries** — `RenderPathLightVolumeProvider` and
   the builtin light structs in `shader/types.hpp` are closed enums today;
   custom light abstractions (area/IES/volumetric/custom game lights) require
   the same open-registry treatment, with light *types* additive on top of the
   shared lighting math library. *Not started; `PassIdRegistry` is the shape to
   reuse (rule of two).*
3. **Material graph compiler** — complex material authoring follows the
   material-system roadmap (lib → assembler → node graph → multi-target
   emission: GLSL/Slang/C++), not core enumeration.
4. **Substrate resolution at plan time** — the *substrate* (software / GL /
   Vulkan) is chosen at *authoring* time today: `render_path_recipe.hpp` forks
   both the pass chain and the technique mode on `backend`, so the
   forward-plus recipe and the everything-else recipe are two different
   recipes rather than one recipe resolved two ways. The requirement: a recipe
   states *intent* (what must happen, and a policy such as device-preferred /
   host-preferred / cheapest / exact-match), and substrate becomes a
   **resolution output** of the compiler. `frame_graph.hpp` already plans
   cross-backend chains and `IRenderPass::is_interop_pass()` already names the
   handoff case, so the plan side is the smaller half.
   **Landed 2026-09-18 (code in tree, not a plan).** The pure half is a new
   planning leaf, `substrate_resolution.hpp`: `SubstratePolicy { exact_match,
   device_preferred, host_preferred, cheapest }`, a `SubstrateResolutionRequest`
   (intent + realizability mask + admissible mask + policy + predecessor) and
   `resolve_substrate`, which returns a substrate or *unresolved* — never a
   guess. Nothing in it touches a context, a backend or a registry. The choice is
   made by an explicit preference ladder, so the same request always yields the
   same substrate: a plan is data, not ambient state.
     - **Recipe states intent.** `RenderPathPassEntry::domain` carries a per-pass
       `RenderDomain` (`with_domain` sets it), and
       `RenderPathRecipe::substrate_policy` names the policy. `backend` is
       demoted to the *declared* substrate: the `exact_match` target, and the
       fallback when the host declares nothing.
     - **Compiler outputs the resolution.** `RenderPathCompiledPass::substrate` /
       `substrate_resolved` are the answer per pass; the plan echoes
       `substrate_policy` and sets `hybrid` when an accepted crossing occurred.
     - **Admissibility is an explicit opt-in.**
       `RenderPathCapabilitySet::available_substrate_mask` defaults EMPTY, read
       as "unknown / single-substrate host" and resolved as the declared
       substrate only. Every pre-RP-1 caller therefore resolves exactly where it
       did — measured, not asserted: the whole suite was green before a single
       new test existed. Widening the mask is what lets one recipe resolve two
       ways.
     - **The gate that had to move.** The old check asked "does this pass support
       the recipe's declared backend?" — a question that *forbids* the
       substitution req 4 introduces, since a device-only pass could never join a
       software-declared recipe. It now asks "does this pass intersect the
       admissible set?", which on a single-substrate host IS the declared
       substrate and so answers identically (same verdict, same
       `BackendUnavailable`). Only a host that declares more than one substrate
       can see a difference.
     - **Hybrid legality at plan time.** `PassFactoryDescriptor::declares_interop`
       (with `realized_substrate_mask_hint` / `declares_interop_hint`) supplies
       the declared boundary; two adjacent passes resolving onto different
       substrates is `HybridBoundaryUndeclared` unless one of them declares it,
       mirroring `frame_graph.hpp`'s pair rule. Shared staging stays the half
       enforced where resources are materialized.
   Verification: **73/73** CTest gates green, including the
   `SHS_CONTRACTS_ENFORCED=1` guardrail targets. Five gates pin it in
   `shs_renderer_renderpath_tests` — `substrate_policy_resolution`,
   `substrate_intent_binds_policy`, `substrate_hybrid_legality`,
   `substrate_resolution_snapshot_contract`, and
   `registry_single_recipe_two_substrates` — and all five were prove-failed
   rather than trusted: disabling the crossing rejection fails
   `substrate_hybrid_legality`; disabling intent narrowing fails
   `substrate_intent_binds_policy` and the snapshot gate; forcing the registry's
   policy to `exact_match` fails the registry gate on the device host; and
   restoring the fork's two-recipe shape fails it on the recipe count. Inventory
   delta attributable to this change: +1 header (`header_count` 229 → 230) plus
   its graph edges.
   **The authoring-time fork is now REMOVED (same day).**
   `make_default_soft_shadow_culling_recipe(backend)` — which authored two
   different pass chains *and* two different technique modes for one intent, and
   so was the last place where the substrate chose the *shape* of the path — is
   deleted. All four references migrated: `RenderPathRegistry::register_default_recipes()`
   now registers ONE recipe (`soft_shadow_culling`, `device_preferred`);
   `renderpath.contract.hpp` re-exports the unified maker along with the
   `SubstratePolicy` / `RenderDomain` / `with_domain` authoring vocabulary; and
   `hello_soft_shadow_culling_vk.cpp` declares its substrate and lets the policy
   resolve — plan-identical, because with no advertised substrate set
   admissible == {declared} == {Vulkan} and the chain was already the fork's
   Vulkan branch. Nothing looked the two old names up, so the migration is a
   rename plus a deletion. Full suite re-verified **73/73** green after it.
   Residuals, stated: (1) The authoring-time substrate choice survives one tier
   up, at **composition** resolution: the demos call
   `resolve_builtin_render_composition_recipe(..., RenderBackendType::Vulkan, ...)`
   and their parity harness obtains the software plan by cloning the resolved
   recipe and flipping `backend` (`demo_forward_classic_renderpath.cpp:1080-1085`;
   `hello_rendering_paths.cpp:1010-1015`) against a second, substrate-flavored
   `pass_contract_registry_sw_`. That clone is a two-resolution *comparison* —
   same chain, same technique mode, already value-shaped — not the authoring fork
   this requirement removed; making it policy-driven rather than field-flipped
   belongs with the contract-registry axis split in req 5. Both files are Vulkan
   targets and are **not built** in the software-only configuration
   (`exps-gpu-renderer` is commented out at `cpp-folders/CMakeLists.txt:120`), so
   that migration cannot be compile-verified here — stated, not attempted.
   (2) The builtin registry registers every standard pass
   **software-only**, so a device-preferring policy over the builtin table still
   resolves to software; a device realization exists only where a descriptor
   declares one (the gates author one deliberately). (3) `cheapest` is greedy
   over the authored chain order — a crossing count, not a cost model, because no
   honest per-pass cost data exists to build one from. (4) `plan.backend` remains
   the *declared* substrate; the resolved primary is
   `pass_chain.front().substrate`. (5) The plan tier enforces only the
   declared-boundary half of the hybrid rule.
   Residual, stated: techniques stay a function of technique mode only (the
   Rule 17 value invariant), and the plan's snapshot-equality contract
   (`operator== = default`) must keep holding across resolutions — a
   resolution is data, never ambient state.
5. **Execution-unit × substrate, as two axes** — `ContractDomain`
   (`planning/pass_contract.hpp`) and `PassResourceDomain`
   (`execution/render_pass.hpp`) are two enums with **identical value sets**,
   and the relation between them collapses the axes (`host → software raster`,
   `device → {GL, Vulkan}`). That conflation forces "runs on the host" to mean
   "is the software rasterizer", which cannot express host-assisted *device*
   work at all. The requirement: one vocabulary with two independent axes —
   where the pass *executes* vs which *substrate* it targets — an explicit
   handoff relation between them, and one enum retired rather than two kept in
   sync by hand.
   **RULED 2026-09-18 (owner), final:** two independent axes —
   `ExecutionUnit { host, device }` × `Substrate { software_raster, opengl,
   vulkan }`; `RenderBackendType` maps to exactly one `Substrate`;
   `PassResourceDomain` is **retired** rather than kept in sync by hand.
   **Hybrid legality rule:** a chain crossing execution units or substrates is
   legal only where the crossing passes declare an interop boundary
   (`IRenderPass::is_interop_pass()`) *and* share a declared staging resource;
   otherwise a cross-substrate mismatch is a **rejection**, not the warning it
   is today (`frame_graph.hpp:132-139`). Cutover mechanics precedent:
   `docs/backlog/namespace_cutover_mapping.md`.
   Measured migration surface (2026-09-18 — size, not guessed):
   `ContractDomain` carries **three dead values** (`CPU` 0 uses, `OpenGL` 0,
   `Vulkan` 0); live are `Any` (11), `GPU` (59 — every one inside
   `pass_contract_registry.hpp`, i.e. the default descriptor table) and
   `Software` (62 — almost all `pass_adapters.hpp`, i.e. the
   software-realization overrides). `PassResourceDomain::CPU/GPU/OpenGL/Vulkan`
   appear **only** inside `render_pass.hpp`'s own three switches (lines
   146-179), nowhere else; externally only `Any` (7) and `Software` (68) are
   used. This is therefore a **mechanical re-annotation of two uniform literals
   plus a three-value deletion**, not a semantic rewrite. The one load-bearing
   site is `ContractDomain::Software`, which today means *both* "host
   execution" *and* "software realization" — those two meanings must separate.
   **Landed 2026-09-18 (code in tree, not a plan):** both old enums are deleted
   — `ExecutionUnit`, `Substrate`, `RenderDomainKind` and `RenderDomain` live in
   `planning/pass_contract.hpp`, and all 216 annotation sites were re-labelled
   mechanically to `render_domain_host()` / `render_domain_device()`; the old
   identifiers appear nowhere outside this rationale. Two properties are worth
   recording because they were *verified*, not asserted:
     (a) The refactor alone is semantics-preserving — `render_domains_compatible`
         returns the same boolean as the retired `pass_resource_domains_compatible`
         on **all** input pairs (host vs device was already `false`; the old
         `CPU ≡ Software` clause collapses into "equal units"). The enum split
         changed no plan's verdict by itself.
     (b) The warning→rejection promotion is therefore a *separate, deliberate*
         change, and it landed too: `frame_graph.hpp` pushes to `report_.errors`
         and sets `report_.valid = false` for an undeclared cross-unit write,
         waiving only where a pass declares an interop boundary
         (`is_interop_pass()`); the staging resource is shared by construction,
         since both refs carry the same key.
   `substrate_of_backend` is an ordinal identity (`RenderBackendType` and
   `Substrate` are 1:1) rather than a hand-kept table. Verified: full build
   clean and **73/73** CTest gates green, including the
   `SHS_CONTRACTS_ENFORCED=1` guardrail targets. The *enum split itself* is
   inventory-neutral — the regenerated `engine_header_inventory.json` is
   byte-identical with and without it; the only inventory delta in this change
   is a single `tracked_consumers` line, because the new gate test includes
   `frame_graph.hpp`. Pinned by four gates in `shs_renderer_renderpath_tests` —
   `domain_axes_are_independent`, `domain_compatibility_relation`,
   `domain_backend_matching` and `hybrid_interop_legality`; the last is
   prove-failed (disabling the rejection makes it fail), so the law binds rather
   than merely documents.
   Residual, stated: the current "matches anything" sentinel does double duty
   (wildcard in one call site, unresolved in another); the split must name
   those two meanings separately rather than inherit the ambiguity.
6. **Open shader-identity registry** — **DONE 2026-09-18.** `ShaderId`
   (`render/shader/shader_identity.hpp`) was the remaining identity namespace
   that was still a closed enum with a count sentinel, while `PassId` had had an
   open registered range since 2026-09-18 and the law names technique/light
   presets as the next two. The per-backend *shape* already existed — one
   authored shader, each backend resolving its own realization, loud refusal
   when none exists (P1.5, `shader_identity_manifest_evidence_2026-09-18.md`);
   what was missing was that a **consumer** could not mint one.
   Now: `ShaderId` carries builtin + **open registered** ranges
   (`render/shader/shader_id.hpp`, mirroring `planning/pass_id.hpp`), the mint
   registry is `ShaderIdRegistry` (`render/shader/shader_id_registry.hpp`,
   mirroring `execution/pass_id_registry.hpp`) and is **owned by**
   `ShaderManifest`, so a consumer mints a shader identity from a name
   (`intern_shader` / `register_named_shader`) and resolves it through the same
   backend-blind path a builtin uses — with **zero core edits**. This was the
   third instance of the `PassIdRegistry` shape, so the rule of two invoked at
   req 2 was discharged by *hoisting the mechanism* rather than copying it: the
   content-addressing law now lives in `shs/core/open_id_hash.hpp` and
   `PassIdRegistry::open_offset` delegates to it, so pass and shader ids cannot
   drift. Verified: full build clean, **74/74** CTest (was 73) and **74/74**
   with `SHS_CONTRACTS_ENFORCED=1`; gate
   `shs_renderer_shader_id_open_tests` (14 named cases) plus parity guards in
   `shs_renderer_shader_identity_tests`; **four** mutation probes, one of which
   (collapsing the shared law) fails **both** namespaces' gates. Contract
   placement now spans 232 headers (was 230); `header_count` 230 → 233
   (regenerated, idempotent). Evidence:
   `docs/backlog/shader_id_open_registry_evidence_2026-09-18.md`.
   Residuals, stated: the id space is 16-bit (64,511 open slots, one shader per
   name-hash slot); a refused registration leaves its name minted (the registry
   has no `remove`, exactly like `PassIdRegistry`); `ShaderManifest::operator==`
   is no longer `constexpr` (its open half is `std::deque`/`std::string`); and
   P1.5's past-the-manifest residuals (Vulkan loader input, the GPU-half census,
   P2 reflection) are untouched — this item closed P1.5's "no consumer/open
   shader ids yet" residual and nothing else.


7. **Open `PassSemantic` registry** — ✅ **DONE 2026-09-18** (added and shipped
   the same day; `PassSemanticEncoding` was in the original title but is
   deliberately still closed — it is the attachment-packing axis, discussed in
   "Blind spot" below). `PassSemantic` (`planning/pass_contract.hpp:37`) is a
   **closed 16-value enum** (`Unknown = 0` … `HistoryMotion = 15`) with no
   `Custom` member and no open registered range, so a consumer **cannot** name a
   G-buffer channel the builtins did not anticipate — which is exactly what the
   requirement to build *arbitrary G-buffer layouts* needs. `PassSemanticEncoding`
   is closed in the same way, which additionally means there is no consumer
   vocabulary for **physical attachment packing** (which channels occupy which
   target format/channel). The change surface is small and enumerable: four live
   switches — `pass_semantic_name` (`pass_contract.hpp:253`),
   `default_pass_semantic_descriptor` (`pass_contract.hpp:357`, a 15-case switch
   giving each builtin semantic its space/encoding/lifetime/`sampled`/`storage`
   defaults), `render_path_resource_id_for_semantic`
   (`planning/render_path_resource_plan.hpp:112`) and
   `make_default_resource_spec_for_semantic` (same file, `:160`) — plus two
   debug switches in `exp-plumbing/hello_rendering_paths.cpp` and
   `demo_forward_classic_renderpath.cpp`, which are **not built**
   (`exps-gpu-renderer` commented out at `cpp-folders/CMakeLists.txt:120`).
   Unlike technique modes (`TechniquePassContract::semantics` is a
   `std::vector<PassSemanticRef>`, not a bitmask), semantics open **without**
   a mask-representation change. `PassIdRegistry` is the shape to reuse and the
   content-addressing law is already shared (`shs/core/open_id_hash.hpp`), so
   this is an application of the hoisted mechanism, not a fourth copy of it.

   ✅ **DONE 2026-09-18.** Landed as the same shape as reqs 1 and 6, and the
   three-property argument is now the same three properties (stability across
   processes, order independence, no cross-registry aliasing), so it inherits
   rather than re-derives. What shipped: the range law + pins in
   `pass_contract.hpp` next to the enum; `PassSemanticRegistry`
   (`planning/semantic_registry.hpp`) delegating its offsets to
   `core::open_id_offset`; all **four** switch surfaces given an explicit
   open-range answer; `pass_semantic_name_or_null` and `parse_pass_semantic`
   as the honest builtin-only counterparts; and registry-aware overloads
   `render_path_resource_id_for_semantic(PassSemantic, const PassSemanticRegistry&)`
   and `make_default_resource_spec_for_semantic(semantic, recipe, registry)`.

   Two design rulings worth recording, because the second is the non-obvious
   one:
   - **The core does not guess an open channel's intent.** Open semantics take
     the neutral descriptor defaults (Screen / Linear / Transient / sampled) by
     an explicit early return, and the author states the parts only they know
     through `make_semantic_ref`'s overrides. The early return is guarded on
     `pass_semantic_is_open`, deliberately **not** on `!is_builtin` — so
     `Unknown` keeps its old `None`/`Unknown`/unsampled descriptor verbatim.
   - **A bare resource id cannot distinguish two open semantics.** Without a
     registry both fall back to `kPassSemanticOpenSpelling` and would MERGE
     into one planned resource — a silent, load-bearing collision, worse than
     the equivalent in the pass-id namespace (where ids address factories, not
     resources). Hence the registry-aware overload is the *recommended* path,
     and the gate asserts both halves: distinct names stay distinct ids through
     the registry, and the aliasing failure of the bare spelling is asserted
     rather than hidden.
   Residuals, stated: `PassSemanticEncoding` is **still closed** (see below);
   the two debug switches in `exp-plumbing/hello_rendering_paths.cpp` and
   `demo_forward_classic_renderpath.cpp` were **not** touched because they are
   not built — they only print names, and `pass_semantic_name` is total for
   open ids, so they stay correct-or-honest if and when
   `exps-gpu-renderer` is un-parked. Gate:
   `shs_renderer_semantic_id_open_tests` (12 sub-checks; value-tier, GPU-free,
   added to the suite as test 4-adjacent).
8. **Unify, then open, the render-technique vocabulary** — *not started; added
   2026-09-18.* Two enums describe one axis and disagree:
   `TechniqueMode` (`render/frame/technique_mode.hpp:19` — `Forward`,
   `ForwardPlus`, `Deferred`, `TiledDeferred`, `ClusteredForward`) and
   `RenderPathRenderingTechnique` (`planning/render_path_recipe.hpp:76` —
   `ForwardLit`, `ForwardPlus`, `Deferred`). Both are **closed**, and the
   consequence is not merely stylistic: `RenderPathRecipe::render_technique`
   carries the **3-value** enum and `technique_mode_for()`
   (`renderpath.gateway.hpp:102`) maps exactly those three, defaulting anything
   else to `Forward`. So **`TiledDeferred` and `ClusteredForward` have no
   authoring path from a recipe today** — they are reachable only as a
   `supported_modes_mask` bit a pass may advertise. Retiring one enum (rule of
   two) and making the relation total precedes opening the range, and opening
   additionally forces a **mask-representation decision**: `TechniqueMode` is
   used as a bitmask (`technique_mode_bit` is `1u << value`;
   `supported_modes_mask` / `active_modes_mask` are `uint32_t`), so an open
   range is bounded at 32 unless the mask widens or a side set of open ids is
   added. The same closure blocks lightweight mobile lighting, which needs a
   cheap shading model: `ShadingModel` (`render/frame/frame_params.hpp:130`) and
   `RenderTechniquePreset` (`planning/render_technique_presets.hpp:24`) are a
   **second duplicate pair** on that axis (PBR/Blinn vs PBRMetalRough/BlinnPhong,
   bridged by `render_technique_preset_from_shading_model`), to be unified under
   the same ruling. *Not started.*

Blind spot (stated 2026-09-18): this list scheduled passes (1), lights (2),
materials (3), substrate (4), axes (5) and shader identity (6) — but **never
scheduled semantic vocabulary or technique vocabulary**, even though req 1's
`PassIdRegistry` had already proved the shape and §7.1 of `conventions.md` names
"techniques, materials, lights" as the additive axes. The goals these two
unlisted axes own — *arbitrary G-buffer layouts*, *tiled/clustered/deferred/
forward+*, and *lightweight mobile lighting* — were therefore treated as
satisfied by RP-1/2/3 orchestration when only the *orchestration* half was
built. A closed enum on a consumer-authorable axis is a Lock-In residual
regardless of how much arrangement machinery surrounds it; reqs 7–8 name the two
that were missing. Also unaudited and deliberately left out of this amendment:
**`Substrate`** (three values; a fourth substrate touches RHI and is a larger
decision than an id-range job), the light *type* vocabulary
(`lighting/light_types.hpp:27`, nominally 6 values and richer than req 2
implies), and the **attachment-packing schema** — a *new* abstraction, not an
enum to open, since nothing today states which channels occupy which target
format. Ordering for the new track: req 7 → req 8 → req 2, each gated by a
consumer that mints a name and plans a pass with it; req 7 is independent of
req 8 and may land first. **Reckoning 2026-09-18 (after req 7 shipped):** the
order held — req 7 landed first — but the **gate phrasing did not**, and the
correction matters more than the ordering did. "Gated by a *consumer*" is not
achievable today: the render-path library has **no consumer outside its own
tests** (the active `exps-rendering-adventures` tree links `shs::renderer` only
for a platform seam and includes no `shs/renderpath` header; the three `exps`
trees that might have contained one are parked behind a whole-tree alias/
include-path migration — `cpp-folders/CMakeLists.txt:103-107`, `644be48`). So
req 7's gate is a **library test**, `shs_renderer_semantic_id_open_tests`,
exactly where reqs 1 and 6 put theirs, and the honest statement is that these
axes are proven *in-library* with demo-level proof still deferred. The phrasing
predates knowing that; the three precedents should have made it obvious, since
none of them had an out-of-library consumer either. Restating the gate norm for
the remaining track: **a library gate that exercises the consumer *role*
(minting, declaring, planning, resolving) counts; an end-to-end demo does not
exist yet and is not a blocker for reqs 8 or 2.**

Ordering constraint: RP-2's *vocabulary decision* precedes RP-1's *acceptance
test*. The input RP-1's resolver needs already exists and is **substrate**-keyed,
not execution-unit-keyed — `PassFactoryDescriptor::backend_mask` /
`backend_mask_known` over `RenderBackendType` (`pass_registry.hpp:34`), live and
populated (`pass_adapters.hpp:1532` registers software-only builtins;
`pass_registry.hpp:188` consumes it). So a per-pass resolver can be written
today without touching either domain enum. What *cannot* be written today is the
**hybrid** half: a chain mixing host and device passes is validated by
`pass_resource_domains_compatible` (`render_pass.hpp:169`), which encodes the
conflation (`CPU` ≡ `Software`). The decision that must land first is therefore
narrow — *what must be true for two passes on different substrates to share one
chain* — after which RP-1 can state what it accepts. Sequence: RP-2 ruling (a
decision, not code) → RP-1 → RP-2 code refactor → RP-3.

Not a graduation requirement: **schedule / concurrency as plan data** (a
resolved topological order plus a per-pass concurrency class). Today the plan
carries pass order and essentially one hint; nothing consumes it. Per
Constitution II §11.1 / Rule 12 this is **trigger-gated** — a trigger *and* a
named measurement in `optimization_backlog.md` must exist before any work
starts, and listing it here authorizes nothing.

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
- **Logic**: `shs/renderpath/execution/render_path_executor.hpp`
- **Presets**: `shs/renderpath/planning/render_composition_presets.hpp`
- **Compiler**: `shs/renderpath/planning/render_path_compiler.hpp`
- **Pass registry (open ids)**: `shs/renderpath/execution/pass_registry.hpp`
- **Vulkan Bindings**: `shs/rhi/vulkan/runtime/vk_render_path_descriptors.hpp` *(exists; the `rhi/drivers/` forwarders were retired in `8b6484b`)*
- **Domain Value Object rearchitecture**: `docs/arch/render_path_domain_pod_architecture.md` —
  wraps this pipeline in the Core 4 DVO canon (`domains/renderpath/`: contract =
  recipe/plan types, action = `RenderPathCommand` intents, gateway = `renderpath_gateway`
  with keep-on-reject hot-swap invariant, event = `PATH_COMPILED` / `PATH_SWAP_REJECTED`
  log). Rollout phases in `docs/roadmap/domain_pod_engine_rollout_roadmap.md`.
