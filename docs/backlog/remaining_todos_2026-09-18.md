# Remaining Todos — Consolidated Tracker (2026-09-18)

> Status: **active (2026-09-18)** — created during the 2026-09-18 docs sweep so
> all remaining work is scannable in one place. This file is schedule, not law
> (Constitution II §2.2) and owns no task canonically: each item links its owner
> backlog, which remains the single source of truth. An item may only be ticked
> here in the same commit its owner-doc entry closes, with the same DoD
> evidence. Verification after every code item: full `build/` CTest green +
> `check_kdba_boundaries.sh` green + `check_include_graph.py` green; a touched
> header regenerates the inventory in the same commit (`inventory_headers.py
> --write`, Rule 15). Provenance: compiled after the R-series rasterizer
> hot-path close-out (backlog archived at
> [`docs/outdated/rasterizer_hot_path_todo.md`](../outdated/rasterizer_hot_path_todo.md))
> and the governance-review close-out (archived same day); CTest baseline
> **50/50** at commit `1f8400f`.

## State snapshot (2026-09-18)

- Library migration stabilized (KDBA Runs A–C closed, `05bb646`).
- Rasterizer hot-path track closed & archived (R1–R5); governance review
  closed & archived (G1–G3); `rhi/drivers/vulkan/` forwarders retired (`8b6484b`).
- **Correction (2026-09-18, owner ruling — FINAL): the previous snapshot line
  ("only unblocked work = Bucket A") was wrong.** The owner ruled that the DVO
  architecture — domain-context-separated Domain Value Objects, monadic (Kleisli)
  gateway composition, contract guardrails — is final and binding. The library-side
  programs are *closed* (domain-separation migration archived; namespace cutover
  0 open; contract guardrails C0–C3 done with only C4.3 toolchain-blocked;
  enforcement W-A…W-E 0 open; KDBA A–C, G1–G4 closed), and the genuine library
  remainder is **A0** below. Bucket A's demo track is *not* the only startable work;
  it was believed to be because
  [`domain_pod_engine_rollout_roadmap.md`](../roadmap/domain_pod_engine_rollout_roadmap.md)
  was stale (its P0.5 tree record contradicted the live tree, its P3 boxes 1–3 were
  done-unticketed, and its parked backlog was superseded). That file now carries a
  reconciliation banner.

## Bucket A — Unblocked, ready to start

### A0. Library DVO-architecture remainder (verified 2026-09-18 — owner: [`domain_pod_engine_rollout_roadmap.md`](../roadmap/domain_pod_engine_rollout_roadmap.md) reconciliation banner)

- [ ] **P3 box 4 — monolith → thin DVO composition.** `demo_forward_classic_renderpath.cpp`
      is 9,800 lines against a <1.5k DoD; its extracted pieces already exist and are
      CTest-gated (`demo_input_actions.hpp`, `demo_renderpath_bridge.hpp`,
      `demo_frame_planner.hpp`), so the work is wiring + deleting inline copies.
      The largest remaining structural win, and it is the proof-by-consumer for the
      whole DVO spine. Library-adjacent, not library-internal.
- [x] **P3 box 6, first half — open pass IDs** (Constitution I §7). **CLOSED
      2026-09-18:** `PassId` range law (`planning/pass_id.hpp`) + content-addressed
      `PassIdRegistry` (`execution/pass_id_registry.hpp`); a consumer pass is
      first-class with zero core edits — verified plan participation, typed plan
      queries, factory/descriptor registration under a verified `(id, name)`
      pair, loud collision refusal, no cross-registry aliasing. Gate
      `shs_renderer_pass_id_open_tests` (12 GPU-free checks); full CTest **72/72**
      (was 71); evidence
      [`open_pass_id_registry_evidence_2026-09-18.md`](open_pass_id_registry_evidence_2026-09-18.md).
      Independently corroborated by the 2026-09-18 Antigravity library review.
- [x] **Slang-plan P1.5 — shader identity as data (manifest core).** **CLOSED
      2026-09-18:** one `ShaderId` names one authored shader and each backend
      resolves its own realization — software yields a `ShaderProgram`, Vulkan
      yields module + entry points, and a backend with no realization reports
      `BackendNotRealized` instead of falling back. `ShaderManifest` is
      caller-owned/copyable/comparable (no ambient state), registration is
      verified `(id, name)` pairing, and the software offscreen realization's
      entry-name law now resolves through it (its duplicated literals are gone).
      Gate `shs_renderer_shader_identity_tests` (83 assertions, GPU-free,
      verified against the authored `offscreen_pipeline.slang`); full CTest
      **73/73**; contract-placement coverage 226 → **228** headers; three
      mutation probes failed the gate and were reverted byte-identically.
      Evidence
      [`shader_identity_manifest_evidence_2026-09-18.md`](shader_identity_manifest_evidence_2026-09-18.md).
      Residuals stated there: "no consumer/open shader ids yet" — **since closed**
      2026-09-18 by the open shader-identity registry entry below; the Vulkan
      binding is descriptive truth, not yet the loader's input (P2); the value
      builtins are registered software-only on purpose, making the
      dual-realization gap a census rather than invisible debt. The OpenGL
      selection note in `backend_factory.hpp` was corrected in the same commit —
      the identity layer now refuses OpenGL by data, where before selecting it
      silently ran software.
- [x] **P3 box 6, third instance — open shader IDs** (Constitution I §7; arch §4
      req 6). **CLOSED 2026-09-18:** `ShaderId` range law
      (`render/shader/shader_id.hpp`) + content-addressed `ShaderIdRegistry`
      (`render/shader/shader_id_registry.hpp`), **owned by** `ShaderManifest`, so
      a consumer mints a shader identity from a name
      (`intern_shader` / `register_named_shader`) and resolves it through the
      same backend-blind path a builtin uses, with zero core edits. The rule of
      two was discharged by **hoisting** the offset law to
      `shs/core/open_id_hash.hpp` (`PassIdRegistry::open_offset` now delegates),
      so the two namespaces cannot drift; two new named refusals
      (`UnregisteredOpenId`, `NameCollision`) name the two genuinely new failure
      modes. Gate `shs_renderer_shader_id_open_tests` (14 GPU-free cases) +
      parity guards in `shs_renderer_shader_identity_tests`; full CTest **74/74**
      (was 73) and 74/74 under `SHS_CONTRACTS_ENFORCED=1`; contract placement
      228 → **232** headers; four mutation probes, one of which fails **both**
      namespaces' gates. Evidence
      [`shader_id_open_registry_evidence_2026-09-18.md`](shader_id_open_registry_evidence_2026-09-18.md).
      Residuals stated there: the id space is 16-bit; a refused registration
      leaves its name minted; `ShaderManifest::operator==` is no longer
      `constexpr`; P1.5's other residuals are untouched; and no shipping consumer
      uses an open shader id yet (the gate proves the capability).
- [ ] **P3 box 6, second half — open light/technique registries.**
      `RenderPathLightVolumeProvider` + the technique/light preset enums get the
      same treatment (rule of two: reuse the shape, and call the ONE shared
      offset law in `shs/core/open_id_hash.hpp` — do not add a third copy).
      Unblocked, pure library API work.
- [x] **RP-1 Substrate resolution at plan time** (owner:
      [`render_path_architecture.md`](../arch/render_path_architecture.md) §4
      graduation req 4). The recipe stops choosing the backend: per-pass
      intent + a policy (device-preferred / host-preferred / cheapest /
      exact-match) decide the substrate, so one recipe resolves to software,
      device, or a hybrid instead of forking into two recipes at authoring
      time. Reuses `frame_graph.hpp`'s existing cross-backend chain support;
      must preserve the plan's `operator==` snapshot contract. **DONE
      2026-09-18** — full suite **73/73** green including the
      `SHS_CONTRACTS_ENFORCED=1` guardrail targets.
      What landed: a pure planning leaf `planning/substrate_resolution.hpp`
      (`SubstratePolicy`, `SubstrateResolutionRequest`, `resolve_substrate` —
      intent + realizability mask + admissible mask + policy → a substrate or
      *unresolved*, never a guess, with explicit preference ladders so the same
      request always resolves the same way); `RenderPathPassEntry::domain` +
      `with_domain` as the per-pass intent and
      `RenderPathRecipe::substrate_policy` as the policy, with `backend`
      demoted to the *declared* substrate; `RenderPathCompiledPass::substrate` /
      `substrate_resolved` as the per-pass answer and `plan.substrate_policy` /
      `plan.hybrid` as plan data; `PassFactoryDescriptor::declares_interop` plus
      `realized_substrate_mask_hint` / `declares_interop_hint` as the registry
      facts; and two rejection reasons mirrored into
      `PathSwapRejectionReason`.
      Backward compatibility is by construction, not by hand-kept exception:
      `RenderPathCapabilitySet::available_substrate_mask` defaults EMPTY
      ("single-substrate host"), so every pre-RP-1 caller resolves onto the
      declared substrate exactly as before — the suite was green *before* any
      new test existed. The one gate that had to move is the pass-realizability
      check: "does this pass support the recipe's declared backend?" became
      "does this pass intersect the admissible set?", which on a
      single-substrate host answers identically (same verdict, same
      `BackendUnavailable`), and is the only phrasing under which a device-only
      pass can join a software-declared recipe at all.
      Pinned by five gates in `shs_renderer_renderpath_tests` —
      `substrate_policy_resolution`, `substrate_intent_binds_policy`,
      `substrate_hybrid_legality`, `substrate_resolution_snapshot_contract`,
      `registry_single_recipe_two_substrates`; all five are prove-failed
      (disabling the crossing rejection fails the hybrid gate; disabling intent
      narrowing fails the intent and snapshot gates; forcing the registry's
      policy to `exact_match` fails the registry gate on the device host; and
      restoring the fork's two-recipe shape fails it on the recipe count), so
      the laws bind rather than merely document.
      **Fork removed (same day).** `make_default_soft_shadow_culling_recipe(backend)`
      — two different pass chains *and* two different technique modes for one
      intent, the last place the substrate chose the *shape* of the path — is
      deleted, and all four of its references migrated:
      `RenderPathRegistry::register_default_recipes()` now registers ONE recipe
      (`soft_shadow_culling`, `device_preferred`); the `renderpath.contract.hpp`
      re-export names the unified maker and now also carries the
      `SubstratePolicy` / `RenderDomain` / `with_domain` authoring vocabulary;
      and `hello_soft_shadow_culling_vk.cpp` declares its substrate and lets the
      policy resolve — plan-identical, since with no advertised substrate set
      admissible == {declared} == {Vulkan} and the chain was already the fork's
      Vulkan branch. Nothing looked the two old names up, so this is a rename
      plus a deletion. Suite re-verified 73/73 green after it; header inventory
      regenerated (`header_count` 230).
      Residuals, stated: the authoring-time substrate choice survives one tier
      up, at **composition** resolution — the demos call
      `resolve_builtin_render_composition_recipe(..., RenderBackendType::Vulkan, ...)`,
      and their parity harness obtains the software plan by cloning the resolved
      recipe and flipping `backend`
      (`demo_forward_classic_renderpath.cpp:1080-1085`,
      `hello_rendering_paths.cpp:1010-1015`) against a second, substrate-flavored
      `pass_contract_registry_sw_`. That clone is a two-resolution *comparison*
      (same chain, same technique mode — already value-shaped), not the removed
      authoring fork; making it policy-driven belongs with RP-2's
      contract-registry axis split. Both files are Vulkan targets and are **not
      built** in the software-only configuration (`exps-gpu-renderer` is
      commented out at `cpp-folders/CMakeLists.txt:120`), so that migration
      cannot be compile-verified here and was not attempted.
      The builtin registry still registers every standard pass software-only,
      so device-preferring over the builtin table resolves to software until a
      descriptor declares a device realization. `cheapest` is a crossing count
      (greedy over authored order), not a cost model — no honest cost data
      exists to build one from. The plan tier enforces only the
      declared-boundary half of the RP-2 hybrid rule.
- [x] **RP-2 Execution-unit × substrate axis split** (owner: same, req 5). **DONE
      2026-09-18** — full suite 73/73 green; the enum split is
      inventory-neutral.
      `ContractDomain` and `PassResourceDomain` are duplicate enums with
      identical value sets, and the mapping between them collapses "host" into
      "software rasterizer", which cannot express host-assisted *device* work.
      Split into one vocabulary with two independent axes, make the cross-axis
      relation an explicit handoff, retire one enum. **RULED 2026-09-18:
      `ExecutionUnit { host, device }` × `Substrate { software_raster, opengl,
      vulkan }`, `PassResourceDomain` retired, hybrid chains legal only with a
      declared interop boundary + shared staging resource (else a rejection,
      not a warning).** Measured size: three dead `ContractDomain` values
      (`CPU`/`OpenGL`/`Vulkan`, 0 uses each); live surface is one uniform
      literal (`GPU`, 59 uses in `pass_contract_registry.hpp`) plus `Software`
      (62, `pass_adapters.hpp`) — mechanical re-annotation, not a rewrite. The
      single load-bearing split is `ContractDomain::Software`, which must
      separate "host execution" from "software realization". **Landed
      2026-09-18**: both enums deleted; `ExecutionUnit` × `Substrate` +
      `RenderDomain` in `planning/pass_contract.hpp`; all 216 annotation sites
      re-labelled; cross-unit is now a rejection not a warning (promoted in
      `frame_graph.hpp`, waived only for a declared `is_interop_pass()`
      boundary). The split was verified semantics-preserving *on its own* — the
      new relation returns the same boolean as the retired one on every input
      pair — so the rejection is a separate deliberate change bundled here.
      Pinned by `hybrid_interop_legality` plus three relation gates in
      `shs_renderer_renderpath_tests`; the hybrid gate was prove-failed.
- [x] **RP-3 Open shader-identity registry** (owner: same, req 6; closes the
      P1.5 residual "no consumer/open shader ids yet"). **DONE 2026-09-18.**
      `ShaderId` now carries the `PassIdRegistry` shape — builtin range +
      content-addressed open registered range — with the mint registry
      (`render/shader/shader_id_registry.hpp`) **owned by** the manifest
      (`ShaderManifest::intern_shader` / `register_named_shader`), so a consumer
      mints a shader identity from a name and resolves it through the same
      backend-blind path a builtin uses, with **zero core edits** (no enum
      change, no census change, no consumer include change — `shader_identity.hpp`
      is the umbrella that re-exports the two new headers).
      Third instance of the shape, and the rule of two was **discharged rather
      than copied**: the offset law was hoisted to `shs/core/open_id_hash.hpp`
      and `PassIdRegistry::open_offset` now delegates to it, so pass and shader
      ids cannot drift (gate case `shared_offset_law`; mutation probe 1 fails
      **both** namespaces' gates). Two new named refusals —
      `UnregisteredOpenId` (a foreign/unminted open id) and `NameCollision` (the
      real failure mode of content addressing) — because opening the id space
      created two genuinely new failure modes.
      Verified: build clean/0 warnings; full CTest **74/74** (was 73) and
      **74/74** under `SHS_CONTRACTS_ENFORCED=1`; new gate
      `shs_renderer_shader_id_open_tests` (14 named cases) + parity guards in
      `shs_renderer_shader_identity_tests`; contract placement 230 → **232**
      headers, `header_count` 230 → **233** (regenerated, idempotent, md5
      stable); **four** mutation probes, all reverted byte-identically.
      Evidence: `docs/backlog/shader_id_open_registry_evidence_2026-09-18.md`.
      Residuals, stated: 16-bit id space (64,511 open slots); a refused
      registration leaves its name minted (no `remove`, same as
      `PassIdRegistry`); `ShaderManifest::operator==` is no longer `constexpr`
      (open half is `std::deque`/`std::string`); P1.5's past-the-manifest
      residuals (Vulkan loader input, GPU-half census, P2 reflection) are
      **not** advanced by this slice; and **no shipping consumer uses an open
      shader id yet** — the gate proves the capability, P2.5/P3 will exercise it.

- [ ] **RP-0 — re-enable the consumer-verification build (enabler, not a
      dynamism slice).** `exps-gpu-renderer` is commented out at
      `cpp-folders/CMakeLists.txt:120`, so the demo-side technique work
      (`demo_forward_classic_renderpath.cpp`, `hello_rendering_paths.cpp`,
      `demo_renderpath_bridge.hpp`) is **not compile-verified**, and neither is
      any RP-4…RP-8 capability against a real consumer. This is the same root
      cause as RP-3's stated residual "no shipping consumer uses an open shader
      id yet". Every row below is library-internal until this lands, so it gates
      their *proof*, not their *implementation*. **Timeboxed and resolved
      2026-09-18 — not started, and deliberately so.** Recon found the parking
      reason documented in-tree at `cpp-folders/CMakeLists.txt:103-107`: the
      three `exps` trees were parked on 2026-09-16 (`644be48`) with a known
      queued refactor — "exps CMake/TUs still reference retired aliases + facade
      include paths, so they need a pass before they build again". That is a
      whole-tree migration, not a one-line uncomment. Recon also found the
      **active** tree (`exps-rendering-adventures`, tier0/tier1 technique demos,
      which owns the `t1_08_*` gates) is **not a render-path consumer either**:
      it links `shs::renderer` only for the `window_backend.hpp` platform seam
      and **zero** of its TUs include `shs/renderpath`. So the honest position is
      that the render-path library currently has **no consumer outside its own
      tests** — which is exactly where its three open-registry precedents put
      their gates, so RP-4…RP-8 proceed with library gates and consumer-level
      proof stays deferred. Revisit only if a real (non-test) consumer is wanted;
      then the work is the parked-tree migration, sized separately.
- [x] **RP-4 — open `PassSemantic`** (owner: same, req 7). **DONE 2026-09-18.**
      `PassSemanticEncoding` was in the original scope and was **not** opened —
      it is the attachment-packing axis, deliberately unscheduled (see RP-4's
      residual note in the arch §4 req 7 entry); semantics were the half that
      unblocks arbitrary G-buffer *channel* naming, and opening both at once
      would have conflated "name a channel" with "decide its physical format".
      Shipped: range law + vocabulary pins in `pass_contract.hpp` beside the
      enum (`kPassSemanticOpenBase/Max`, `pass_semantic_is_builtin/_is_open/
      _in_valid_range`, `kPassSemanticOpenSpelling`); `PassSemanticRegistry`
      (`planning/semantic_registry.hpp`) with `intern`/`try_name`/`is_open`
      delegating offsets to the shared `core::open_id_offset` law (no fourth
      copy); all four switch surfaces answered — `pass_semantic_name` (open
      spelling via `default:`), `default_pass_semantic_descriptor` (explicit
      early return on `pass_semantic_is_open`, so `Unknown`'s old descriptor
      survives), `render_path_resource_id_for_semantic`, and
      `make_default_resource_spec_for_semantic` (its existing `default: break`
      already gave open semantics the neutral Full/Texture2D default);
      `pass_semantic_name_or_null` + `parse_pass_semantic` as builtin-only
      counterparts; and the registry-aware overloads for resource id and spec.
      Gate: `shs_renderer_semantic_id_open_tests` — 12 sub-checks covering the
      range law, content-addressed/order-independent ids, null + reserved
      rejection, capacity arithmetic, a real brute-forced collision, foreign-id
      hard miss, the neutral descriptor, contract acceptance with and without
      overrides, resource-id distinguishing vs the bare-spelling alias, and
      AD0 parity for every builtin (including `Unknown`). **Full suite 75/75.**
      Split from the RP-1/2/3 track 2026-09-18: that track delivered
      *orchestration* dynamism (pass arrangement, shader identity, substrate
      policy) but not *vocabulary* dynamism, and "build arbitrary G-buffer
      layouts" needs the vocabulary. `PassSemantic` is a closed 16-value enum
      with no `Custom`/open range; `PassSemanticEncoding` likewise, so there is
      also no consumer vocabulary for physical attachment packing. Change surface
      is four live switches (`pass_contract.hpp:253`, `:357`,
      `render_path_resource_plan.hpp:112`, `:160`) — enumerable, and **no mask
      change is needed** (semantics are a `std::vector`, unlike technique modes).
      Reuse the `PassIdRegistry` shape and the already-shared
      `shs/core/open_id_hash.hpp` law; do **not** add a fourth copy.
      Gate (as shipped): `shs_renderer_semantic_id_open_tests` — a consumer
      mints a semantic by name, declares it in a `TechniquePassContract`, and it
      plans; a foreign semantic hard-misses. Note the switch line refs above are
      the *pre-change* tree; they moved when the range law was inserted beside
      the enum. Evidence:
      [`semantic_id_open_registry_evidence_2026-09-18.md`](semantic_id_open_registry_evidence_2026-09-18.md).
- [ ] **RP-5 — unify, then open, the render-technique vocabulary** (owner: same,
      req 8). **Three** closed enums describe one axis and disagree:
      `TechniqueMode` (`render/frame/technique_mode.hpp:19`, 5 values),
      `RenderPathPreset` (`planning/render_path_presets.hpp:31`, the same 5 values
      with identical spellings), and `RenderPathRenderingTechnique`
      (`planning/render_path_recipe.hpp:76`, 3 values, names differ — `ForwardLit`
      vs `Forward`). `RenderPathRecipe`
      carries the 3-value one and `technique_mode_for()`
      (`renderpath.gateway.hpp:102`) maps exactly those three, defaulting the
      rest to `Forward` — so **`TiledDeferred` and `ClusteredForward` currently
      have no authoring path from a recipe**, which is precisely the goal named
      for this track. Retire one enum (rule of two), make the relation total,
      then open. The mask-representation decision this row was blocked on is
      **made, 2026-09-18: cap at 32** — builtins 0–4 pinned, reserved 5, open
      range 6–31 (26 consumer modes), no widening and no side set, because
      `1u << v` is well-defined for `v ≤ 31` and therefore needs no representation
      change at all. Retirement order, the evidence for each option, and the
      exact range law: [`technique_vocabulary_mask_ruling_2026-09-18.md`](technique_vocabulary_mask_ruling_2026-09-18.md).
      Unblocked; **not started**.
- [ ] **RP-6 — unify, then open, the shading-model / technique-preset axis**
      (owner: same, req 8 second half → mobile lighting). `ShadingModel`
      (`render/frame/frame_params.hpp:130`, PBRMetalRough/BlinnPhong) and
      `RenderTechniquePreset` (`planning/render_technique_presets.hpp:24`,
      PBR/BlinnPhong) are a **second duplicate pair** on this axis, bridged by
      `render_technique_preset_from_shading_model`. Unify under RP-5's ruling,
      then open, so a consumer can register a cheap mobile model (`mobile_unlit`
      / `mobile_lambert`) instead of paying for the builtin two. Note §3 of
      `render_path_architecture.md` currently contradicts §4 by instructing
      authors to "Edit `render_technique_presets.hpp`" — a documented core edit;
      that guide text must be corrected as part of this row.
- [ ] **RP-7 — open the light vocabulary** (owner: same, req 2). Light *type*
      (`lighting/light_types.hpp:27`) is nominally 6 values and **richer than
      req 2 implies** (RectArea, TubeArea, EnvironmentProbe already exist), so
      audit before opening rather than assuming. The real mobile gap is
      `LightAttenuationModel` (`:52`, 3 values) and
      `RenderPathLightVolumeProvider` (3 values) — both closed, both
      consumer-authorable. Do this after the audit; do not open
      `LightCullingShape` (11 values, engine-internal bounds vocabulary) without
      a stated consumer.
- [ ] **RP-8 — composition-tier substrate policy + contract-registry axis
      split** (owner: same, req 1 residual + req 5). Closes RP-1's stated
      residual: composition-tier choice is still hardcoded (a
      `RenderBackendType::Vulkan` argument plus a second substrate-flavoured
      `pass_contract_registry_sw_` table that exists only to produce a software
      plan), and the builtin registry is software-only. **Deliberately excludes
      `Substrate` itself** (3 values): registering a fourth substrate touches RHI
      and is a larger decision than an id-range job — it gets its own ruling or
      none. What is *not* in this tracker at all: the **attachment-packing
      schema** (which channels occupy which target format) — that is a *new*
      abstraction, not an enum to open, and needs design before it can be
      scheduled.

- [ ] **P5 box 1 — role/suffix completion.** Today each DVO zone carries a single
      `.contract.hpp` (`render` carries two) and only 5 gateways exist (`renderpath`,
      `render/frame`, `input`, `logic`, `app/session_orchestrator`); there are **0**
      `.plan.hpp` / `.edge.hpp` splits (`renderpath` holds 40 headers against 1
      contract header). Extends the existing `check_contract_placement.sh` gate's
      coverage (225 headers today).
- [ ] **P5 box 4 — retire `frame_graph.hpp` / `pluggable_pipeline.hpp`.** Genuinely
      open but **PATH_COMPILED-gated** (P6): `frame_graph.hpp` (218 lines) has one
      internal consumer; `pluggable_pipeline.hpp` (1,050 lines) still has 3
      (monolith, `hello_rendering_paths`, `core_tests`). Its facade role is already
      thin — `PluggablePipeline::execute` delegates to
      `PipelineRuntimeExecutor` + `PipelineExecutionPlanner` inside the same file.
- [ ] **P3 box 7 — migrate or retire the `hello_*_vulkan` probes.**
- [ ] **P4 — demo-internal Core 4 debt** (tetris/snake vocabularies; regenerate
      `docs/pods/EVENT_FLOW.md`).
- [ ] **Enforcement actually runs.** The repo has **no CI at all** (no
      `.github/workflows`), so the 71 CTest gates + boundary/include-graph/purity
      gates are opt-in — they run only when someone remembers. VOP-first roadmap
      item 5 ("wire CTest gates into CI so regressions fail automatically") is
      therefore genuinely open. Platform choice is an owner decision.
      **Evidence, 2026-09-18 (RP-2):** this is not hypothetical. On a pristine
      `HEAD` checkout with RP-2 reverted, `shs_renderer_header_inventory_check`
      is already **red**: `engine_header_inventory.json` is stale by 9
      insertions / 2 deletions of `shader_identity.hpp` consumers from work
      merged without regenerating it. A red gate sat in the tree unnoticed.
      RP-2 turned no green gate red; its enum split is inventory-neutral (the
      regenerated file is byte-identical with and without it) and its only
      inventory delta is one `tracked_consumers` line for the new gate test's
      include. It is now green.
      **Also: one gate is flaky** — `shs_renderer_lifecycle_semantics_tests`
      fails ~1 run in 25 on `thread_pool_shutdown_order` (a
      `ThreadPoolJobSystem` race, zero render-path involvement). Flakiness must
      be fixed before these gates can be trusted as CI blocking signals.
      **Also: one gate is mis-argued for the common case, and fails with a
      misleading message.** `shs_renderer_package_consumer_test`
      (`tests/package_consumer_test.sh`) takes
      `<build-dir> <source-dir> [toolchain] [cxx]`, but `CMakeLists.txt` only
      appends the toolchain argument `if(CMAKE_TOOLCHAIN_FILE)`. A configure with
      no toolchain file therefore shifts every later argument: the C++ compiler
      lands in the `[toolchain]` slot and the prefix path in `[cxx]`, so the
      nested consumer configure runs `-DCMAKE_TOOLCHAIN_FILE=/usr/bin/c++`
      (CMake then tries to *include the compiler binary* as a script — "Parse
      error. Expected a command name, got unquoted argument with text \"ELF\"")
      and `-DCMAKE_CXX_COMPILER=<vcpkg prefix>`, ending in "unable to find a build
      program" rather than anything that names the real cause. Found 2026-09-18
      while verifying the RP-1…RP-4 split in a fresh out-of-tree build dir; the
      canonical `cpp-folders/build` sets a toolchain file, which is why the gate is
      green there and the bug stayed invisible. Fix: pass the toolchain slot as an
      explicit empty positional, or key the arguments off their names rather than
      their positions.
- Host-blocked, not startable: **P6** replay/rollback (PATH_COMPILED-driven executor
  rebuilds) and **C4.3** (needs `__cpp_contracts` + `<contracts>`; GCC 13.3.0 lacks
  both — owner baseline ruling outstanding).

### A1. GPU execution path (owner: [`kdba_conformance_backlog.md`](kdba_conformance_backlog.md) §"GPU execution path — selected next work")

- [x] **K-G1 tail — explicit recording failures** — CLOSED 2026-09-18:
      command-buffer prerequisite branch proven for non-empty streams (headless
      recorder suite + real-device prerequisite suite, correct
      stage/index/command attribution); `recording_ready` prerequisite
      precedence proven over stream validation with zero sink calls;
      missing-buffer branch and order-validation/stage-diagnostics breadth
      already table-pinned. Full CTest 50/50. Non-goals stand: sink failures
      stay fail-fast; headless evidence only. Pass/pipeline realization and
      in-pass execution are G2; factory-facing execution is G3. Next in this
      track: K-G2.
- [x] **K-G2 Minimal offscreen graphics realization** — finish attachment
      setup / pipeline creation+binding / begin-end-pass recording breadth in
      the new driver (slice 1 partial: `b2d7d79`; realization now at
      `shs/rhi/vulkan/value/vk_offscreen.hpp` after the R5 sweep). Acceptance:
      one deterministic scene through value commands; explicit
      formats/layouts/features; failed creation unwinds; zero validation
      errors on an available backend. Depends on G1.
      CLOSED 2026-09-18: acceptance evidenced end-to-end — deterministic scene
      through value commands to known pixels, explicit supports() tables, unwind
      on failure (incl. injected Vulkan faults), validation-clean lavapipe runs,
      SW/Vulkan triangle parity; see the kdba G2 DONE entry. Follow-on work
      belongs to G3 (upload tail — landed, factory-facing execution) and G4.
- [x] **K-G3 remaining: factory-facing execution** — CLOSED 2026-09-18: the
      minimal scene now runs from `create_render_backend()` + `app::Context`
      through a new vendor-free `IOffscreenExecution` hook on `IRenderBackend`
      (no `dynamic_cast`, no Vulkan type in the consumer), pinned by the portable
      gate `shs_renderer_vk_factory_offscreen_tests` (known pixels derived from
      the authored scene, rejection + reset/re-prepare, SKIP 77 when the surface
      or device is unavailable). Full CTest 68/68; boundary + include-graph gates
      green; header inventory regenerated. Evidence:
      [`kdba_g3_factory_facing_evidence_2026-09-18.md`](kdba_g3_factory_facing_evidence_2026-09-18.md).
      Async retirement unclaimed. Next in this track: K-G4 (closed 2026-09-18 —
      see the K-G4 entry below).
- [x] **K-G4 Library SW/Vulkan equivalence** — same minimal scene/policy
      through actual library execution paths; documented per-output tolerances;
      independent known-answer checks; portable CTest gates. Adventure AD1/AD4
      are related, not substitute evidence. Depends on G3 and a verified
      software realization of the selected recipe.
      CLOSED 2026-09-18: the software side now realizes the generic contract
      instead of declining it (`SoftwareOffscreenExecution` — no device to open,
      descriptor-derived stable ids, entry-name-bound CPU realization), and both
      realizations share one vendor-free descriptor gate so acceptance cannot
      drift. New portable gate `shs_renderer_sw_vk_equivalence_tests` drives ONE
      consumer function (factory → `app::Context` → generic `IRenderBackend` →
      generic `IOffscreenExecution`) against both backends with the same command
      stream, checks each readback against the authored scene independently, then
      compares them under two documented tolerances (≤ 1/255 per channel where
      both cover; coverage budget 16 at 32×32, GPU may only add coverage).
      Measured `cpu_covered=113 gpu_covered=128 mismatches=15 (both=0 gpu_only=15
      cpu_only=0)`. The software half is always asserted; only the Vulkan half may
      skip (77, equivalence explicitly not claimed). Full CTest 69/69; boundary +
      include-graph + self-containment + package-consumer gates green; inventory
      regenerated (225 → 226). Evidence:
      [`kdba_g4_sw_vk_equivalence_evidence_2026-09-18.md`](kdba_g4_sw_vk_equivalence_evidence_2026-09-18.md).
      Not claimed: CPU realization bound to the authored recipe by entry name;
      software accepted set is a strict subset of Vulkan's; coverage tolerance
      calibrated for this fixture; gate not compiled in a GPU-free build.
      **The library G-track (G1–G4) is now complete.**

### A2. Adventure demo conformance (owner: [`adventure_demo_conformance_backlog.md`](adventure_demo_conformance_backlog.md) — 13 open checkboxes; AD0, AD1, AD2, AD3, AD4 closed 2026-09-18)

- [x] **AD0 Fresh reproducible baseline** — DONE 2026-09-18:
      evidence in [`adventure_demo_baseline_2026-09-18.md`](adventure_demo_baseline_2026-09-18.md).
      All six pairs rebuilt + parity-recorded (no backend unavailable);
      two envelope breaches (01, 02) proven pre-existing device drift
      (Mesa 25.2.8) via a pre-R1 (`983925c`) worktree cross-check and
      re-pinned with dated rationale (shrink-only from here); zero refactor
      regressions. Depends on: —
- [x] **AD1 Portable automated demo gates** — DONE 2026-09-18:
      suite fully de-machined (CMake-supplied dirs, isolated scratch
      outputs, capability-skip vs failure split); 6 SW smoke + 6 per-pair
      parity CTest entries (incl. tier1 08) + negative-probe test; probes
      caught and fixed a real `--tol 1` parsing bug in `t0_parity.py`.
      Full CTest 63/63. Depends on AD0.
- [x] **AD2 Shared semantic state + explicit draw inputs** — CLOSED 2026-09-18:
      one execution-neutral `PassPolicy` (`common/adventures_pass_policy.hpp`)
      replaces the parallel `SwState`/`VkPipelineSetup` copies; `SwState` is
      deleted and the policy is a per-draw argument, so the prerequisite
      `raster.state = ...` sequencing (and the leak it allowed) is gone; the
      Vulkan harness bakes pipeline state from the same policy and derives its
      dynamic scissor from it. Stencil mode is closed, retiring the
      silently-ignored "invert without test" pair. Gate `t0_policy_tests` (45
      checks, GPU-free, cannot skip) covers defaults, depth storage, order +
      analytic source-over, all four stencil modes, scissor bounds/clamp, and
      the policy-isolation property; three mutation probes failed their gate.
      Six-pair baseline numbers identical to AD0 — no pixel moved. Depends on AD0.
- [x] **AD3 Typed compositional orchestration pilot** (depth/blend pair) —
      CLOSED 2026-09-18: demo 03 is now pure preparation
      (`depth_blend_plan.hpp`: request → `expected<DepthBlendPlan,
      DepthBlendError>`, plan owns its geometry) → explicit execution/PNG edges
      (`depth_blend_edges.hpp`, plus the Vulkan twin's own executor) with one
      host-boundary diagnostic mapping and stage-derived exit codes
      (preparation 1 / execution 2 / output 3). Closed 14-member error
      vocabulary; both twins consume the same plan. Gate
      `t0_composition_tests` (91 checks, GPU-free) covers plan shape,
      short-circuiting with the original error preserved, execution's refusal of
      unvalidated plans, the output edge (descriptor count unchanged on failure),
      plan ownership, vocabulary totality, and policy propagation into the
      executed result. Evidence:
      [`adventure_demo_ad2_ad3_evidence_2026-09-18.md`](adventure_demo_ad2_ad3_evidence_2026-09-18.md).
      Full CTest 71/71. Depends on AD2.
- [x] **AD4 Independent known-answer tests** — DONE 2026-09-18: evidence in
      [`adventure_demo_ad4_evidence_2026-09-18.md`](adventure_demo_ad4_evidence_2026-09-18.md).
      Always-active checks that fail the process; demos 01/04/08 verified
      against analytic oracles over the real SW demo PNGs, 02/03/05 driven
      in-process with direct depth/stencil STORAGE inspection; every tolerance
      justified in the note. Validation found and fixed five oracle defects
      (28 mismatches → 0); prove-fail covers both oracle styles. Full CTest
      67/67 + `check_kdba_boundaries.sh` + `check_include_graph.py` green.
      Depends on AD0 (registered through AD1).
- [ ] **AD5 Single-source shared scene inputs.** Depends on AD0.
- [ ] **AD6 Execution-format adapters** (vertex/push-constant packing isolated
      at the Vulkan boundary). Depends on AD2.
- [ ] **AD7 Rollout to remaining five pairs + close-out with evidence.**
      Depends on AD1–AD6.

Guardrails (from the owner backlog): no per-pixel events, no dummy stateful
pods, no vacuous errors, no general-purpose demo engine for signature
uniformity; preserve lesson-specific kernels (incl. demo 01's from-scratch
barycentric lesson); tolerance envelopes preserved unless separately
investigated.

## Bucket B — Blocked on owner ruling

- [ ] **C4.3 / P4 C++26 native contracts switch** (owner:
      [`cpp26_native_switch_runbook.md`](cpp26_native_switch_runbook.md),
      DRAFT) — the in-house `SHS_PRE`/`SHS_POST`/`SHS_CONTRACT_ASSERT` bridge
      is the ratified contract mechanism (owner ruling 2026-09-17); the native
      switch is the sanctioned replacement. Runbook fully drafted (14 sites,
      all single-expression; keyword mapping, handler-move plan, acceptance
      gates, rollback). G3.1 threshold DONE 2026-09-18: fires on
      `__cpp_contracts` + matching `<contracts>` library (feature-test macro
      only, never compiler-name checks). Local baseline GCC 13.3.0 has
      neither; mixed-toolchain items stay gated until Clang ships it.
      **Execution requires only the owner baseline ruling**; the replay-parity
      CTest is the release blocker when the switch fires.

## Bucket C — Blocked on demo/windowed host (P6, owner: kdba backlog)

- [ ] **P6.1 PATH_COMPILED-driven executor rebuilds** — live plan switching,
      resize, safe GPU retirement (accepted plans rebuild; rejected plans
      preserve the working renderer; in-flight resources outlive submission).
      BLOCKED: needs a live demo host.
- [ ] **P6.2 Replay harness (cross-session codec + CI replay)** — explicit
      codecs, versions, reconstruction rules remain unbuilt (K1.5's pod-level
      replay tests are not portable serialization). Host integration BLOCKED;
      **headless codec/fixture preparation may proceed independently via S5**.
- [ ] **P6.3 Rollback snapshots + time-travel overlay** — BLOCKED on windowed
      host; K4.1's generation counter is a precondition contributor.
- [ ] **P4.4 Seeded determinism implementation (STANDING)** — baseline
      contract adopted in Constitution II §11.1; the first stochastic pod
      version-pins its algorithm and proves S1. No completion claimed.

## Bucket D — Trigger-gated (no speculative infrastructure)

Owner: [`kdba_conformance_backlog.md`](kdba_conformance_backlog.md) §
"Future-domain scalability preparedness" (Constitution II §11.1, Rule 12).
Each item carries its explicit trigger; none authorizes speculative work.

- [ ] **S1 Seeded determinism proof (P4.4)** — Trigger: first stochastic pod.
- [ ] **S2 Large-state headless spike** — Trigger: before the first
      high-volume mutable domain; available independently of a windowed host.
- [ ] **S3 Streaming edge contract** — Trigger: first asynchronous
      IO/streaming domain.
- [ ] **S4 Time policy integration** — Trigger: next real-time host
      integration (P6.1).
- [ ] **S5 Replay persistence and retention (P6.2/P6.3)** — Trigger: before
      shipping a persisted session format; headless preparation need not wait
      for GPU/window availability.
- [ ] **S6 Production saga proof (Rule 12)** — Trigger: first multi-domain
      transaction.

## Bucket E — Demand/measure-gated (optimization register)

Owner: [`optimization_backlog.md`](optimization_backlog.md). Not mirrored
checkbox-by-checkbox here: the owner doc carries its own checkboxes and its own
tick rule (a named measurement must show the change pays for itself before any
item closes). 18 boxes / **16 distinct tasks** (QW3≡#1, QW5≡#4), all gated on
consumer demand and measured evidence — no speculative scheduler, render graph,
meshlet pipeline, or LOD framework is authorized by listing them.

## Suggested sequencing

1. **Render-path dynamism track (RP-1…RP-3) — owner-chosen current priority**
   (2026-09-18). Goal: one recipe resolves to a software, device, or hybrid plan
   with no consumer-side recipe cloning. Ordering constraint, stated in the
   owner entry: **RP-2's vocabulary decision precedes RP-1's acceptance test**
   — RP-1's resolver input already exists and is substrate-keyed
   (`PassFactoryDescriptor::backend_mask`), but RP-1's *hybrid* half would be
   validated by `pass_resource_domains_compatible`, which encodes the
   conflation. Sequence: RP-2 ruling (a decision, not code) → RP-1 → RP-2 code
   refactor → RP-3. **All three are done 2026-09-18.** RP-1 closed including
   consumer migration (the authoring-time substrate fork
   `make_default_soft_shadow_culling_recipe(backend)` was deleted and all four
   references migrated — see its entry above), so the "no consumer-side recipe
   cloning" goal is met for the *authoring* half; RP-3 closed the identity half
   (`ShaderId` open range, with the rule of two discharged by hoisting the
   offset law rather than copying it). What remains of this track is stated
   residual, not unfinished scope:
   - **RP-1 residual** — composition-tier substrate choice is still hardcoded
     (a `RenderBackendType::Vulkan` argument plus a second substrate-flavoured
     `pass_contract_registry_sw_`); making it policy-driven belongs with RP-2's
     vocabulary.
   - **RP-2 residual** — the demo's two-resolution *parity harness* is still a
     value clone with `backend` flipped (same chain, same technique mode) in
     `demo_forward_classic_renderpath.cpp` and `hello_rendering_paths.cpp`. It
     is **not** the removed authoring fork — that was a different leak, in a
     different layer — and it belongs with RP-2's contract-registry axis split,
     not with RP-1.
   A0's other items are deferred behind these, not dropped.
   **Amended 2026-09-18 — the track was not complete, and this list said so
   wrongly.** An audit against the owner's four stated goals (software path;
   tiled / clustered / deferred / forward+; arbitrary G-buffer layouts;
   lightweight mobile lighting) found that RP-1/2/3 delivered *orchestration*
   dynamism but not *vocabulary* dynamism. The four goals live on closed enums
   this sequencing never scheduled — see arch §4 reqs 7–8 and its "Blind spot"
   note. Plainly: `PassId`, `ShaderId` and now `PassSemantic` are open;
   technique modes, shading models and the light vocabulary are not, and
   `TiledDeferred` / `ClusteredForward` have no authoring path from a recipe at
   all. Next on this track: **RP-4** is ✅ **DONE 2026-09-18** (semantics; the
   one row with no pending decision, which is why it went first). **RP-0 is
   timeboxed-and-closed, not started** — recon showed the parked trees need a
   whole-tree alias/include-path migration and that even the *active* demo tree
   is not a render-path consumer, so the whole series proceeds on library gates
   (see RP-0's row and the arch §4 reckoning). Remaining: **RP-8** (composition-
   tier substrate policy — independent of everything else and pullable to the
   front if the software-path goal takes priority) or **RP-5** (technique
   vocabulary; carries the series' one genuine design decision — the `uint32_t`
   mode-bitmask representation — so it stays blocked until that ruling lands)
   → **RP-6** (shading model, which unblocks the mobile model) → **RP-7**
   (lights). `Substrate` itself and the attachment-packing schema are explicitly
   *not* in this series. Cost honesty: RP-4 landed at roughly the predicted
   scale; RP-5…RP-7 are each roughly RP-3-sized (~1.3k lines across headers plus
   a named gate, an evidence doc and mutation probes).
2. **Adventure demo track (A2)** — AD5 + AD6 next (single-source shared inputs,
   then execution-format adapters), then the AD7 roll-out and close-out of the
   remaining five pairs onto AD3's preparation/execution/error shape. Gives
   consumer-contract feedback at demo granularity, which the library track no
   longer does.
3. **S5 headless prep** — banks host-blocked P6.2 work early, if desired.
4. C4.3 waits on the owner baseline ruling; P6.1/P6.3 wait on a windowed host;
   Bucket E stays demand/measure-gated.
5. Standing cautions on this reorder: **P3 box 4** (monolith → thin DVO
   composition) is the largest deferred structural win and the proof-by-consumer
   for the DVO spine — deferring it is a deliberate trade, not an oversight.
   And **"Enforcement actually runs" (no CI)** should be revisited *before* this
   track adds gates, since a headless dynamism gate that never runs is
   opt-in evidence.

## Related demowork

The adventure-demo backlog (A2) is the consumer-facing track;
[`kdba_conformance_backlog.md`](kdba_conformance_backlog.md) §"Related
demowork" keeps the scopes separate — demo evidence never substitutes for
library gates and vice versa.
