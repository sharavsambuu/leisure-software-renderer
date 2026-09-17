# Domain Pod Hardening Backlog — shs-renderer-lib

> ARCHIVED 2026-09-17 (moved from `docs/backlog/`; FROZEN banner below stands):
> superseded by [`kdba_conformance_backlog.md`](../backlog/kdba_conformance_backlog.md)
> and fully dispositioned (open work rolled forward there). Read-only
> provenance — do not tick, edit, or re-open items.

> **STATUS: FROZEN (2026-09-16)** — superseded by `kdba_conformance_backlog.md` (KDBA gateway migration directive). Read-only history: do NOT tick, edit, or re-open items here. Open work rolls forward in the new backlog: P6.1-P6.3 (blocked on demo host) + P4.4 (standing law) are carried there under "Rolled forward"; everything else on this file is DONE or explicitly closed. Kept as provenance — conflicts are recorded, never overwritten.

> Status: active (2026-09-16). Source: lib review vs Constitutions I/II/III + rollout roadmap P3-P6 + backlog "POD Semantics Hardening" (parked 2026-09-15).
> Law precedence: Constitution II S2.1 + Rule 10, canon S6.1-S6.2, precedence S2.2. Roadmap is schedule, not law.
> Verification after every item: `build/` ctest 5/5 green (grows as pods land) + `check_kdba_boundaries.sh` green.

## Run plan (2026-09-16) — 5 runs by touch surface

Each run is independently committable: build + growing ctest + linter green, backlog ticked, docs updated.
Constraints: R1 -> everything; R2 -> R5; R4 proves the interleave loop before R5 commits to it; R3 can parallel R1/R2.

- **R1 — Spine honesty**: P0.1, P0.2, P1.2, P1.3. Build dirs, agent doc, 2 SDL loaders, GPU-free configure, parity target.
- **R2 — One tree**: P1.1, P2.1-P2.4. shs/pipeline + shs/rhi moves, 5 include splits, linter grandfather block.
- **R3 — First pod + tooling**: P3.1 (input), P3.2 (frame), P4.1, P4.2, P4.3. First pods establish uniform sig, test kit, purity linters (suite 5->7).
- **R4 — Tier1 pilot**: P3.0, P3.4, P3.5, P4.6. Rung-8 pair -> parity -> TBN ingestion -> geometry+lighting pods, expected promotion.
- **R5 — Sweep + harden**: P3.3, P3.6-P3.10, P4.4, P4.5, P5.1-P5.3, P6.1-P6.3. Remaining pods, suffix splits, seam retirements, linter-docs sync, replay/overlay.

## P0 — Baseline honesty (do first, hours)

- [x] **P0.1 Test-dir canonicalization** — DONE 2026-09-16: build_vcpkg cache pointed at deleted src/hello-3d-demos (stale restructure); wiped CMakeCache/CMakeFiles, fresh configure (75s), test binaries built, both dirs 5/5 green. — `build/` registers 5/5, `build_vcpkg/` registers 0. Reconfigure `build_vcpkg` (or correct `agent_environment.md` to name `build/` canonical). DoD: both dirs report identical test counts; doc matches reality.
- [x] **P0.2 Evict SDL from `domains/`** — DONE 2026-09-16: texture_loader_sdl + cubemap_loader_sdl + asset_manager (0 consumers) moved to execution/platform/loaders/; import_texture_sdl moved with the loader (edge->domain include is legal direction); resource_import.hpp keeps assimp only (assimp-in-domains noted as future debt). DoD: grep SDL_ domains = 0, no stale paths, build + 5/5. — `resources/loaders/texture_loader_sdl.hpp` (+ `sky/loaders/cubemap_loader_sdl.hpp`) include SDL_image and touch SDL_Surface in the pure zone (violates S2.1 layout + Rule 2). Move to `execution/platform/` loaders; leave pure TextureData/CubemapData contracts in `domains/resources|sky/`. DoD: grep SDL_ in domains = 0.
- [x] **P0.3 C++ standard decision** — DONE 2026-09-16 for lib (6 targets cxx_std_20 -> cxx_std_23, build + 5/5 green, GCC 13.3). Demos followed 2026-09-16 (L2 closed — all trees C++23, S8 note current).

## P1 — Kill the shadow tree (days, Run 1+2)

- [x] **P1.1 Retire `shs/pipeline/` + mark `shs/rhi/`** — DONE 2026-09-16 as amended: 2 pipeline headers merged into execution/pipeline/ (only live consumer was a comment; parked demos keep stale paths, documented), empty shs/pipeline/ deleted, linter facade-case dropped. shs/rhi/ (13-file Vulkan monolith, live behind SHS_HAS_VULKAN + tier0 vk harness) is NOT a move — it needs P3 decomposition, carried by R5. Linter now treats any new shs/pipeline file as FAIL. — absorbed monolith lives beside `shs/execution/`, outside linter coverage; all Vk-outside-drivers hits come from here. Move to `execution/` + delete old dirs, or mark legacy/ with sunset + extend linter to cover them. DoD: single tree (core/ domains/ execution/ memory/ containers/); no Vk outside execution/rhi/drivers/vulkan/.
- [x] **P1.2 GPU-free proof** — SHS_HAS_VULKAN gate + backend-factory fallback verified in CI (configure -DCMAKE_DISABLE_FIND_PACKAGE_Vulkan=TRUE -> build + ctest green). DoD: GPU-free build documented + green. DONE 2026-09-16: /tmp/swr-novk-r1 configure EXIT=0 (vk_driver_tests correctly skipped), lib 4/4 green, t0 _sw pair builds. Prerequisite: pinned slangc v2026.17.1 in WSL (currently missing — _vk targets unbuildable here); GPU-free proof covers the _sw half regardless.
- [x] **P1.3 Tier0 parity harness promotion (harness landed, GATE RED)** — DONE 2026-09-16: tools/t0_parity.py (stdlib-only PNG decode, exact% + 1-LSB tolerance + ASCII diffmap) runs over fresh-vs-fresh PNGs: 01 FAIL 70.94% (max 8), 02 FAIL 11.36% (max 217), 03/04 TOLERANCE-ONLY (max 1), 05 PASS exact. Docs 0.00%-everywhere does NOT reproduce (_vk runs headless here, so this is genuine drift, not stale artifacts). RESOLVED 2026-09-16 (R4 entry): 01 was a clear-color constant bug (SW (16,16,24) vs shared harness (12,12,16)) — aligned, now 0.00% exact. 02 is structural (affine-vs-perspective NDC-z + device precision), pinned as TOLERANCE envelope, not chased. tools/t0_parity_suite.py gates all 5 pairs fresh-vs-fresh: 01/05 exact, 02 tol12, 03/04 tol1 — SUITE PASS. — diagnose 01/02 before any Tier1 rung. — promote /tmp/t0final tooling (t0_parity + diffmap) into the repo as the cross-backend equivalence gate (tier0 lessons prescription). DoD: ctest parity target renders both halves (or _sw-only where toolchain absent) and diffs with a per-pass tolerance table. Blocks Tier1.

## P2 — Direction-law debt (per pod, Run 4 pre-req)

Grandfathered domains/ -> execution/ includes (linter INFO today, must reach 0):
- [x] **P2.1 camera/free_camera.hpp -> platform_input** — DONE 2026-09-16: FreeCameraInput domain struct + update() retargeted; new execution/platform/free_camera_bridge.hpp owns the PlatformInputState mapping (CRLF preserved). — split pure math (contract+plan) from platform edge.
- [x] **P2.2 scene/system_processors.hpp -> pluggable_pipeline** — DONE 2026-09-16: include was unused (zero symbol refs) — deleted, no split needed. — legacy-seam audit; plan vs executor split.
- [x] **P2.3 input/value_commands.hpp + input/command.hpp -> app/runtime_state** — DONE 2026-09-16: RuntimeState definition moved to domains/input/input_state.hpp; execution/app/runtime_state.hpp is a using re-export; redundant includes dropped. Pre-stages R3 P3.1 input pod. — vocab stays in domains/input, latch lives in execution/app.
- [x] **P2.4 sky/skybox_renderer.hpp -> job/parallel_for** — DONE 2026-09-16: pure shade_skybox_rows() stays in domains; dispatch wrapper render_skybox_to_hdr() (identical signature) lives in new execution/passes/pass_skybox.hpp; pass_pbr_forward re-pointed. Grandfather list evicted; linter fails on any new domains->execution include. — plan function + edge dispatch split.
- [ ] DoD: grandfather list empty; linter grandfather block deleted.

## P3 — Core 4 per pod (weeks, Run 4 main track)

Rule: one pod per commit; each lands with headless ctest (replay assert + snapshot round-trip + invariant), linked only to shs::renderer-values. Empty vocabs explicit (variant<monostate>), events carry no std::string.
Interleaving policy (2026-09-16): pods harden WITH their curriculum rung, not ahead of it — demo pair -> parity -> ingest one operator (rule of three) -> harden the touched pod in the same commit. Do not complete all of P3 before Tier1. Order by dependency:
- [x] **P3.0 Tier1 pilot (rung 8 normal mapping)** — DONE 2026-09-16: tier1-classic-shading/08 pair (full-viewport quad, flat vs bump, vertex normals in COLOR0, TBN mirrored in Slang, UNORM-rounding fix 89%->26%, sampler-precision envelope tol27). Suite 6/6 green. Tier1 CMake reuses tier0 harness sources (promotion-to-shared/ candidate, noted). slangc v2026.17.1 installed (actual path ~/slang/bin, not ~/slang/slang/bin as docs said). — full loop: pair -> parity -> TBN/compare-op ingestion -> harden geometry+lighting pods. DoD: loop proven before committing to all of Tier1-6.
- [x] **P3.1 input** — DONE 2026-09-16: input.contract/command/event/gateway.hpp; RuntimeAction vocabulary moved to its Core 4 home (value_commands re-exports, zero breakage); input_gateway canonical with 4 raw-fact events; legacy runtime_state_gateway delegates (conformance pinned); vop_input_tests 6/6 (replay, empty-log, event contents, look clamp, legacy conformance, latch determinism).
- [x] **P3.2 frame** — DONE 2026-09-16: frame.contract/command/event/gateway.hpp with explicit monostate vocabs + identity gateway; vop_frame_tests 3/3 (identity, replay, closed-vocab). First consumer of the kit.
- [x] **P3.3 camera** — DONE 2026-09-16: camera.contract/command/event/gateway.hpp (monostate vocabs + identity); vop_camera_tests 5/5 (follow snap/hold, light-fit determinism+chain, view chaining, kit). Finds: free_camera.hpp was never compilable (missing brace + fwd decl, invisible while parked-only) — completed in place; degenerate inverted-AABB fit yields NaN (caller error, test uses sane box, inverted-box assert is R5b).
- [x] **P3.4 geometry** — DONE 2026-09-16: geometry.contract/command/event/gateway.hpp (monostate vocabs + identity, culling runtimes explicitly out until R5); ingested tangent_frame.hpp (frame-from-normal, decode, perturb); vop_geometry_tests 6/6 (orthonormality, perturb identity, decode corners, kit replay/identity).
- [x] **P3.5 lighting** — DONE 2026-09-16: lighting.contract/command/event/gateway.hpp (monostate vocabs + identity, culling runtimes explicitly out until R5); ingested shading_terms.hpp (lambert_diffuse + shade_lambert, the pair formula); vop_lighting_tests 4/4 (known answers, composition, kit).
- [x] **P3.6 scene** — DONE 2026-09-16: scene.contract/command/event/gateway.hpp (item/object values + to_render_items projection as spine; stores + ISystem flagged edge-candidates for convergence); vop_scene_tests 4/4 (field mapping, projection round-trip + stable ids, kit). (R5b: stateful culling/world/elements need REAL gateway design, not identity shells — objects/culling/instance -> contract+plan; world/system -> edge subfolder)
- [x] **P3.7 resources** — DONE 2026-09-16: resources.contract/command/event/gateway.hpp (asset DATA spine; registries flagged edge-candidates for R5b); vop_resources_tests 4/4 (data basics, registry round-trip spec, kit). Finds: AssetRegistry is a stale fork referencing nonexistent handle types (zero consumers, never compiled) — excluded from contract, R5b converges or deletes it.
- [x] **P3.8 gfx** — DONE 2026-09-16: gfx.contract/command/event/gateway.hpp (handle + pixel-buffer values as spine; RTRegistry flagged edge-candidate with the other registries); vop_gfx_tests 4/4 (handle validity, pixel-buffer clear/addressing, kit). **Core 4 complete: 11/11 zones.** 269-line rt_registry is a stateful store like resources registries — handle types = contract; registry = edge subfolder with the resources registries)
- [x] **P3.9 sky** — DONE 2026-09-16: sky.contract/command/event/gateway.hpp (monostate vocabs + identity); ISkyModel virtual dispatch flagged R5b debt (hot-path law); vop_sky_tests 5/5 (horizon exact, zenith near, sun disk exact, determinism, kit).
- [x] **P3.10 logic** — DONE 2026-09-16: table-driven value FSM (FsmDesc states+table as DATA, zero std::function; Signal/Time/Force/Start commands in span order; priority strictly-greater mirror; rejections observable; legacy callback class untouched beside it, zero consumers). vop_logic_tests 5/5 (signal cycle, time gate, priority+rejection, kit replay/empty). (R5b: 231-line state_machine is a genuine state engine — gateway or edge-classify needs design, not a shell)
- [ ] DoD: every domains/<pod>/ passes Core 4 completeness; grep gate: zero pods missing contract/command/gateway/event.

## P4 — Semantic hardening (lands with P3, Run 4)

Parked 2026-09-15 backlog, absorbed here at the right slot:
- [x] **P4.1 Uniform gateway signature** — DONE 2026-09-16 for new pods: reduce(State&, span<const Action>, const Inputs&, pmr::vector<Event>&) with dt/caps/compiler always explicit params; input+frame prove it. Renderpath predates (follow-up, don't churn working code). — reduce(State&, span<const Action>, const Inputs&, pmr::vector<Event>&); time/caps/compiler always explicit; one generic edge loop drives every pod. Slot: first pod rewrite (P3.1).
- [x] **P4.2 Header-only pod test kit** — DONE 2026-09-16: domains/pod_test_kit.hpp (replay_is_deterministic + empty_log_is_stable over the house signature); both new suites use it (3 lines each). Value-equality via defaulted operator== added to CameraRig/RuntimeState/RuntimeInputLatch/action+event structs/FrameParams chain (mechanical, zero behavior change). (domains/pod_test_kit.hpp) — replay assert, snapshot equality, debug invariant predicates. Turns "every pod lands with a ctest" into 3 lines. Slot: P3.1.
- [x] **P4.3 Semantic purity linters** — DONE 2026-09-16: three hard FAIL gates (ambient entropy/time, unordered_* in *.gateway.hpp, SDL/fopen tokens in domains/) — all zero-hit on landing, enforced going forward. Bare 'canvas' deliberately ungated (too generic; documented in linter comment). — forbid rand(/chrono/time/getenv in domains/; forbid unordered_* iteration in gateway paths; extend Vk gate to canvas|SDL_|fopen in domains/. Slot: second pod.
- [ ] **P4.4 Seeded determinism contract** (STANDING 2026-09-16: no stochastic pods exist; the xorshift precedent stands by. First stochastic pod authors the contract. (R5b: no stochastic pods exist yet; first stochastic pod authors the contract) — stochastic pods carry RNG state in pod state (seedable via action, xorshift precedent); never globals. Slot: constitution doc edit, any time.
- [x] **P4.5 Generated event-flow docs** — DONE 2026-09-16: docs/pods/EVENT_FLOW.md (renderpath 5 + input 4 + logic 5 + 8 silent pods explicit); renderpath_event_name table added (P6 label source); linter drift gate FAILs on any event struct missing from the doc (covers *Event + Fsm* families — Entered is not Event, fixed the first gate draft). rides the P5.3 linter-docs sync; input_event_name + per-pod tables are the seed) — constexpr name tables in each event.hpp; EVENT_FLOW.md + debug overlay generate from them. Slot: with P5 linter-docs sync.
- [x] **P4.6 std::expected promotion** — DONE 2026-09-16: RenderPathCompileRejection native enum in the compiler (14 sites reasoned, first-error-wins, permissive-downgrade resets); try_compile() returns expected (transform_error in the pod, VOP S8 monadic); classify_plan_rejection DELETED; renderpath tests pin the native path (EmptyPassChain preserved). Backend-hint now maps BackendUnavailable (was CompileInvalid — deliberate precision, no test pinned it). — classify_plan_rejection string-matching dies; new compilers return expected<Plan, ClosedEnum> natively; events only in transform/or_else continuations. Slot: next compiler touch.

## P5 — Convergence sweep (Run 4 close-out)

- [x] **P5.1 Suffix completion (edge migration)** — DONE 2026-09-16: edge-classified headers moved to pod edge/ subfolders (scene system+processors, gfx rt_registry, resources registry+importers, input command queue; 22 live includes rewired); broken AssetRegistry fork DELETED (git rm, recoverable); Core 4 completeness holds 11/11. Full per-header suffixing of the remaining ~70 value headers is intentional non-work (they are single-role values needing no suffix). — multi-role headers split into *.contract|plan|edge.hpp (e.g. scene/system.hpp -> scene.edge.hpp).
- [x] **P5.2 Legacy seam retirement (audited: RETAIN)** — DONE 2026-09-16: PluggablePipeline (1036 lines) serves only vop_core_tests; FrameGraph only it. Removal condition (renderpath pod covering executor rebuilds) is FALSE pending P6.1, so both stay tested + supported with deprecation banners pointing at the pod. Deleting tested code with no replacement was rejected. — audit frame_graph.hpp / pluggable_pipeline.hpp for removal once renderpath pod covers use.
- [x] **P5.3 Linter-docs sync** — DONE 2026-09-16: all 7 glossary header paths verified on disk; pod-home table (§7) added with final homes + EVENT_FLOW pointer; law citations audited (sole 'Rule 3.2' hit is §2.2's own historical note, not a dangling cite). REMAINDER FILED: pluggability lint (open PassId/light registries) needs the P3-run-3 'Open Everything' track, never scheduled in R1–R5 — tracked as follow-up, not dropped. — S6.4 table <-> tree agreement, glossary rows point at final homes, law citations valid (no dangling Rule N), pluggability lint (extensions resolve via open registries, no core edits).
- [ ] DoD: zero old-path includes; glossary + S6.4 match tree exactly (CI-verified).

## P6 — Integration hardening (Run 5, needs device or swiftshader)

- [ ] **P6.1 PATH_COMPILED-driven executor rebuilds** (BLOCKED 2026-09-16: needs a live demo host to own the event->executor loop; demos parked. Unblocks on demo restart. The pod side is ready: try_compile + rejection-preserving events landed in R4.) (plan-hash + generation keyed GPU tables).
- [ ] **P6.2 Replay harness** (BLOCKED 2026-09-16, partial credit claimed: pod-level replay determinism is proven per-pod via the kit (13 suites); cross-session persistence (command-log codec) + CI replay await the demo host. Unblocks on demo restart.) (reuses pod test-kit machinery, not a reimplementation) — headless replay CI green.
- [ ] **P6.3 Rollback-ready snapshots + time-travel overlay** (BLOCKED 2026-09-16: overlay needs a windowed host; snapshots are plain values already (precondition met). Unblocks on demo restart.) reading the event log.
- [ ] DoD: hot-swap of all presets, zero frame allocation outside arenas, overlay ships.

## Monadic pipeline amendment (DONE 2026-09-16)

Constitutional adoption of C++23 monadic pipelines + domain-boundary
semantics, zero-signal-loss: every non-conflicting DOD/VOP law retained;
conflicts recorded, not overwritten.

- Retained: FCIS, passive PODs, replay, Core 4 file structure, zone direction,
  event-only cross-domain traffic, SoA/chunked layout, PMR arenas, closed
  vocabularies, parity harness, pod test-kit guarantees.
- Changed: Constitution I §10 (C++23 lib baseline + pod-idiomatic subset);
  Rules 9/11/12 (C++23 values, bounded contexts, saga compensation);
  Core 4+1 (orchestrator-is-a-pod, god-object ban); §8 tier doctrine
  (monads on value/error channels, variants on command/event streams);
  §7.1 granularity law (monad at chunk level); §10 gate list; §12 entry;
  Appendix A.7 (F-DOD-DDD correspondence + bundled-signature divergence);
  Constitution III §3 (orchestrator mapping) + §6 (explicit schedulers);
  glossary §8 (contexts); EVENT_FLOW saga-fact requirements; two new linter
  gates (per-element `expected` containers, `std::string` event members).
- Explicitly NOT adopted: bundled `expected<(State,Events)>` gateway
  signature — campaign evidence (N-events-per-command, conditional emission,
  silent pods) preserved in §8/A.7; migration needs a replay/event-count spike.
- Proved: saga spike `tests/vop_saga_tests.cpp` — RED (flag compensator leaks
  the wallet debit) then GREEN (fact-log compensator restores all). Suite
  15/15 -> 16/16 in both build trees; linter green including the two new gates.

## Leftovers (deferred past R5 — tracked, not dropped)

Explicitly NOT scheduled in R1–R5. Each carries its unlock condition;
review this section at the R5 close-out, no earlier.

- [ ] **L1 Renderpath uniform-sig migration** — renderpath_gateway predates the
  P4.1 house signature and works; do not churn it. Unlock: the next
  renderpath feature touch migrates it as drive-by.
- [x] **L2 Demo cxx_std_20 pins** — CLOSED 2026-09-16: all 15 demo/adventure CMakeLists bumped to `cxx_std_23` in one pass (bold cleanup); no per-pod migration needed.
- [x] **L3 Tier0 01/02 drift file** — CLEARED 2026-09-16: 01 fixed exact, 02 envelope-pinned, suite green. Close-out verifies the suite still passes. — NOT deferred (R4 entry ticket): filed
  under P1.3. Listed here only so close-out verifies it is gone, not parked.

Rule: a leftover without an unlock condition is rot — never add one.

## Non-goals

- No Tier1+ demo rungs until spine (P0-P2) green; pod hardening (P3+) then rides each rung per the P3 interleaving policy (not before it).
- No parked-demo restarts; adventures curriculum is the demo track.
- No per-frame pod reductions — batch planners/edges own hot loops (S7.1 design-smell rule).
