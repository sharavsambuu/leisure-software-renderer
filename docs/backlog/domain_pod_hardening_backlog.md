# Domain Pod Hardening Backlog — shs-renderer-lib

> Status: active (2026-09-16). Source: lib review vs Constitutions I/II/III + rollout roadmap P3-P6 + backlog "POD Semantics Hardening" (parked 2026-09-15).
> Law precedence: Constitution II S2.1 + Rule 10, canon S6.1-S6.2, precedence S2.2. Roadmap is schedule, not law.
> Verification after every item: `build/` ctest 5/5 green (grows as pods land) + `check_vop_boundaries.sh` green.

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
- [x] **P0.3 C++ standard decision** — DONE 2026-09-16 for lib (6 targets cxx_std_20 -> cxx_std_23, build + 5/5 green, GCC 13.3). Demos still pin cxx_std_20; follow in P3 per-pod commits. DoD met for lib; S8 prerequisite note still stale for demos (follow-up).

## P1 — Kill the shadow tree (days, Run 1+2)

- [ ] **P1.1 Retire `shs/pipeline/` (2 files) + `shs/rhi/` (13 files)** — absorbed monolith lives beside `shs/execution/`, outside linter coverage; all Vk-outside-drivers hits come from here. Move to `execution/` + delete old dirs, or mark legacy/ with sunset + extend linter to cover them. DoD: single tree (core/ domains/ execution/ memory/ containers/); no Vk outside execution/rhi/drivers/vulkan/.
- [x] **P1.2 GPU-free proof** — SHS_HAS_VULKAN gate + backend-factory fallback verified in CI (configure -DCMAKE_DISABLE_FIND_PACKAGE_Vulkan=TRUE -> build + ctest green). DoD: GPU-free build documented + green. DONE 2026-09-16: /tmp/swr-novk-r1 configure EXIT=0 (vk_driver_tests correctly skipped), lib 4/4 green, t0 _sw pair builds. Prerequisite: pinned slangc v2026.17.1 in WSL (currently missing — _vk targets unbuildable here); GPU-free proof covers the _sw half regardless.
- [x] **P1.3 Tier0 parity harness promotion (harness landed, GATE RED)** — DONE 2026-09-16: tools/t0_parity.py (stdlib-only PNG decode, exact% + 1-LSB tolerance + ASCII diffmap) runs over fresh-vs-fresh PNGs: 01 FAIL 70.94% (max 8), 02 FAIL 11.36% (max 217), 03/04 TOLERANCE-ONLY (max 1), 05 PASS exact. Docs 0.00%-everywhere does NOT reproduce (_vk runs headless here, so this is genuine drift, not stale artifacts). Follow-up filed: Tier0 parity regression (R4 track) — diagnose 01/02 before any Tier1 rung. — promote /tmp/t0final tooling (t0_parity + diffmap) into the repo as the cross-backend equivalence gate (tier0 lessons prescription). DoD: ctest parity target renders both halves (or _sw-only where toolchain absent) and diffs with a per-pass tolerance table. Blocks Tier1.

## P2 — Direction-law debt (per pod, Run 4 pre-req)

Grandfathered domains/ -> execution/ includes (linter INFO today, must reach 0):
- [ ] **P2.1 camera/free_camera.hpp -> platform_input** — split pure math (contract+plan) from platform edge.
- [ ] **P2.2 scene/system_processors.hpp -> pluggable_pipeline** — legacy-seam audit; plan vs executor split.
- [ ] **P2.3 input/value_actions.hpp + input/command.hpp -> app/runtime_state** — vocab stays in domains/input, latch lives in execution/app.
- [ ] **P2.4 sky/skybox_renderer.hpp -> job/parallel_for** — plan function + edge dispatch split.
- [ ] DoD: grandfather list empty; linter grandfather block deleted.

## P3 — Core 4 per pod (weeks, Run 4 main track)

Rule: one pod per commit; each lands with headless ctest (replay assert + snapshot round-trip + invariant), linked only to shs::renderer-values. Empty vocabs explicit (variant<monostate>), events carry no std::string.
Interleaving policy (2026-09-16): pods harden WITH their curriculum rung, not ahead of it — demo pair -> parity -> ingest one operator (rule of three) -> harden the touched pod in the same commit. Do not complete all of P3 before Tier1. Order by dependency:
- [ ] **P3.0 Tier1 pilot (rung 8 normal mapping)** — full loop: pair -> parity -> TBN/compare-op ingestion -> harden geometry+lighting pods. DoD: loop proven before committing to all of Tier1-6.
- [ ] **P3.1 input** (already action-shaped — start here)
- [ ] **P3.2 frame** (contract values; empty action/event legal per S6.1)
- [ ] **P3.3 camera** (contract + plan transforms; absorbs P2.1)
- [ ] **P3.4 geometry** (shapes/jolt adapters = contract + plan; empty command vocab)
- [ ] **P3.5 lighting** (sets/types = contract; culling = plan; runtime = reducer)
- [ ] **P3.6 scene** (objects/culling/instance -> contract+plan; world/system -> edge subfolder)
- [ ] **P3.7 resources** (value types + load planners; absorbs P0.2)
- [ ] **P3.8 gfx** (handle types = contract; registry = edge subfolder)
- [ ] **P3.9 sky** (contract + plan; absorbs P2.4)
- [ ] **P3.10 logic** (fsm/state_machine -> contract + reducer or edge-classify)
- [ ] DoD: every domains/<pod>/ passes Core 4 completeness; grep gate: zero pods missing contract/action/reducer/event.

## P4 — Semantic hardening (lands with P3, Run 4)

Parked 2026-09-15 backlog, absorbed here at the right slot:
- [ ] **P4.1 Uniform reducer signature** — reduce(State&, span<const Action>, const Inputs&, pmr::vector<Event>&); time/caps/compiler always explicit; one generic edge loop drives every pod. Slot: first pod rewrite (P3.1).
- [ ] **P4.2 Header-only pod test kit** (domains/pod_test_kit.hpp) — replay assert, snapshot equality, debug invariant predicates. Turns "every pod lands with a ctest" into 3 lines. Slot: P3.1.
- [ ] **P4.3 Semantic purity linters** — forbid rand(/chrono/time/getenv in domains/; forbid unordered_* iteration in reducer paths; extend Vk gate to canvas|SDL_|fopen in domains/. Slot: second pod.
- [ ] **P4.4 Seeded determinism contract** — stochastic pods carry RNG state in pod state (seedable via action, xorshift precedent); never globals. Slot: constitution doc edit, any time.
- [ ] **P4.5 Generated event-flow docs** — constexpr name tables in each event.hpp; EVENT_FLOW.md + debug overlay generate from them. Slot: with P5 linter-docs sync.
- [ ] **P4.6 std::expected promotion** — classify_plan_rejection string-matching dies; new compilers return expected<Plan, ClosedEnum> natively; events only in transform/or_else continuations. Slot: next compiler touch.

## P5 — Convergence sweep (Run 4 close-out)

- [ ] **P5.1 Suffix completion** — multi-role headers split into *.contract|plan|edge.hpp (e.g. scene/system.hpp -> scene.edge.hpp).
- [ ] **P5.2 Legacy seam retirement** — audit frame_graph.hpp / pluggable_pipeline.hpp for removal once renderpath pod covers use.
- [ ] **P5.3 Linter-docs sync** — S6.4 table <-> tree agreement, glossary rows point at final homes, law citations valid (no dangling Rule N), pluggability lint (extensions resolve via open registries, no core edits).
- [ ] DoD: zero old-path includes; glossary + S6.4 match tree exactly (CI-verified).

## P6 — Integration hardening (Run 5, needs device or swiftshader)

- [ ] **P6.1 PATH_COMPILED-driven executor rebuilds** (plan-hash + generation keyed GPU tables).
- [ ] **P6.2 Replay harness** (reuses pod test-kit machinery, not a reimplementation) — headless replay CI green.
- [ ] **P6.3 Rollback-ready snapshots + time-travel overlay** reading the event log.
- [ ] DoD: hot-swap of all presets, zero frame allocation outside arenas, overlay ships.

## Non-goals

- No Tier1+ demo rungs until spine (P0-P2) green; pod hardening (P3+) then rides each rung per the P3 interleaving policy (not before it).
- No parked-demo restarts; adventures curriculum is the demo track.
- No per-frame pod reductions — batch planners/edges own hot loops (S7.1 design-smell rule).
