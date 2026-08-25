# PROJECT STRUCTURE — Final-Version Plan

> Goal (user, 2026-08-24): make this demo the blueprint for a future-proof
> personal C++ game project structure. Level CONTENT work is frozen
> temporarily; structure hardening comes first. Every phase ends with a green
> build + verify.sh gates.

---

## Current state (audited 2026-08-24)

```
tetris/
├── hello_3d_tetris.cpp      ← main + wiring (~940 lines, grew organically)
├── CMakeLists.txt           ← per-demo target
├── config/                  ← rules + campaign manifest + level headers
├── domains/{6 pods}         ← pure center (suffix laws enforced)
├── edges/{input,audio,rasterizer,ui,lua}
├── docs/{ARCHITECTURE,TODOS,STATUS,pods/*}
├── scripts/generate-event-flow.mjs   (added 2026-08-24)
└── verify.sh                ← headless gates
```

**Strengths to preserve:** pod suffix laws, purity greps, root-relative
includes, Lua sandbox isolation, headless verification culture,
docs-as-single-source-of-truth.

**Structural debts (what this plan fixes):**

| # | Debt | Why it hurts "final version" |
| --- | --- | --- |
| D1 | Main is a 940-line wiring monolith | every new pod/level edits god-file; merge conflicts; unreadable dataflow |
| D2 | No unit-test target | only end-to-end screenshot gates; reducers are pure but untested in isolation |
| D3 | Levels are C++ headers (`config/levels/*.hpp`) | content changes recompile the binary; contradicts the scripting thesis |
| D4 | Event vocabulary is implicit | no single enum/table of all facts; EVENT_FLOW doc is generated but code has no registry |
| D5 | Engine loop mixed into main | tick ordering is buried in main(); can't unit-test a frame |
| D6 | No shared library boundary between demos | each demo recompiles everything; shared pods impossible |

---

## Target structure (end state)

```
tetris/
├── CMakeLists.txt                    # thin: adds engine/, game/, targets
├── app/
│   └── main.cpp                      # ≤150 lines: parse args, own SDL window,
│                                     # call engine::run(), nothing else
├── engine/                           # REUSABLE ACROSS GAMES (the big win)
│   ├── include/engine/
│   │   ├── loop.hpp                  # fixed-timestep accumulator (pure)
│   │   ├── events.hpp                # EventBus: typed fact queue + fan-out
│   │   ├── command_buffer.hpp        # intent queue drained each tick
│   │   └── clock.hpp                 # FrameClock abstraction (headless-safe)
│   └── src/loop.cpp ...
│   └── CMakeLists.txt                # static lib `shs_engine`
├── game/                             # TETRIS-SPECIFIC assembly
│   ├── include/game/
│   │   ├── world.hpp                 # GameWorld struct-of-pods (owns all state)
│   │   ├── step.hpp                  # step(world, commands, dt) -> StepResult
│   │   │                             #   THE tick: session→matrix→progression→
│   │   │                             #   powerups→environment→mission, event fan-out
│   │   ├── script_host.hpp           # ScriptHost: lua.edge + hook registry
│   │   └── stage.hpp                 # StageDef loaded from data (see P3)
│   └── src/step.cpp ...              # extracted from today's main()
│   └── CMakeLists.txt                # static lib `shs_tetris_game`
├── render/                           # tetris PRESENTATION assembly (thin)
│   ├── include/render/present.hpp    # present(world_snapshot, fx_plan) ->
│   │                                 #   shs::Canvas draw list (tiles, camera,
│   │   #                             #   HUD projection wiring)
│   └── src/...                       # tiling/job orchestration lives here;
│                                     # ALL pixel work delegates to shs-renderer-lib
├── domains/                          # UNCHANGED layout (pods stay header-only)
├── edges/input/ edges/audio/ edges/lua/   # device edges (unchanged)
├── config/rules.hpp                  # base tuning (unchanged)
├── assets/
│   ├── campaign/campaign.lua         # manifest becomes DATA (P3)
│   └── levels/<id>/level.lua         # per-stage: rules overrides + goals
├── tests/                            # NEW (P2)
│   ├── test_matrix.cpp               # reducer unit tests (doctest)
│   ├── test_progression.cpp
│   ├── test_step.cpp                 # whole-frame tests: command → facts
│   └── CMakeLists.txt                # ctest integration
├── scripts/                          # tooling (exists)
└── docs/                             # exists
```

**Key rule:** `engine/` knows nothing about tetris. `game/` knows nothing
about SDL. `app/main.cpp` is the ONLY file allowed to touch both.

---

## Phases (each ends build-green + verify PASS)

### P0 — Baseline safety net (half day)

Before moving anything: capture current behavior as executable truth.

- [ ] Add doctest (or Catch2) via vcpkg; create `tests/` CMake target
- [ ] Write ~15 reducer unit tests FIRST against existing behavior
      (matrix: move/rotate/kick/lock/sweep; progression: scoring tiers,
      combo ladder, B2B; session: screen FSM transitions) — these pin
      semantics so refactors cannot silently change gameplay
- [ ] Wire ctest into verify.sh as a new gate (UNIT=PASS)

### P1 — Extract the engine loop + game step (2–3 days) — kills D1+D5

The heart of the plan. Move the tick out of main().

- [ ] `engine/include/engine/loop.hpp`: fixed-step accumulator
      `run_loop(clock, fixed_dt, step_fn)` — pure, header-only, no SDL
- [ ] `game/include/game/world.hpp`: aggregate struct
      `GameWorld { matrix::State; progression::ScoreState; powerups;
      environment; session; fx; }` — plain data, swappable snapshots
- [ ] `game/include/game/step.hpp`: `StepResult step(GameWorld&,
      std::span<const Command>, float dt, arena)` — the ordered tick copied
      verbatim from main()'s playing branch (session → frozen check → boot
      queue → matrix → progression → powerups → environment → fx)
- [ ] main.cpp shrinks to: arg parse → SDL window → load stage → run_loop
- [ ] Headless screenshot paths keep byte-identical output (verify DETERMINISM
      gate proves the move changed nothing)

### P2 — Unit test expansion (1–2 days, overlaps P1)

- [ ] Whole-frame tests through `step()`: inject command sequence, assert
      resulting facts (this replaces what only screenshots proved before)
- [ ] Input-edge tests: synthetic SDL-free harness — feed KEYDOWN/KEYUP
      timelines into InputEdge, assert DAS/ARR timing and soft-drop flag
      (closes TODOS Part 6 V1–V3 properly)
- [ ] Script purity test in CI shape: iterate domains/*/scripts/*.lua,
      assert forbidden globals absent (port of verify.sh grep into ctest)

### P3 — Levels & campaign become DATA (2 days) — kills D3

Content must not require recompilation. This is the scripting thesis applied
to structure itself.

- [ ] Define the level-data contract: one Lua table per level
      `{ id, name, unlock_after, rules_overrides = {...}, script = "path or nil", goals = {...} }`
- [ ] `assets/campaign/campaign.lua` — ordered manifest, loaded at boot
- [ ] `config/levels/*.hpp` deleted; `Stage` struct gains a loader
      (`game/include/game/stage.hpp`) that reads Lua → Rules via the existing
      get_config bridge pattern
- [ ] Fallback rule: missing/broken level file → clean error + marathon
      defaults (never crash)
- [ ] verify.sh gains: LEVELS_LOAD=PASS (all levels parse), plus keeps
      determinism gates (proves data-driven == header-driven output)

### P4 — Event registry (1 day) — kills D4

- [ ] Single source of truth for facts:
      `domains/shared/event_ids.hpp` — `enum class FactId : uint16_t` +
      `constexpr std::array names` (generated later if desired)
- [ ] Reducers switch from ad-hoc struct pushes to `(FactId, payload)` pairs
      OR keep structs but register them (choose minimal churn: registry first,
      migration optional)
- [ ] generate-event-flow.mjs reads the registry instead of grepping —
      generator becomes exact, not heuristic
- [ ] Gate: every emitted FactId has ≥1 registered producer (the JS twin's
      undeclared-producer warning, now compile-time adjacent)

### P5 — Shared library boundary (1–2 days) — kills D6

Two shared libs at REPO level, both already justified:

- [ ] `cpp-folders/libs/engine/` (`shs_engine`): loop/event-bus/clock from P1 —
      snake/fps/plane adopt at their own pace
- [ ] `shs-renderer-lib` STAYS the single rendering dependency. Tetris's
      `render/` is a thin presentation ASSEMBLY over it (tile scheduling,
      camera, HUD projection), never a fork of rasterization code. Any
      rasterizer improvement you make in shs-renderer-lib automatically
      benefits every demo — that is the upgrade path for your software
      renderer development going forward.
- [ ] Dependency rule recorded in ARCHITECTURE.md:
      app -> game -> {engine, domains, edges} -> shs-renderer-lib (only from
      render/ + ui edge). Game logic NEVER includes shs headers.

### P6 — Documentation sync (half day)

- [ ] ARCHITECTURE.md Part III rewritten for the new tree (as-built section)
- [ ] Doc map gains engine/game/render ownership table
- [ ] STATUS.md records the migration with before/after line counts
- [ ] pods/EVENT_FLOW.md regenerated from the P4 registry

---

## Does this plan deliver "future-proof + fully scriptable"? (audit)

| Requirement | Where it lands | Status |
| --- | --- | --- |
| SHS renderer stays THE renderer, keeps evolving independently | Tetris consumes shs-renderer-lib via a thin render/ assembly; no raster code forks into the demo; lib upgrades flow to all demos automatically | covered (P1 render split + P5 rule) |
| New games reuse the skeleton | engine/ + pods/ playbook + porting checklist; snake/fps can adopt shs_engine (P5) | covered |
| Fully scriptable RULES | existing: 4 pods x Lua hooks behind sandboxed evaluator (blitz/canyon/storm/overseer) | shipped pre-plan |
| Fully scriptable LEVELS | P3: campaign manifest + levels become Lua data under assets/; zero recompiles for content | planned (P3) |
| Fully scriptable GOALS/MISSIONS | mission pod + event-batch bridge + Lua predicate DSL — deliberately sequenced AFTER this plan (needs step() extraction from P1 + data-driven levels from P3 to land cleanly); tracked TODOS Part 7 G1-G4 | sequenced, not forgotten |
| Scripting safety | sandbox purity gates stay; P2 ports them into ctest so every gate runs in one command | covered (P2) |
| Determinism as a contract | P0 behavioral pins + byte-compare gates after each phase; L3 flake made reproducible by P2 frame tests | covered |

Verdict: the plan completes the STRUCTURE half of future-proofing; the
SCRIPTING half finishes with the post-P3 mission work (Part 7). Together they
are the full vision.

## Explicitly deferred (recorded, not forgotten)

| Item | Why deferred |
| --- | --- |
| Mission/goal pod + Lua predicate DSL (MISSIONS.md port) | depends on P1 step extraction + P3 data-driven levels; lands cleanly AFTER this plan. Tracked in TODOS Part 7 |
| Entity-array world pods / AI perception | documented in pods/FPS_EXAMPLE.md + AI_PODS.md for future games on this skeleton |
| Rasterizer FP-nondeterminism fix | pre-existing flake, separate issue; P2's frame tests will make it reproducible |
| Netcode/rollback | architecture already supports it; zero demand yet |

## Risk notes

- The biggest regression risk is P1's tick extraction. Mitigation: P0's
  behavioral pins + byte-compare determinism gates after every move.
- P3 changes how stages boot; keep the old header path behind a flag until
  the Lua path passes ALL existing gates, then delete.
- Total estimate: **6–9 focused days**, each independently shippable.
