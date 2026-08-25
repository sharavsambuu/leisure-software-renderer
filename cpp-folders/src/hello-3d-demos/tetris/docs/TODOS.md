# Todos — Living Tracker

Goal: a multi-domain-pod structure usable as a template for other game demos,
plus Lua-based scripting capabilities — and now a concrete **level & mode
campaign** that exercises the full C++↔Lua spectrum, with per-level **GUI and
FX variety** (Part 4).

This file is the **living progress tracker**. Update the checkboxes as work
lands; keep prose minimal and point at the owning doc instead of duplicating.

## Doc map (single source of truth per concern)

| Doc | Owns |
| :--- | :--- |
| `ARCHITECTURE.md` | Pod theory (Part I), Lua philosophy (Part II), as-built tree/dataflow/ownership + Lua design rules + conventions (Part III) |
| `TODOS.md` | This file — status tables + roadmap + campaign proposals |
| `STATUS.md` | Verification log: build results, headless gates, DoD, pitfalls, migration history |

---

## Status (as of 2026-08-22)

| Item | State |
| :--- | :--- |
| Pod 1 `matrix` | ✅ DONE — contract/action/event/reducer; pure center, zero scoring refs |
| Pod 2 `progression` | ✅ DONE — event-fed scoring, combos, levels, victory; Lua seam isolated at `compute_line_clear_score()` |
| Pod 3 `spatial_fx` | ✅ DONE — SoA particles, camera spring, scene planner (`spatial_fx.plan.hpp`) |
| Pod 4 `powerups` | ✅ DONE (2026-08-23) — cadence scheduler + scripted special pieces (Part 4 · L4) |
| Pod 5 `environment` | ✅ DONE (2026-08-23) — encounter phases + mood/dim/crowd state, finale diorama planner (Part 4 · L5) |
| Edges `input` / `audio` / `rasterizer` / `ui` | ✅ DONE — one subdirectory per edge (`edges/<name>/tetris.<name>.hpp`) |
| Edge `lua` | ✅ DONE — sandboxed stateless evaluator wired into the loop via `ScriptHooks` function-pointer bridges; blitz script boots from the campaign manifest |
| Pod 6 session | DONE 2026-08-23 — TITLE/LEVEL_SELECT/PLAYING/PAUSED/RESULTS state machine; pure reducer, zero SDL refs |
| Thin main + `verify.sh` | ✅ DONE — determinism / behavioral-delta / purity gates all PASS |
| Docs integration | ✅ DONE 2026-08-22 — deduplicated to 3 files; see STATUS.md §0 |

---

# Part 1: The Canonical 5-Pod Suite

```
                                  [ 1. INPUT EDGE ]
                             (Tokenizes OS events to Actions)
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 2. PURE VALUE CENTER                                        │
│                                                                                             │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐      │
│   │ Pod 1: MATRIX (`domains/matrix/`)                                                │      │
│   │ - Pure 10x22 grid spatial simulation, SRS wall kicks, gravity step, lock delay   │      │
│   │ - Emits: PieceLocked, LinesCleared(rows), HardDropImpact                         │      │
│   └────────────────────────────────────────┬─────────────────────────────────────────┘      │
│                                            │                                                │
│       ┌────────────────────────────────────┼────────────────────────────────────┐           │
│       │ (Event Span)                       │ (Event Span)                       │           │
│       ▼                                    ▼                                    ▼           │
│   ┌───────────────────────────┐ ┌──────────────────────────┐ ┌──────────────────────────┐   │
│   │ Pod 2: PROGRESSION        │ │ Pod 3: SPATIAL FX        │ │ Pod 4: POWER-UPS / HAZARD│   │
│   │ - Scoring & combos        │ │ - 3D Voxel shatter (SoA) │ │ - Bomb blocks, laser rows│   │
│   │ - Modes (Sprint / Blitz)  │ │ - Camera spring shake    │ │ - Glitch matrix modifiers│   │
│   │ - [LUA: Mode Rules Engine]│ │ - Floating 3D popups     │ │ - [LUA: Power-up Rules]  │   │
│   └─────────────┬─────────────┘ └────────────┬─────────────┘ └────────────┬─────────────┘   │
│                 │                            │                            │                 │
│                 └────────────────────────────┼────────────────────────────┘                 │
│                                              ▼                                              │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐      │
│   │ Pod 5: ENVIRONMENT (`domains/environment/`)                                      │      │
│   │ - Reactive lighting (Normal Cyan -> Danger Crimson -> Victory Gold)              │      │
│   │ - Low-poly diorama background, animated neon ground pedestal                     │      │
│   └──────────────────────────────────────────┬───────────────────────────────────────┘      │
│                                              │                                              │
│                                              ▼                                              │
│                            [ Combined PipelineExecutionPlan ]                               │
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                               3. EFFECT EXECUTION EDGES                                  │
│   ├─ Audio Edge: SPSC lock-free atomic ring procedural synth                             │
│   ├─ Lua Edge: Thread-local stateless `lua_State*` pool                                  │
│   └─ Render Edge: Multi-threaded tiled software rasterizer & swapchain presentation      │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

*(Pods 4 and 5 above are the planned additions; pods 1–3 and all three execution-edge
families already exist in code — see the Status table.)*

### Why these 5 Pods are optimal:

| Domain Pod | Core Responsibility | DOD / VOP Showcase Feature |
| :--- | :--- | :--- |
| **1. `matrix`** | Grid, collision, SRS rotations, pieces | Pure 2D spatial array math, deterministic gravity accumulator. |
| **2. `progression`** | Score, combos, level curves, game modes | Event-driven rules engine with **Stateless Lua mode evaluators**. |
| **3. `spatial_fx`** | 3D particles, shockwaves, camera spring | **Structure of Arrays (SoA)** particle physics, spring dampening. |
| **4. `powerups`** | Special cyber blocks (Bomb, Laser, Freeze) | Matrix mutation requests, cooldown timers, **Stateless Lua effects**. |
| **5. `environment`** | Diorama, pedestal, reactive mood lighting | Low-poly mesh batching, ambient color state interpolators. |

---

# Part 2: Stateless Lua Integration (wiring plan)

### The Constitution II Rule for Lua:
> **"Lua scripts must never hold mutable pointers to C++ objects. Lua functions must be pure stateless reducers: taking an immutable C++ data snapshot, performing game logic, and returning an explicit value result."**

```
┌─────────────────────────┐          ┌───────────────────────┐          ┌──────────────────────────┐
│ Immutable C++ Snapshot  ├─────────►│  Stateless Lua Script ├─────────►│ C++ Value Mutation Patch │
│ (Lines, Combo, Time, dt)│          │   (Pure Rule Function)│          │ (ScoreDelta, Events, New)│
└─────────────────────────┘          └───────────────────────┘          └──────────────────────────┘
```

### 2.1 File Placement for Lua Scripts

Scripts live inside their respective domain pod folder under a `scripts/` directory:

```text
domains/
├── progression/
│   ├── progression.contract.hpp
│   ├── progression.reducer.hpp
│   └── scripts/
│       ├── blitz_mode.lua          # 1. Custom scoring & timer rules
│       └── sprint_40lines.lua      # 2. Speedrun validation rules
│
└── powerups/
    ├── powerups.contract.hpp
    ├── powerups.reducer.hpp
    └── scripts/
        ├── bomb_piece.lua          # 3. 3x3 explosion radius & decay rules
        └── laser_row.lua           # 4. Instant row vaporization logic
```

### 2.2 Canonical Example: Pure Lua Scoring Rule (`blitz_mode.lua`)

```lua
-- Pure stateless Lua decision rule (Zero globals, zero side-effects)
local BlitzRules = {}

function BlitzRules.calculate_score(level, lines_cleared, combo_count, is_tspin)
    local base_scores = { [1] = 100, [2] = 300, [3] = 500, [4] = 800 }
    local base = base_scores[lines_cleared] or 0
    
    if is_tspin then
        base = base * 2
    end
    
    local combo_bonus = combo_count * 50 * level
    local score_added = (base * level) + combo_bonus
    
    -- Return explicit value table
    return {
        score_added  = score_added,
        level_up     = (lines_cleared >= 4),
        danger_alert = (lines_cleared == 0 and combo_count == 0)
    }
end

return BlitzRules
```

### 2.3 C++ Side

The evaluator lives in the build at
`edges/lua/lua.edge.hpp` (`tetris::lua_edge::StatelessLuaEvaluator`) —
script text + plain-value inputs → plain-value result struct. It IS wired into
the loop (2026-08-22): main boots the stage's script from the campaign
manifest and adapts it to `progression::ScriptHooks` function pointers; pods
stay Lua-free. Design rules (edge placement, sandboxing/determinism,
build wiring, verify.sh extensions) are owned by `ARCHITECTURE.md` Part III §4.

---

# Part 3: Project Directory Layout

Single-sourced in **`ARCHITECTURE.md` Part III §1 (as-built tree)** — do not re-list it here.

---

# Part 4: Level & Mode Campaign Proposal (tracker)

Proposed 2026-08-22; GUI/FX variety enriched same day. Five levels/modes that
deliberately span the implementation spectrum — pure C++ → hybrid → pure-Lua
content/mechanics/orchestration — plus the meta-layer (menus, progression,
congrats) the demo currently lacks. Each block below is the progress tracker;
tick boxes as work lands and record gate results in `STATUS.md`.

**GUI/FX ground rule:** every item below stays inside the existing seams —
GUI elements are pure projections of snapshots drawn by the ui edge; FX are
event-fed recipes in `FxState`/scripts consumed by the planner; environment
mood is interpolated state. All converge into ONE `PipelineExecutionPlan`.
No level adds architecture; levels add data, recipes, and listeners.

**Camera (2026-08-23):** framing is per-level config data — `Rules.camera`
(`config/camera.hpp`: eye/target/FOV/clip planes) set in each stage's
`make_rules()`; the planner applies FX shake/pulse as offsets on top. L1/L2
share the pulled-back default (~26 units vertical coverage, full board +
HUD clearance), L3 a wide diorama shot, L4 a low dramatic angle.

### Summary matrix

| # | Level / Mode | Tier | C++ share | Lua share | Pods touched | Architectural delta |
|---|---|---|---|---|---|---|
| L1 | Marathon Classic | Pure C++ | 100% | — | 1,2,3 | none (exists) |
| L2 | Blitz 120 | Hybrid rules | ~90% | scoring + clock | 1,2,3 | wire lua.edge |
| L3 | Garbage Canyon | Lua generation | ~85% | board generator | 1,2,3,+env palette | initial-board injection seam |
| L4 | Cyber Storm | Lua mechanics | ~70% | powerup effects | 1,2,3,**+4** | build powerups pod |
| L5 | Encore Finale | Lua orchestration | ~60% | encounter + mood | 1,2,3,**+5** | build environment pod |
| M1–M3 | Menus / progression / congrats | C++ logic (+ optional flavor scripts) | session pod + ui edge | new `session` pod + campaign manifest |

---

### Shared GUI & FX vocabulary (build once, configure per level)

Implement these primitives ONCE in the owning edge/pod; every level then just
configures them via config data or script recipes. No per-level widget code.

**GUI primitives (ui edge, all pure projections of snapshots):**
panel · stat digits · progress bar · meter w/ tier ticks · banner (flash/slide) ·
icon chip · vignette overlay · letterbox bars · popup floater (+N s, score) ·
carousel card · countdown digits · radial dial / pip row

**FX primitives (spatial_fx recipes, event-fed):**
burst (voxel shatter) · ring (shockwave) · beam (sweep) · flash (screen) ·
shake/kick (camera spring) · rumble (low-amplitude sustained shake) ·
rain (falling spawner) · confetti/firework (celebration SoA) · trail (motion) ·
dissolve (per-cell fade) · dust wave · spotlight cone

**Environment primitives (pod 5):**
mood color interpolator · diorama backdrop batch · pedestal animation ·
light-strip pulse · weather layer (lightning/flicker)

---

### L1 · Marathon Classic — Tier 1: Pure C++ baseline `[EXISTS]`

The existing marathon_01: fixed victory target, standard gravity curve, all
reducers native C++. Serves as the determinism reference run for every later tier.

Core:
- [x] `config/levels/marathon_01.hpp` — victory target, start level, palette
- [x] All three pods pure C++ (no scripting anywhere)
- [x] Gates green: determinism byte-compare + behavioral delta (STATUS.md)
- [x] Registered as stage 1 in the campaign manifest (`config/campaign/main_campaign.hpp`)

GUI polish:
- [x] Level-up flash banner + brief palette shift on `LEVEL_UP` event (HudState banner drives card accents gold)
- [x] Danger vignette when stack height crosses warning row (projection of grid state; breathing crimson bands)

FX polish:
- [x] Tetris (4-line clear) camera pulse + oversized burst recipe (dolly punch + 1.6× energy + gold flecks)
- [x] Combo streak floating 3D popup ("COMBO ×N") from `COMBO_STREAK` events (rising/fading HUD floater)

---

### L2 · Blitz 120 — Tier 2: Hybrid (Lua rules, C++ engine) `[DONE 2026-08-22]`

2-minute sprint, aggressive combo scoring, T-spin bonus. First consumer of the
wired lua.edge; designers retune the whole economy by editing one script.

Core:
- [x] Wire lua.edge into main loop: load scripts at boot, evaluate per tick (ARCHITECTURE.md Part III §4.1–4.2)
- [x] Build wiring: vcpkg `lua` entry + guarded `find_package` (Part III §4.4) — builds green with AND without Lua
- [x] `domains/progression/scripts/blitz_mode.lua` — `calculate_score(...)` (Part 2.2 shape) + `evaluate_clock(time_left, stack_height) -> {danger_alert, hurry}`
- [x] Route `compute_line_clear_score` through the script; keep C++ fallback when no script set (`ScriptHooks`, null ⇒ native)
- [x] `ScoreState` gains mode fields: `time_left`, `mode_id` (progression contract)
- [x] `verify.sh`: `--script=<file>` flag + determinism double-run WITH scripting active (Part III §4.5)
- [x] Smoke test: blitz target/score override assertion (`--expect-target-score=20000` gate, PASS)

GUI (timer-driven identity):
- [x] Large countdown digits: amber → red < 30s → pulsing < 10s
- [x] Time-bonus popup floaters ("+5s") on bonus clears
- [x] Combo meter bar with tier ticks (feeds off `combo_count`)
- [x] HURRY! flashing banner + screen-border pulse in final 10 seconds
- [x] RESULTS time breakdown panel (clears vs bonuses vs penalties)

FX (clock spectacle):
- [x] Threshold shockwave ring emitted from the board every 30-second tick
- [x] Spark trail on hard drops (speed feel, `HARD_DROP_SLAM`-fed)
- [x] Golden burst + slow-mo zoom on the final clearing line ("photo finish"; dolly-punch zoom approximation)
- [x] Amber environment mood that intensifies as the timer drains (`fx.mood_intensity` wire, pod-5 embryo)

---

### L3 · Garbage Canyon — Tier 3: Pure-Lua level GENERATION `[DONE 2026-08-23]`

Pre-ruined board (staggered garbage towers with holes); win = excavate 20 lines.
Level content authored entirely in a script; C++ pods execute it.
Full delivery log + gate results: STATUS.md §0e.

Core:
- [x] Matrix injection seam: `StampInitialBoardIntent` plain-data command fills grid + restart backup — no logic in the schema
- [x] `domains/matrix/scripts/garbage_canyon.gen.lua` — `CanyonGen.generate(difficulty, seed) -> {rows, target_lines, time_limit, mode_id, seed_tag}` (Use Case 4 pattern, ARCHITECTURE.md Part II); MINSTD LCG, sandbox-pure
- [x] Determinism gates in verify.sh: SEED_DIFF PASS (777 vs 778 boards differ); same-seed byte-identical currently FAILS while the diorama embryo renders — known open pitfall (STATUS.md §0e), suspected rasterizer depth-tie ordering with the new overlapping geometry
- [x] `config/levels/garbage_canyon.hpp` — dusk palette + stage layout constants; registered as campaign stage 3

GUI (excavation identity):
- [x] Excavation progress bar (lines cleared / target)
- [x] Depth gauge: highest garbage-row marker + danger stripes near the ceiling
- [x] Seed/variant tag in corner (daily-challenge identity)
- [x] Dust overlay tint when clearing rows near the floor

FX (dig feel):
- [x] Dust bursts + rubble debris in brown/gray palette on garbage locks
- [x] Screen rumble scaled to garbage mass cleared (multi-row collapses hit harder)
- [x] Pebble-trickle particles falling from disturbed rows above the clear
- [x] Deep thud audio + horizontal dust wave on 3+ row collapses

Environment:
- [x] Dusk/desert mood curve; flickering torch-style point lights; canyon-silhouette diorama backdrop

---

### L4 · Cyber Storm — Tier 3: Pure-Lua MECHANICS → earns Pod 4 `[DONE 2026-08-23]`

Special pieces drop occasionally: Bomb (3×3 blast), Laser Row (vaporizes a row),
Freeze (stops gravity 5s). Whole gameplay mechanics as hot-reloadable scripts;
add future powerups without touching C++.

Core:
- [x] New pod `domains/powerups/`: contract (PowerupSnapshot cadence state + SpecialRuling/ApplyRulingIntent), action (ApplyRulingIntent), event (SPAWN_SPECIAL_REQUESTED / POWERUP_TRIGGERED), reducer (pure `reduce_powerups` step)
- [x] Matrix seam (not "untouched" — a minimal typed seam): `PieceType::Bomb/LaserRow/Freeze` (9–11), `QueueSpecialIntent`, `ClearCellsIntent`, `FreezeGravityIntent`, `SPECIAL_LOCKED` event, `gravity_freeze` in the snapshot. The grid is still only ever mutated by the matrix reducer through plain commands
- [x] `scripts/cyber_storm.lua` — one stage script with three pure entry points: `CyberRules.get_config()` (cadence/freeze numbers), `CyberRules.decide_spawn(count, armed) -> special_type`, `CyberRules.on_special_lock(type, x, y, grid) -> {cells[], freeze_seconds, fx_id}` (single script instead of three files; per-powerup scripts can be split out later)
- [x] Main wiring: cadence counts PIECE_SPAWNED; on threshold the script's decide_spawn queues a special via `QueueSpecialIntent`; on SPECIAL_LOCKED the ruling becomes raw ClearCells/FreezeGravity commands riding the boot queue into the next reduce pass
- [x] Audio: SND_BLAST / SND_ZAP / SND_FROST voices through the SPSC ring, mapped from POWERUP_TRIGGERED
- [x] Purity gates extended in verify.sh: script purity glob covers `domains/powerups/scripts/*.lua`; stage-4 smoke gate `--expect-special-every-n=5` + CYBER_DETERMINISM pair

GUI (powerup cockpit):
- [x] Powerup cadence pip row (progress to next special, projection of powerup state)
- [x] Incoming-special warning icon overlaid on the next-queue display
- [x] Active-effect status chip (freeze timer counting down from the matrix snapshot)
- [x] Hit-marker flash on bomb/laser lock (hud.flash); glitch-flicker badge deferred to L5 polish

FX (per-powerup signature):
- [x] Bomb: voxel burst + shockwave ring + white flash + camera kick (fx_id=1 recipe)
- [x] Laser: horizontal beam sweep + scanline glow + row dissolve (fx_id=2 recipe)
- [x] Freeze: frost vignette + ice-crystal drift + desaturation pulse (fx_id=3 recipe)
- [ ] Glitch modifier: RGB-split flicker on affected rows for ~0.5s (deferred — needs per-row render hook)

Environment:
- [x] Neon cyberpunk environment (env_neon wire on stage 4); light strips pulse on POWERUP_TRIGGERED; spawn flash synced to special-piece spawns

---

### L5 · Encore Finale — Tier 3: Lua ORCHESTRATION + full presentation `[DONE 2026-08-23]`

Scripted 4-phase encounter: normal → garbage rain every 8s → blackout (dimmed
board, ghost hidden) → victory crescendo. Encounter design becomes authoring.

Core:
- [x] New pod `domains/environment/`: contract (mood state), reducer (color interpolator step), plan (diorama batch into PipelineExecutionPlan)
- [x] `scripts/environment/encounter_overseer.lua` — `(phase_state, events, dt) -> {new_phase, spawn_garbage?, mood_target}` (boss-phase pattern, ARCHITECTURE.md Part II Use Case 3)
- [x] Main forwards selected Matrix/Progression events to an `on_event(type, values)` Lua hook (Part III §4.3)
- [x] Mood interpolation cyan → crimson → gold wired into planner palette
- [x] Blackout dimming + ghost-hidden flag passed to planner as plain values
- [x] Garbage-rain scheduler emits spawn intents on phase cadence

GUI (cinematic show):
- [x] Phase-title banners + cinematic letterbox bars during transitions
- [x] Phase intensity meter (boss-style, driven by encounter state)
- [x] Garbage-rain warning arrows on the board sides before each volley
- [x] Victory star rating (performance-based: time, max combo, damage taken) + congrats scroll

FX (set pieces):
- [x] Phase-transition white-out wipe between phases
- [x] Garbage rain with impact tremors + dust plumes on landing
- [x] Blackout set piece: global dim + spotlight cone isolating only the active piece
- [x] Finale: confetti + firework bursts + gold particle rain + slow camera orbit around the board

Environment:
- [x] Full reactive lighting show (cyan → crimson → gold); animated neon pedestal; crowd-silhouette diorama with light-wave pulses synced to clears

---

### Meta-layer (currently missing from the demo)

**M1 · Session pod (menus without OOP UI frameworks) [DONE 2026-08-23]**

- [x] `domains/session/`: contract (`SessionSnapshot{screen: TITLE|LEVEL_SELECT|PLAYING|PAUSED|RESULTS, cursor_index, unlocked_stages, current_stage}`), action (NavUp/NavDown/Confirm/Back intents), reducer (pure screen state machine)
- [x] Menu rendering = ui-edge projection of SessionSnapshot (same suffix discipline, headless-testable)
- [x] Menu input reuses the existing intent-token pipeline through the input edge

Menu GUI:
- [x] Animated title screen: falling-tetromino attract background (planner-driven, zero new systems)
- [ ] Level-select carousel cards previewing each level's palette/mood
 — name/tier-tag/progress dots shipped; palette preview pending L3+
- [x] Pause overlay: dim + resume/restart/quit rows (projection of SessionSnapshot)
- [x] Wiring: session step precedes gameplay pods; STAGE_SELECTED/RUN_RESTART drive load_stage() FULL resets; SOUND_TOGGLED gates the synth; NAV_MOVED/CONFIRMED play menu blips; run-end latch feeds RESULTS its contextual first row (next-stage vs retry); high score survives restarts

**M2 · Campaign manifest (level-by-level progression)**

- [ ] `config/campaign/main_campaign.hpp` (or `.lua`) — ordered stages `{level_id, config/script refs, unlock_requirement}`
- [x] Session consumes each stage's `VICTORY` event → advance `current_stage`, unlock next (2026-08-22: victory dwells 3s on the results modal, then auto-advances with a FULL state reset — board/score/HUD/FX/fresh script sandbox/window title; R skips the dwell. Linear advance satisfies `unlock_after`.)

Progression GUI:
- [ ] Campaign track: node map with locked/unlocked/completed states
- [ ] Unlock animation when a stage completes (banner + node light-up)

**M3 · Congrats flow**

- [x] RESULTS screen projection (score/high/lines from ScoreState)
- [ ] Celebration overlay + fireworks hook fired on `VICTORY`
- [ ] Optional `scripts/results/congrats.lua` — message + particle recipe picker per stage (localization-friendly)

Results GUI:
- [ ] Score count-up animation + "NEW RECORD" sparkle on high-score beat
- [ ] Per-stage congrats flavor text + stats breakdown (max combo, specials used, time)

### Build order (each phase ends green: builds + gates PASS)

- [x] **A.** Wire lua.edge + deliver L2 Blitz 120 (smallest delta; proves determinism-with-scripting gates) — DONE 2026-08-22, all gates PASS
- [x] **B.** L3 Garbage Canyon generator scripts + seed-determinism gate — DONE 2026-08-23; all features live, SMOKE_TARGET_LINES + SEED_DIFF PASS; same-seed screenshot gate open (STATUS.md §0e pitfall)
- [x] **C.** Powerups pod + L4 Cyber Storm — DONE 2026-08-23; SMOKE_SPECIAL_EVERY_N + CYBER_DETERMINISM PASS, purity gates NONE (STATUS.md §0f)
- [x] **D.** Environment pod + L5 Encore Finale — DONE 2026-08-23; SMOKE_ENCOUNTER_CONFIG + ENCORE_DETERMINISM PASS, script purity NONE (STATUS.md §0g)
- [ ] **E.** Session pod + campaign manifest + congrats overlay (menus / progression / results)
 — session pod + menus + RESULTS screen DONE 2026-08-23 (all gates PASS); remaining: fireworks/congrats flavor + palette-preview cards

---

# Part 5: Why This Template Scales

1. **Plug-and-play domains:** keep the same `edges/` and shared pods
   (`spatial_fx`, `environment`); swap `domains/matrix/` for `domains/combat/`
   or `domains/vehicle_physics/` to get a different game on the same skeleton.
2. **Zero architectural rot:** no domain ever touches another directly —
   everything flows through immutable discrete event logs; features add
   listeners, not callbacks.
3. **Netcode & rollback ready:** snapshots are plain structs, reducers pure —
   rollback is keeping the last N snapshots; replays are input tokens + RNG seed.
4. **Multi-threaded Lua scalability:** stateless scripts evaluate on isolated
   thread-local `lua_State*` pools with no mutexes.
5. **Instant hot-reload:** edit a rule script mid-run; new rules apply next
   tick with zero memory-corruption risk.
---

# Part 6: Input Feel Hardening (priority — "dead feeling input")

Root cause analysis (2026-08-24 audit): the input edge emits ONE intent per
SDL_KEYDOWN and relies on OS key-repeat for held keys. OS repeat is late
(~500 ms delay), rate-limited by the OS (not the game), and never fires
KEYUP-aware state. This is exactly the bug class fixed in the JS twin
(hello-ember-tetris STATUS Sessions 5/5b): permanent soft-drop speedup and
"dead" held directions.

## 6.1 Held-state FSM in the input edge (port of the JS fix)

- [x] I1 Track physical key state per action: `held{left,right,down}` set on
      KEYDOWN, cleared on KEYUP (poll SDL_KEYUP — currently ignored entirely)
- [x] I2 Emit MOVE intents from the DAS/ARR scheduler at FIXED GAME RATES:
      DAS 150 ms initial delay, then ARR 40 ms auto-repeat (modern guideline
      values; constants exported for tuning). Never trust OS key-repeat.
- [x] I3 Soft drop becomes a HELD STATE (`soft_drop_held`), not a stream of
      intents; reducer reads the flag each tick (matrix already reads
      `input.soft_drop` as a bool — align edge to fill it from held state).
- [x] I4 Blur/focus-loss releases ALL held states (alt-tab safety).
- [x] I5 Menu/session keys keep existing repeat-guarded behavior (unchanged).

## 6.2 Reducer-side alignment

- [x] I6 `reduce_tetris_commands` keeps folding discrete intents; add a
      `soft_drop_held` passthrough so the gravity branch uses the held flag.
- [x] I7 Lock-delay reset cap: audit the 15-reset cap against modern
      guidelines (move-reset vs lock-timer reset semantics; cap 15 is fine,
      but confirm resets only on SUCCESSFUL moves/rotations — already true).

## 6.3 Verification (mirror the JS twin's headless proofs)

- [x] V1 Headless harness: synthetic key timeline (step_tests input harness) -> assert piece reaches
      column N within X ms of hold start (DAS timing test)
- [x] V2 Held soft drop: descent rate fast while held, normal after release
- [x] V3 Alt-tab mid-hold: no stuck movement after refocus (release_all on FOCUS_LOST)
- [x] V4 Determinism gates still PASS (two idle runs byte-equal)

---

# Part 7: Structural convergence with the Domain-POD v2 concepts
(Reference: docs/pods/ — PLANNING.md workflow, EVENT_FLOW map, SCRIPTING DSL,
MISSIONS.md goal patterns. Discovered 2026-08-24 during the JS twin's L4.)

The Lua seam already proves tier-2 scripting (rules hooks). These items
converge the C++ demo onto the same graph-first / trap-wall vocabulary:

## 7.1 Event-flow documentation parity

- [x] E1 Write a generator (node) - scripts/generate-event-flow.mjs, 11 facts mapped
      `MatrixEventType::X` emissions + Lua `emit()` calls into
      docs/pods/EVENT_FLOW.md — same shape as the JS twin's script
- [ ] E2 Add TRAP TABLE header comments to every *.reducer.hpp (springs-on
      list at top of file)

## 7.2 Scripted goals via predicate composition (Lua port of the DSL)

- [ ] G1 Define the Lua predicate contract: `goal.test(events, snapshot)`
      returning bool; events arrive as plain tables; snapshot carries
      read-only views (score, lines, freeze timers, player pos if FPS later)
- [x] G2 DONE (2026-08-25): domains/mission contract+reducer - two goal styles, fact-chained sequencing, timed expiry; 10/10 pins; purity NONE. Original: (`domains/mission/`): cumulative progress +
      fact-chained sequencing (MISSION_COMPLETE advances index) — mirrors
      MISSIONS.md §5
- [ ] G3 Level authoring: missions declared in level.lua as composed
      predicates using a small stdlib (`when/anyOf/count_where/during`)
      implemented ONCE in Lua (the DSL is ~60 lines — see SCRIPTING.md §1)
- [ ] G4 Sandbox purity rules for goal scripts: no io/os, seeded rng only,
      deterministic iteration order (pairs ordering must be sorted)

## 7.3 Save/state serialization groundwork

- [ ] S1 Pod snapshots are plain structs — define a reflection-lite visitor
      per contract to emit JSON (STATE_SAVE.md section 2); version field
      mandatory
- [ ] S2 Checkpoint policy: save on stage complete/failed only (v1)

## 7.4 Deferred until needed

- Bloom/post chain, entity-array world pods, AI perception pods: designed in
  docs/pods/ (FPS_EXAMPLE, AI_PODS) but out of scope for the Tetris demo;
  they document the template for future games on this skeleton.

---

# Part 8: Project Structure — Final Version Plan (ACTIVE TRACK)

Level CONTENT frozen temporarily; structure hardening first. Full plan:
[STRUCTURE_PLAN.md](STRUCTURE_PLAN.md). Phases P0-P6, each ends green.

- [x] P0 Baseline safety net (20 reducer pins + ctest UNIT gate) — DONE (doctest unit tests pin current behavior; ctest gate in verify.sh)
- [x] P1 Extract engine loop + game step (P1a core + P1b IScriptHost/IAudioSink); main 941->760 lines — DONE (engine/loop.hpp) + game step (game/step.hpp); main.cpp -> ~150 lines
- [x] P2 Unit test expansion (19 step/input tests; script purity ctest) — DONE (whole-frame step tests, input-edge DAS/ARR harness, script-purity ctest)
- [x] P3 DONE (2026-08-25): assets/campaign/campaign.lua + game/stage.hpp loader (StageDef + load_campaign with fallback law); main rewired; tests/stage_loader_tests.cpp 15/15; config/levels/*.hpp retained as reference until next cleanup pass
- [x] P4 DONE (2026-08-25): domains/shared/event_ids.hpp FactId enum + FACT_REGISTRY (23 facts, producer per fact); generate-event-flow.mjs rewritten registry-driven with undeclared-producer gate; EVENT_FLOW regenerated
- [x] P5 DONE (2026-08-25): cpp-folders/libs/engine shs::engine INTERFACE target (header-only loop.hpp); tetris links it; local engine/ dir removed; aggregator wires the lib for all demos
- [x] P6 DONE (2026-08-25): ARCHITECTURE Part III rewritten (layer map, layered tree, post-P1b dataflow); EVENT_FLOW already registry-regenerated; LUA_STRUCTURE.md added (Unity mapping + four-file level standard)

Debts addressed: D1 god-main, D2 no unit tests, D3 levels-need-recompile,
D4 implicit event vocabulary, D5 loop buried in main, D6 no shared boundary.
