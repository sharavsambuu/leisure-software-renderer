# Tetris Architecture — Domain Pods: Theory & As-Built

Single architecture reference for the Hello3DTetris demo: Part I–II hold the
VOP/DOD **theory** (formerly NOTES.md), Part III holds the **as-built** tetris
reference (formerly DETAILS.md). Merged 2026-08-22 during the docs-integration
pass (per-demo docs convention, mirroring snake).

## Doc map (3 files, single source of truth per concern)

| Doc | Owns |
| :--- | :--- |
| `ARCHITECTURE.md` | This file — pod theory, Lua philosophy, as-built tree/dataflow/ownership, Lua design rules, conventions |
| `TODOS.md` | Living tracker: pod/edge status + level/mode campaign proposals (Part 4) |
| `STATUS.md` | Verification log: build results, headless gates, DoD, pitfalls, migration history |

---

# Part I — Domain Pod Theory

In **Value-Oriented Programming (VOP)** and **Data-Oriented Design (DOD)**, a Domain Pod is not an "object" or an "asset"—it is a **Self-Contained Mathematical Rulebook** governing a specific facet of game state.

Here is a practical guide on how to define, name, split, and boundary-check domain pods for any game genre.

### 1. The Mental Model: What is a Domain Pod?

A Domain Pod owns:
1. **Plain Data Schema (`*.contract.hpp`)**: The flat state tables (SoA arrays, tables).
2. **Pure State Reducer (`*.reducer.hpp`)**: The pure mathematical transition function:
   $$\text{State}_{t+1}, \text{Events} = f(\text{State}_t, \text{Commands}, \Delta t)$$
3. **Pure Visual Planner (`*.plan.hpp`)**: The batch generator that converts snapshots to draw tokens.

```
       ┌────────────────────────────────────────────────────────┐
       │                      DOMAIN POD                        │
       │                                                        │
       │  [*.contract.hpp] ◄── Plain Data Schema                │
       │  [*.action.hpp]   ◄── Intent Tokens                    │
       │  [*.event.hpp]    ◄── Discrete Occurrences             │
       │  [*.reducer.hpp]  ◄── Pure Transition: (State)->(State)│
       │  [*.plan.hpp]     ◄── Batch Mesh Planner               │
       └────────────────────────────────────────────────────────┘
```

### 2. The 4 Litmus Tests: When to Create a New Domain

Create a new domain if and only if the system passes at least two of these tests:

#### Test 1: The Independent Lifecycle Test
*Does this state exist or update on its own timeline?*
- **Example**: In an RPG, your `inventory` exists whether you are fighting in a dungeon, talking in a shop, or dead. Therefore, `inventory` is its own domain, completely separate from `combat`.
- **Example**: In a racing game, `track_timing` (laps, sector splits, checkpoints) exists whether a car is driving cleanly or crashing into a wall.

#### Test 2: The Reusable Vocabulary Test
*Does this system operate on its own distinct data schemas that have no reason to be polluted by other mechanics?*
- `combat` operates on `ProjectileTableSoA`, `DamagePackets`, `Hitboxes`.
- `quest` operates on `ObjectiveTable`, `QuestGraph`, `MilestoneCounters`.
- If mixing them would make your state struct messy or cause unused memory padding, split them into separate domain pods.

#### Test 3: The Event-Decoupling Test (Constitution II, Rule 8.1)
*Can this system do 100% of its work by listening to discrete events from other systems rather than calling functions synchronously?*
- `Quest` does **not** need to call `enemy->take_damage()`. It simply listens to `EventBotKilled` emitted by `Combat`.
- `SpatialFX` does **not** need to touch the Tetris grid. It simply listens to `EventLinesCleared` and spawns voxel debris.

#### Test 4: The Designer / Lua Rulebook Boundary Test
*Could a game designer tweak, rewrite, or hot-reload the rules of this feature (via Lua or C++) without risking a crash in other systems?*
- Example: Tuning the scoring multiplier curve in `progression` should never introduce bugs into the SRS piece-rotation math in `matrix`.

### 3. The 4 Red Flags: When NOT to Create a Domain

#### ❌ Red Flag 1: The OOP Entity Trap (Creating pods per object type)
- **Wrong**: `domains/player`, `domains/enemy_goblin`, `domains/boss_dragon`.
- **Right**: `domains/combat`, `domains/locomotion`, `domains/ai_behavior`.
*Why*: In DOD, entities are just integer IDs (`uint32_t`). A goblin and a player both have a position in `locomotion` and health in `combat`. Logic belongs to functional domains, not object categories.

#### ❌ Red Flag 2: The Micro-Function Trap (Over-fragmentation)
- **Wrong**: `domains/health_calculator`, `domains/camera_shake_math`, `domains/drop_timer`.
- **Right**: Put pure math in `shs::Math` or keep it inside the owning domain (`combat`, `spatial_fx`, `matrix`).

#### ❌ Red Flag 3: The Driver / Hardware Trap
- **Wrong**: `domains/audio_synth`, `domains/vulkan_renderer`, `domains/sdl_window`.
- **Right**: Hardware drivers, audio DACs, and GPU submitters belong in **`edges/`** (one subdirectory per edge: `edges/audio/tetris.audio.hpp`, `edges/rasterizer/tetris.rasterizer.hpp`, `edges/input/tetris.input.hpp`), because they are impure execution boundaries, not game simulation domains.

#### ❌ Red Flag 4: Synchronous Tight-Loop Coupling
- If Subsystem A and Subsystem B must mutate the exact same arrays at the exact same microsecond with zero event latency, **they are one single domain**.

### 4. Domain Naming Laws

1. **Use Capability Nouns or Gerunds (Never Entity Names)**:
   - ✅ `combat`, `navigation`, `inventory`, `progression`, `spatial_fx`, `crafting`, `locomotion`, `vehicle_physics`.
   - ❌ `player`, `enemies`, `swords`, `cars`, `levels`, `ui_manager`.
2. **Keep it Singular and Concise**:
   - ✅ `dialogue`, `economy`, `environment`, `weather`.
   - ❌ `dialogue_systems`, `economic_manager`, `game_weathers`.
3. **Reflect What It Solves**:
   - If the pod computes where entities move $\to$ `locomotion` or `navigation`.
   - If the pod computes who hurts whom $\to$ `combat`.
   - If the pod computes rules, scores, and win/loss states $\to$ `progression`.

### 5. Domain Blueprints Across Different Game Genres

| Game Genre | Recommended Domain Pods (5–8 per game) |
| :--- | :--- |
| **FPS / Action Shooter** | `combat` (hitscan, projectiles, damage)<br>`locomotion` (walk, jump, slide, crouch)<br>`ai_behavior` (patrol, chase, aim evaluators)<br>`weapon_inventory` (ammo, clips, reloading)<br>`spatial_fx` (tracers, blood, muzzle flash, camera recoil)<br>`progression` (score, kill counters, round timer) |
| **Racing / Vehicle Game** | `vehicle_physics` (drivetrain, tire friction, suspension)<br>`track_navigation` (checkpoints, waypoints, track spline)<br>`track_timing` (laps, sector times, speed traps)<br>`ai_driver` (steering, braking points, overtaking)<br>`vehicle_customization` (tuning, paint, parts)<br>`spatial_fx` (tire smoke, skid marks, spark bursts, camera shake) |
| **Action RPG / Hack & Slash** | `combat` (attacks, hitboxes, parry, poise)<br>`status_effects` (buffs, debuffs, burn, freeze timers)<br>`inventory_equipment` (items, gear stats, loot bags)<br>`skill_tree` (unlocked abilities, mana costs)<br>`quest_progression` (story milestones, dialogue flags)<br>`spatial_fx` (slashes, impact sparks, floating damage numbers) |
| **RTS / Strategy** | `formation_locomotion` (flocking, pathfinding, movement spans)<br>`economy` (minerals, gas, harvester queues)<br>`construction` (building grids, construction timers)<br>`combat` (range checks, armor types, direct damage)<br>`fog_of_war` (visibility grids, vision radiuses)<br>`ai_commander` (build orders, attack waves) |
| **Puzzle / Arcade (Tetris / Match-3)** | `matrix` / `playfield` (grid, collision, piece gravity, line clears)<br>`progression` (modes, scoring, blitz timers, level speed)<br>`spatial_fx` (3D voxel shatter debris, shockwave spring)<br>`powerups` (bomb pieces, row lasers, time freezes)<br>`environment` (3D diorama stage, reactive mood lighting) |

### 6. The Golden Rules of Thumb

1. **The 5-to-8 Sweet Spot**: Almost every complete commercial game can be cleanly partitioned into **5 to 8 domain pods**. If you have 25 pods, you are over-fragmenting; if you have 1 monolith, you are under-modularizing.
2. **The "Airplane Test"**: If you delete a domain pod folder (e.g. delete `domains/quest/`), the rest of the game (e.g. `combat`, `matrix`, `locomotion`) should still compile and run cleanly, simply ignoring the missing events.
3. **The Pure Center Contract**: Reducers must have **zero `#include <vulkan/...>` or `#include <SDL2/...>`** and zero standard heap allocations (`malloc`/`new`). Keep math pure in the center, and leave all side effects at the execution edges.

---

# Part II — Lua Scripting in VOP/DOD

Because Domain Pods use **Value-Oriented Programming (VOP)**, you **do not need bloated intermediate C++ binding libraries** like *Sol2, LuaBridge, Luabind, or SWIG*.

#### Why traditional C++ games need heavy binders:
In traditional Object-Oriented engines, C++ binds complex classes with inheritance hierarchies, virtual methods, raw pointers, and `std::shared_ptr` lifecycles into Lua (e.g., `monster:TakeDamage(50)`, `player:GetInventory():GetItem(3)`). This requires massive template metaprogramming, slows down compilation times, and frequently causes memory leaks and GC crashes.

#### Why VOP only needs the tiny, official C Lua library (`lua.h`):
In VOP, there are **no classes or pointers to bind**. You only pass plain data (numbers, booleans, string tags, flat tables) into a pure Lua function and read back an explicit result struct.

The standard official Lua C API is:
- **Tiny**: ~200 KB compiled, $<30$ source files, compiles in less than 1 second.
- **Fast**: Zero template overhead, zero hidden atomic reference counting.
- **Simple**: Calling a pure Lua function only takes $\approx 15$ lines of standard `lua_push...` and `lua_getfield` calls.

### Is Lua just for configurations, or is it real gameplay programming?

If Lua were only returning static tables (like `{ speed = 5.0, hp = 100 }`), you wouldn't need Lua—plain JSON, YAML, or C++ structs would be enough.

In a VOP Domain Pod architecture, Lua is used for **Turing-complete, algorithmic gameplay programming** written as **Pure Decision Functions**:

$$\text{Next Actions / State Patch} = \text{LuaScript}(\text{Current State Snapshot}, \text{Delta Time}, \text{Context})$$

### 4 Concrete Examples of Real Gameplay Programming in Lua

#### Use Case 1: AI Behavior & Decision Brains (`domains/ai_bots/scripts/stalker_bot.lua`)

```lua
-- domains/ai_bots/scripts/stalker_bot.lua
local StalkerAI = {}

-- Pure decision algorithm: (BotState, PlayerState, EnvironmentContext) -> ActionIntent
function StalkerAI.decide_action(bot, player, context)
    local distance = math.sqrt((player.x - bot.x)^2 + (player.z - bot.z)^2)
    
    -- Tactical Logic: Low health retreats to health packs
    if bot.hp < 30 and context.nearest_medkit_dist < 15.0 then
        return {
            intent = "RETREAT",
            target_x = context.nearest_medkit_x,
            target_z = context.nearest_medkit_z,
            should_shoot = false
        }
    end

    -- Flanking Logic: If player is aiming at us, strafe aggressively
    if player.is_aiming_at_me and distance < 12.0 then
        return {
            intent = "STRAFE_DODGE",
            direction = (bot.id % 2 == 0) and 1 or -1,
            should_shoot = true
        }
    end

    -- Default Aggression
    return {
        intent = "CHASE_AND_FIRE",
        aim_lead_time = distance / 25.0, -- Projectile flight time prediction
        should_shoot = (distance < 18.0)
    }
end

return StalkerAI
```

#### Use Case 2: Dynamic Roguelike / Cyber Modifiers (`domains/powerups/scripts/glitch_laser.lua`)

```lua
-- domains/powerups/scripts/glitch_laser.lua
local GlitchLaser = {}

-- Evaluates what happens when this special block is cleared
function GlitchLaser.on_line_cleared(grid_snapshot, cleared_row, combo_streak)
    local mutated_cells = {}
    local bonus_score = 0

    -- Custom Rule: Laser vaporizes the entire vertical column intersecting the row
    for y = 0, 19 do
        table.insert(mutated_cells, { x = 4, y = y, new_val = 0 }) -- Clear center column
    end

    -- Synergistic Rule: If cleared during a high combo, convert bottom row to gold blocks
    if combo_streak >= 3 then
        for x = 0, 9 do
            table.insert(mutated_cells, { x = x, y = 0, new_val = 2 }) -- Yellow/Gold block ID
        end
        bonus_score = 1500
    end

    return {
        cleared_overrides = mutated_cells,
        score_multiplier = (combo_streak > 2) and 2.5 or 1.0,
        bonus_score = bonus_score,
        trigger_screen_flash = "CYAN"
    }
end

return GlitchLaser
```

#### Use Case 3: Boss Fight Phases & Scripted Encounters (`domains/combat/scripts/boss_overseer.lua`)

```lua
-- domains/combat/scripts/boss_overseer.lua
local BossOverseer = {}

function BossOverseer.evaluate_phase(boss_hp, max_hp, elapsed_time, current_phase)
    local hp_pct = boss_hp / max_hp

    -- Transition to Phase 2 (Rage Mode)
    if current_phase == 1 and hp_pct < 0.60 then
        return {
            new_phase = 2,
            spawn_adds_count = 3,
            arena_laser_active = true,
            music_track = "BOSS_PHASE_2"
        }
    end

    -- Transition to Phase 3 (Bullet Hell Final Stand)
    if current_phase == 2 and (hp_pct < 0.20 or elapsed_time > 180.0) then
        return {
            new_phase = 3,
            shield_invulnerable = true,
            orbital_strike_interval = 2.5,
            music_track = "BOSS_FINAL_STAND"
        }
    end

    -- Maintain current state
    return { new_phase = current_phase }
end

return BossOverseer
```

#### Use Case 4: Procedural Level & Puzzle Generators (`domains/matrix/scripts/puzzle_generator.lua`)

```lua
-- domains/matrix/scripts/puzzle_generator.lua
local PuzzleGen = {}

function PuzzleGen.generate_challenge(difficulty, seed)
    math.randomseed(seed)
    local grid = {} -- 20x10 array initialized to 0

    -- Build staggered "staircase" garbage blocks with holes
    local max_height = math.min(12, 4 + difficulty * 2)
    for y = 0, max_height do
        local hole_x = math.random(0, 9)
        for x = 0, 9 do
            if x ~= hole_x then
                table.insert(grid, { x = x, y = y, cell_type = math.random(1, 7) })
            end
        end
    end

    return {
        initial_blocks = grid,
        time_limit_seconds = 60 - (difficulty * 5),
        target_lines_to_win = 10 + difficulty * 2
    }
end

return PuzzleGen
```

### Summary: The Role of Lua in VOP

| Configuration (JSON/YAML) | VOP Stateless Gameplay Lua |
| :--- | :--- |
| Static values (`hp = 100`, `speed = 5.0`) | **Algorithms, branching logic, formulas, decision trees** |
| Requires C++ code to interpret what to do | **Executes custom game rules and returns actions** |
| Cannot contain `if/else`, loops, or math | **Turing-complete gameplay programming** |
| Same behavior every frame | **Dynamic reactions to real-time snapshot data** |

With this setup:
1. **Engine programmers** build high-performance, crash-proof, multi-threaded C++ DOD/VOP systems.
2. **Gameplay programmers & designers** write complex, hot-reloadable game rules and AI behaviors in simple Lua scripts with **zero risk of corrupting C++ memory or crashing the engine**.

---

# Part III — As-Built Reference (tetris)

## 1. As-built tree

```text
tetris/
├── hello_3d_tetris.cpp        # main edge (~600 lines): SDL lifecycle, per-frame PMR arena, session wiring,
│                              # event→sound map, headless hooks
├── verify.sh                  # reproducible headless verification battery
├── CMakeLists.txt             # single -I root; target-scoped include order
├── config/
│   ├── rules.hpp              # tetris::config::Rules — every gameplay number
│   ├── levels/marathon_01.hpp # tetris::config level definitions (marathon_01, blitz_120)
│   └── campaign/main_campaign.hpp # ordered stage manifest {rules factory, script_path, display_name}
├── domains/
│   ├── matrix/                # contract / action / event / reducer (pure grid rulebook)
│   ├── progression/           # contract / event / reducer (event-fed scoring)
│   ├── spatial_fx/            # contract / reducer / plan (vocabulary + fx + planner)
│   └── session/                # contract / action / reducer (meta screen state machine)
├── edges/
│   ├── input/tetris.input.hpp     # SDL polling → intent tokens
│   ├── audio/tetris.audio.hpp     # verbatim synth port (12 voices, SPSC ring)
│   ├── rasterizer/tetris.rasterizer.hpp # screen-space helpers (tetris::raster::vop)
│   ├── ui/tetris.hud.hpp          # Mongolian UTF-8 font engine + HUD/menu screen projections
│   └── lua/lua.edge.hpp           # StatelessLuaEvaluator — wired via ScriptHooks bridges (see §4)
└── docs/                      # ARCHITECTURE.md · TODOS.md · STATUS.md
```

## 2. Frame dataflow (as built)

```
input::poll_input(arena) ─► InputState{quit, span<TetrisCommand>}
        │
        ▼
world.drop_interval = rules.gravity_for_level(score.level)   ← main-edge wiring
        │
matrix::reduce_matrix(world, commands, dt, arena)
        │  → MatrixStepResult{MatrixSnapshot, pmr::vector<MatrixEvent>}   RAW FACTS
        ├──────────────────────────────────────────┐
        ▼                                          ▼
progression::reduce_progression(events, …)   spatial_fx::step_fx(fx, events, dt)
        │  → ScoreState + ProgressionEvents         │  (particles, shake, fx clock)
        │                                           ▼
        │                          spatial_fx::plan_tetris_scene(world, fx, W, H, arena)
        │                                          │  → PipelineExecutionPlan
        │                                          ▼
        │                          tiled raster jobs ×N (raster::vop helpers)
        ▼                                          ▼
ui::draw_hud(canvas, world, score_state)    present / screenshot
audio map (main): MatrixEvent → audio::SND_* via SPSC ring
```

Contact rules honored: pods talk only through **event spans**, **plain values
wired by main** (`drop_interval`, restart preservation), and the planner's
**read-only** access to the matrix contract.

## 3. Ownership map (what moved where)

| Concern | Was | Now |
|---|---|---|
| Grid/piece/hold/bag/RNG state | `TetrisSnapshot` god-struct | `matrix::MatrixSnapshot` |
| Score/high/lines/level/combo/target/victory | same struct + inline math in `reduce_tetris` | `progression::ScoreState`; scoring derived from `MatrixEvent`s in `reduce_progression` |
| Base score table `{0,100,300,500,800}`, combo bonus, gravity curve, lock constants | hardcoded in reducer/main | `config::Rules` (+ `gravity_for_level`) |
| Gravity cadence updates on level-up | inside matrix reducer | main wires `rules.gravity_for_level(score.level)` each frame; reducer only reads `drop_interval` |
| Soft/hard drop points | silent `s.score += n` | raw facts: `SOFT_DROP{cells}` / `HARD_DROP_SLAM{cells}` → progression converts |
| Combo break on sterile lock | matrix field | progression watches `PIECE_LOCK_IMPACT` w/o clear in same frame |
| High-score preservation across reset | matrix snapshot copy | main edge captures/restores around restart |
| Shatter particles + camera shake + sway clock | main locals + plan args | `spatial_fx::FxState` (stepped by `step_fx`, read by planner) |
| Debris randomness | `rand()` (non-deterministic) | seeded xorshift in `FxState` |
| Piece palette | grid contract | `spatial_fx::get_piece_color` (render vocabulary) |
| Audio synth/ring/callback, raster helpers, HUD+font engine, input polling | ~600 lines inside main | `edges/{audio,rasterizer,ui,input}` |

## 4. Lua integration design (NEXT TASK)

The architecture already reserves the seams. Intended shape, aligned with
the tetris demo's `TODOS.md` Part 2 and the fps lua lessons:

**4.1 Edge placement.** `edges/lua/lua.edge.hpp` stays the ONLY file that
includes Lua headers. Domains never see `lua_State*`. The evaluator remains
stateless-per-call: script text + plain-value inputs → plain-value outputs.
(The evaluator is already scaffolded in the build as
`tetris::lua_edge::StatelessLuaEvaluator`; it is not yet wired into the loop.)

**4.2 First swap: the scoring rule.**
`progression::compute_line_clear_score(rules, lines, level, combo)` is a pure
value-in/value-out function isolated for exactly this. Phase 1 of Lua work:

- Add `config/rules.lua` (or `scripts/scoring.lua`) returning
  `{ base = {...}, combo_bonus = n, hard_drop_per_cell = n, ... }`.
- Main edge loads it once at boot through `lua.edge` into a `config::Rules`
  instance — config stays a plain struct; Lua is just an authoring format.
- Optional Phase 1b: route per-clear evaluation through a Lua function
  `compute_score(lines, level, combo) -> int`, called from
  `reduce_progression`. Keep the C++ path as fallback when no script is set.
  Determinism note: same script + same inputs must give identical outputs;
  no os/io/random in rule scripts (enforce by opening a sandboxed luaL).

**4.3 Future pods arrive WITH their scripts.**

- `domains/powerups/` (bomb/laser/freeze): pod contract holds powerup state;
  trigger/effect rules live in `scripts/powerups/*.lua` evaluated via lua.edge
  against `(matrix facts, powerup state) -> commands/events`. The pod's reducer
  applies returned intents; matrix stays untouched.
- `domains/environment/`: mood/difficulty curves authored in
  `scripts/environment/*.lua` (e.g., backdrop hue as f(score, danger)).
- Event bus extension: main may forward selected `ProgressionEvent`s to a
  `on_event(type, values)` Lua hook for achievements/objectives without any
  domain knowing about Lua.

**4.4 Build wiring (when activated).** vcpkg `"lua"` entry +
`find_package(Lua REQUIRED)` guarded so the demo still builds without it
(follow the SDL2_image optional pattern in the aggregator). Sources list gains
`scripts/*.lua` for IDE visibility only.

**4.5 Verification additions.** Extend `verify.sh` with:
`--script <file>` flag on main → loads a Lua rules file; determinism double-run
must stay byte-identical WITH scripting active; a `blitz_mode.lua` smoke test
(target score override) asserting `ScoreState.target_score` changes.

## 5. Conventions enforced here (fps-standard)

- Includes: root-relative angle brackets across pods (`<domains/matrix/…>`,
  `<config/rules.hpp>`); quotes ONLY for same-directory siblings
  (`"shs_renderer.hpp"`). No `../../` chains anywhere.
- Namespaces: one nested level per pod — `tetris::{matrix,progression,
  spatial_fx,input,audio,raster,ui,config}`; cross-pod visibility via explicit
  `using tetris::matrix::X;` declarations at the consumer (never global
  using-directives in headers outside `main`).
- Purity gates (CI-able greps): no `SDL` under `domains/`; no `score|combo`
  under `domains/matrix/`.
- Include-path hygiene: aggregator leaks sibling roots globally, so
  Hello3DTetris carries a target-scoped `INCLUDE_DIRECTORIES` with its own root
  FIRST (see STATUS.md pitfall 4). Other demos should adopt this later.