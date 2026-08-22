# Domain Pods & Stateless Lua — Design Notes

Reference guide for the VOP/DOD domain-pod architecture used by the hello-3d demos
(tetris is the reference implementation). Companion documents in this folder:
`TODOS.md` (canonical 5-pod blueprint, status, Lua integration plan) and
`REFACTOR_PROPOSAL.md` (how tetris was migrated onto this structure).

---

In **Value-Oriented Programming (VOP)** and **Data-Oriented Design (DOD)**, a Domain Pod is not an "object" or an "asset"—it is a **Self-Contained Mathematical Rulebook** governing a specific facet of game state.

Here is a practical guide on how to define, name, split, and boundary-check domain pods for any game genre.

---

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

---

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

---

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

---

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

---

### 5. Domain Blueprints Across Different Game Genres

Here is how you can slice domains across major genres using the exact same VOP/DOD architecture:

| Game Genre | Recommended Domain Pods (5–8 per game) |
| :--- | :--- |
| **FPS / Action Shooter** | `combat` (hitscan, projectiles, damage)<br>`locomotion` (walk, jump, slide, crouch)<br>`ai_behavior` (patrol, chase, aim evaluators)<br>`weapon_inventory` (ammo, clips, reloading)<br>`spatial_fx` (tracers, blood, muzzle flash, camera recoil)<br>`progression` (score, kill counters, round timer) |
| **Racing / Vehicle Game** | `vehicle_physics` (drivetrain, tire friction, suspension)<br>`track_navigation` (checkpoints, waypoints, track spline)<br>`track_timing` (laps, sector times, speed traps)<br>`ai_driver` (steering, braking points, overtaking)<br>`vehicle_customization` (tuning, paint, parts)<br>`spatial_fx` (tire smoke, skid marks, spark bursts, camera shake) |
| **Action RPG / Hack & Slash** | `combat` (attacks, hitboxes, parry, poise)<br>`status_effects` (buffs, debuffs, burn, freeze timers)<br>`inventory_equipment` (items, gear stats, loot bags)<br>`skill_tree` (unlocked abilities, mana costs)<br>`quest_progression` (story milestones, dialogue flags)<br>`spatial_fx` (slashes, impact sparks, floating damage numbers) |
| **RTS / Strategy** | `formation_locomotion` (flocking, pathfinding, movement spans)<br>`economy` (minerals, gas, harvester queues)<br>`construction` (building grids, construction timers)<br>`combat` (range checks, armor types, direct damage)<br>`fog_of_war` (visibility grids, vision radiuses)<br>`ai_commander` (build orders, attack waves) |
| **Puzzle / Arcade (Tetris / Match-3)** | `matrix` / `playfield` (grid, collision, piece gravity, line clears)<br>`progression` (modes, scoring, blitz timers, level speed)<br>`spatial_fx` (3D voxel shatter debris, shockwave spring)<br>`powerups` (bomb pieces, row lasers, time freezes)<br>`environment` (3D diorama stage, reactive mood lighting) |

---

### 6. The Golden Rules of Thumb

1. **The 5-to-8 Sweet Spot**: Almost every complete commercial game can be cleanly partitioned into **5 to 8 domain pods**. If you have 25 pods, you are over-fragmenting; if you have 1 monolith, you are under-modularizing.
2. **The "Airplane Test"**: If you delete a domain pod folder (e.g. delete `domains/quest/`), the rest of the game (e.g. `combat`, `matrix`, `locomotion`) should still compile and run cleanly, simply ignoring the missing events.
3. **The Pure Center Contract**: Reducers must have **zero `#include <vulkan/...>` or `#include <SDL2/...>`** and zero standard heap allocations (`malloc`/`new`). Keep math pure in the center, and leave all side effects at the execution edges.



# Lua Scripting in VOP/DOD

Because Domain Pods use **Value-Oriented Programming (VOP)**, you **do not need bloated intermediate C++ binding libraries** like *Sol2, LuaBridge, Luabind, or SWIG*.

#### Why traditional C++ games need heavy binders:
In traditional Object-Oriented engines, C++ binds complex classes with inheritance hierarchies, virtual methods, raw pointers, and `std::shared_ptr` lifecycles into Lua (e.g., `monster:TakeDamage(50)`, `player:GetInventory():GetItem(3)`). This requires massive template metaprogramming, slows down compilation times, and frequently causes memory leaks and GC crashes.

#### Why VOP only needs the tiny, official C Lua library (`lua.h`):
In VOP, there are **no classes or pointers to bind**. You only pass plain data (numbers, booleans, string tags, flat tables) into a pure Lua function and read back an explicit result struct.

The standard official Lua C API is:
- **Tiny**: ~200 KB compiled, $<30$ source files, compiles in less than 1 second.
- **Fast**: Zero template overhead, zero hidden atomic reference counting.
- **Simple**: Calling a pure Lua function only takes $\approx 15$ lines of standard `lua_push...` and `lua_getfield` calls.

---

### Is Lua just for configurations, or is it real gameplay programming?

If Lua were only returning static tables (like `{ speed = 5.0, hp = 100 }`), you wouldn't need Lua—plain JSON, YAML, or C++ structs would be enough.

In a VOP Domain Pod architecture, Lua is used for **Turing-complete, algorithmic gameplay programming** written as **Pure Decision Functions**:

$$\text{Next Actions / State Patch} = \text{LuaScript}(\text{Current State Snapshot}, \text{Delta Time}, \text{Context})$$

---

### 4 Concrete Examples of Real Gameplay Programming in Lua

Here is what **actual gameplay logic** looks like inside VOP domain pods:

---

#### Use Case 1: AI Behavior & Decision Brains (`domains/ai_bots/scripts/stalker_bot.lua`)
Instead of hardcoding enemy AI in C++, the AI brain is a pure function that evaluates the tactical situation and returns an intent action:

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

---

#### Use Case 2: Dynamic Roguelike / Cyber Modifiers (`domains/powerups/scripts/glitch_laser.lua`)
Complex branching rules for abilities and power-up synergies:

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

---

#### Use Case 3: Boss Fight Phases & Scripted Encounters (`domains/combat/scripts/boss_overseer.lua`)
State-machine rules for multi-phase boss battles:

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

---

#### Use Case 4: Procedural Level & Puzzle Generators (`domains/matrix/scripts/puzzle_generator.lua`)
Algorithmic logic that creates procedural starting puzzles or obstacle scenarios:

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

---

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