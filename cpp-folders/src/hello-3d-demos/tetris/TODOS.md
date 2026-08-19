# Todos

Do goal is to build mutiple domain pods structure can be used as template for other game demos
And also to have Lua based scripting capabilities



To thoroughly demonstrate **Value-Oriented Programming (VOP)**, **Data-Oriented Design (DOD)**, and a **Stateless Lua Scripting Layer** without over-engineering, the sweet spot is **5 Domain Pods** combined with **3 Execution Edges**.


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

---

### Why these 5 Pods are optimal:

| Domain Pod | Core Responsibility | DOD / VOP Showcase Feature |
| :--- | :--- | :--- |
| **1. `matrix`** | Grid, collision, SRS rotations, pieces | Pure 2D spatial array math, deterministic gravity accumulator. |
| **2. `progression`** | Score, combos, level curves, game modes | Event-driven rules engine with **Stateless Lua mode evaluators**. |
| **3. `spatial_fx`** | 3D particles, shockwaves, camera spring | **Structure of Arrays (SoA)** particle physics, spring dampening. |
| **4. `powerups`** | Special cyber blocks (Bomb, Laser, Freeze) | Matrix mutation requests, cooldown timers, **Stateless Lua effects**. |
| **5. `environment`** | Diorama, pedestal, reactive mood lighting | Low-poly mesh batching, ambient color state interpolators. |

---

# Part 2: Where and How to Integrate Stateless Lua Scripting

### The Constitution II Rule for Lua:
> **"Lua scripts must never hold mutable pointers to C++ objects. Lua functions must be pure stateless reducers: taking an immutable C++ data snapshot, performing game logic, and returning an explicit value result."**

```
┌─────────────────────────┐          ┌───────────────────────┐          ┌──────────────────────────┐
│ Immutable C++ Snapshot  ├─────────►│  Stateless Lua Script ├─────────►│ C++ Value Mutation Patch │
│ (Lines, Combo, Time, dt)│          │   (Pure Rule Function)│          │ (ScoreDelta, Events, New)│
└─────────────────────────┘          └───────────────────────┘          └──────────────────────────┘
```

---

### 2.1 File Placement for Lua Scripts

Following the Glimmer/Ember Pod Standard, scripts live directly inside their respective domain pod folder under a `scripts/` directory:

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

---

### 2.2 Concrete Example: Pure Lua Scoring & Game Mode Rule

#### In `domains/progression/scripts/blitz_mode.lua`:
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

---

### 2.3 The C++ Lua Edge Wrapper (`edges/lua.edge.hpp`)

In C++, you execute the script by pushing values onto the Lua stack, invoking the function, and popping the resulting struct:

```cpp
#pragma once

#include <lua.hpp>
#include <string_view>
#include <span>

namespace edge {

    struct LuaScoreResult {
        int  score_added  = 0;
        bool level_up     = false;
        bool danger_alert = false;
    };

    class StatelessLuaEvaluator {
    public:
        explicit StatelessLuaEvaluator(const std::string& script_path) {
            L_ = luaL_newstate();
            luaL_openlibs(L_);
            if (luaL_dofile(L_, script_path.c_str()) != LUA_OK) {
                // Handle compilation error
                lua_pop(L_, 1);
            }
        }

        ~StatelessLuaEvaluator() {
            if (L_) lua_close(L_);
        }

        // Pure evaluation: takes immutable inputs, returns explicit value struct
        LuaScoreResult evaluate_scoring(int level, int lines_cleared, int combo, bool is_tspin) {
            lua_getglobal(L_, "calculate_score");
            lua_pushinteger(L_, level);
            lua_pushinteger(L_, lines_cleared);
            lua_pushinteger(L_, combo);
            lua_pushboolean(L_, is_tspin);

            LuaScoreResult out{};
            if (lua_pcall(L_, 4, 1, 0) == LUA_OK && lua_istable(L_, -1)) {
                lua_getfield(L_, -1, "score_added");
                out.score_added = static_cast<int>(lua_tointeger(L_, -1));
                lua_pop(L_, 1);

                lua_getfield(L_, -1, "level_up");
                out.level_up = lua_toboolean(L_, -1);
                lua_pop(L_, 1);

                lua_getfield(L_, -1, "danger_alert");
                out.danger_alert = lua_toboolean(L_, -1);
                lua_pop(L_, 1);
            }
            lua_pop(L_, 1); // Pop result table
            return out;
        }

    private:
        lua_State* L_ = nullptr;
    };

} // namespace edge
```

---

# Part 3: Complete Project Directory Layout

```text
hello-3d-demos/tetris/
├── CMakeLists.txt
├── hello_3d_tetris.cpp               # Main edge presentation loop
│
├── domains/                          # 5 Pure Domain Pods
│   ├── matrix/                       # Pod 1: Core grid simulation
│   │   ├── matrix.contract.hpp
│   │   ├── matrix.action.hpp
│   │   ├── matrix.event.hpp
│   │   ├── matrix.reducer.hpp
│   │   └── matrix.plan.hpp
│   │
│   ├── progression/                  # Pod 2: Game modes & scoring
│   │   ├── progression.contract.hpp
│   │   ├── progression.action.hpp
│   │   ├── progression.event.hpp
│   │   ├── progression.reducer.hpp
│   │   ├── progression.plan.hpp
│   │   └── scripts/
│   │       ├── blitz_rules.lua
│   │       └── marathon_rules.lua
│   │
│   ├── spatial_fx/                   # Pod 3: 3D Voxel shatter & Camera spring
│   │   ├── spatial_fx.contract.hpp
│   │   ├── spatial_fx.action.hpp
│   │   ├── spatial_fx.event.hpp
│   │   ├── spatial_fx.reducer.hpp
│   │   └── spatial_fx.plan.hpp
│   │
│   ├── powerups/                     # Pod 4: Cyber Modifiers (Bomb/Laser/Freeze)
│   │   ├── powerups.contract.hpp
│   │   ├── powerups.action.hpp
│   │   ├── powerups.event.hpp
│   │   ├── powerups.reducer.hpp
│   │   ├── powerups.plan.hpp
│   │   └── scripts/
│   │       └── cyber_modifiers.lua
│   │
│   └── environment/                  # Pod 5: 3D Diorama & Reactive Atmosphere
│       ├── environment.contract.hpp
│       ├── environment.action.hpp
│       ├── environment.event.hpp
│       ├── environment.reducer.hpp
│       └── environment.plan.hpp
│
└── edges/                            # Impure Execution Edges
    ├── audio.edge.hpp                # Lock-free procedural SPSC audio synth
    ├── rasterizer.edge.hpp           # Tiled parallel software rasterizer
    ├── lua.edge.hpp                  # Stateless thread-local Lua VM runner
    └── input.edge.hpp                # SDL2 keyboard/mouse command tokenizer
```

---

### Architectural Highlights:
1. **Multi-Threaded Lua Scalability**: Because Lua scripts are purely stateless, multiple worker threads can each evaluate scripts in parallel using isolated, lock-free `lua_State*` instances without mutexes.
2. **Instant Hot-Reloading**: Designers can edit `blitz_rules.lua` or `bomb_piece.lua` while the C++ game is running, and the new rules take effect on the next tick with zero memory corruption risk.
3. **PMR / DOD Sympathy**: C++ simulation remains $\mathcal{O}(1)$ bump-allocated and SoA-packed; Lua only touches boundary decisions.



This architecture scales seamlessly whether building a **Semi-3D Tetris**, an **FPS Combat Arena**, a **Flight Simulator**, or an **Action RPG**.

---

### Why this Template is so Powerful for Any Game:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 REUSABLE CORE FOUNDATION                               │
│                                                                                        │
│  [input.edge] ──► std::span<Action> ──► [PURE VALUE CENTER] ──► [PipelinePlan / Audio] │
│                                         (PMR Bump Arena)                               │
│                                                 │                                      │
│                  ┌──────────────────────────────┼──────────────────────────────┐       │
│                  ▼                              ▼                              ▼       │
│          [audio.edge] (SPSC)          [rasterizer.edge] (Tiled)        [lua.edge]      │
└─────────────────────────────────────────────────┬──────────────────────────────────────┘
                                                  │
                    ┌─────────────────────────────┴─────────────────────────────┐
                    ▼                                                           ▼
         Tetris Game Setup                                           FPS Arena Game Setup
         ├── domains/matrix/                                         ├── domains/combat/
         ├── domains/progression/                                    ├── domains/ai_bots/
         ├── domains/powerups/                                       ├── domains/weapon_inventory/
         ├── domains/spatial_fx/                                     ├── domains/spatial_fx/
         └── domains/environment/                                    └── domains/environment/
```

### 1. True Plug-and-Play Domains
To build a completely different game (like FPS demo or a new racing demo):
- Keep the exact same **`edges/`** (`audio.edge`, `rasterizer.edge`, `input.edge`, `lua.edge`).
- Keep the exact same **`spatial_fx`** (3D particles/camera shake) and **`environment`** (diorama/lighting) pods.
- Simply swap out `domains/matrix/` for `domains/combat/` or `domains/vehicle_physics/`.

### 2. Zero Architectural Rot
In standard OOP games, adding features eventually creates "callback spaghetti" where objects cross-call each other and leak references. In VOP/DOD:
- **No domain ever touches another domain directly**. Everything happens via clean, immutable **Discrete Event Logs**.
- I can add 50 new features or power-ups without breaking existing code.

### 3. Netcode & Rollback Ready
Because state snapshots are plain data structs and reducers are pure mathematical functions:
- State rollback is trivial (just keep the last $N$ snapshots).
- Deterministic replays take kilobytes of memory (just log the input action tokens and RNG seed).

---
