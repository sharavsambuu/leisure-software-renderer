# Todos

Do goal is to build mutiple domain pods structure can be used as template for other game demos
And also to have Lua based scripting capabilities



To thoroughly demonstrate **Value-Oriented Programming (VOP)**, **Data-Oriented Design (DOD)**, and a **Stateless Lua Scripting Layer** without over-engineering, the sweet spot is **5 Domain Pods** combined with **3 Execution Edges**.


---

## Status (as of 2026-08-22)

| Item | State |
| :--- | :--- |
| Pod 1 `matrix` | ✅ DONE — contract/action/event/reducer; pure center, zero scoring refs |
| Pod 2 `progression` | ✅ DONE — event-fed scoring, combos, levels, victory; Lua seam isolated at `compute_line_clear_score()` |
| Pod 3 `spatial_fx` | ✅ DONE — SoA particles, camera spring, scene planner (`spatial_fx.plan.hpp`) |
| Pod 4 `powerups` | ⬜ PENDING — arrives together with its `scripts/*.lua` (Part 2) |
| Pod 5 `environment` | ⬜ PENDING — diorama + reactive mood lighting |
| Edges `input` / `audio` / `rasterizer` / `ui` | ✅ DONE — one subdirectory per edge (`edges/<name>/tetris.<name>.hpp`) |
| Edge `lua` | 🟡 SCAFFOLDED — `edges/lua/lua.edge.hpp` compiles into the build, not yet wired into the loop |
| Thin main + `verify.sh` | ✅ DONE — determinism / behavioral-delta / purity gates all PASS |

Companion docs in this folder: `NOTES.md` (VOP/DOD pod theory + Lua philosophy) and
`REFACTOR_PROPOSAL.md` (migration record). Repo-root `docs/DETAILS.md` §4 holds the
concrete Lua wiring design for the next task.

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

### 2.3 The C++ Lua Edge Wrapper (`edges/lua/lua.edge.hpp`)

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

# Part 3: Project Directory Layout (as built)

```text
hello-3d-demos/tetris/
├── CMakeLists.txt                       # One -I root (demo dir); target-scoped include dirs
├── verify.sh                            # Headless gates: determinism / delta / purity greps
├── hello_3d_tetris.cpp                  # Main edge: SDL lifecycle, PMR arena, loop wiring (~330 lines)
│
├── config/                              # Pure designer-facing tuning data
│   ├── rules.hpp                        #   tetris::config::Rules + gravity_for_level()
│   └── levels/marathon_01.hpp           #   Marathon01::make_rules()
│
├── domains/                             # Pure Value Center (zero SDL, zero global heap)
│   ├── matrix/                          # Pod 1: core grid simulation        [DONE]
│   │   ├── matrix.contract.hpp          #   MatrixSnapshot, ActivePiece, pull_next_piece
│   │   ├── matrix.action.hpp            #   TetrisCommand variant + reduce_tetris_commands
│   │   ├── matrix.event.hpp             #   MatrixEvent raw facts (LOCK_IMPACT, LINES_CLEARED, …)
│   │   └── matrix.reducer.hpp           #   reduce_matrix — pure transition, emits events ONLY
│   │
│   ├── progression/                     # Pod 2: scoring & modes             [DONE]
│   │   ├── progression.contract.hpp     #   ScoreState
│   │   ├── progression.event.hpp        #   ProgressionEvent (SCORE_CHANGED, LEVEL_UP, …)
│   │   ├── progression.reducer.hpp      #   reduce_progression + compute_line_clear_score [LUA SEAM]
│   │   └── scripts/                     #   (PENDING) blitz_mode.lua, sprint_40lines.lua
│   │
│   ├── spatial_fx/                      # Pod 3: FX + scene planning         [DONE]
│   │   ├── spatial_fx.contract.hpp      #   FxState, ProcessedTriangle, piece palette
│   │   ├── spatial_fx.reducer.hpp       #   step_fx — SoA particles, deterministic xorshift
│   │   └── spatial_fx.plan.hpp          #   plan_tetris_scene → PipelineExecutionPlan
│   │
│   ├── powerups/                        # Pod 4: cyber modifiers             [PENDING — lands with scripts/*.lua]
│   └── environment/                     # Pod 5: diorama + mood lighting     [PENDING]
│
├── edges/                               # Impure Execution Edges (SDL appears here ONLY)
│   ├── input/tetris.input.hpp           #   poll_input → InputState{ pmr commands }
│   ├── audio/tetris.audio.hpp           #   SPSC ring procedural synth + SDL audio callback
│   ├── rasterizer/tetris.rasterizer.hpp #   vop:: clip_to_screen_vec4, rasterize_triangle_tile
│   ├── ui/tetris.hud.hpp                #   draw_hud(canvas, MatrixSnapshot, ScoreState)
│   └── lua/lua.edge.hpp                 #   StatelessLuaEvaluator            [SCAFFOLDED, unwired]
│
└── docs/
    ├── TODOS.md                         # this file — canonical blueprint + status
    ├── NOTES.md                         # VOP/DOD pod theory + Lua-in-VOP philosophy
    └── REFACTOR_PROPOSAL.md             # migration record of the domain-pod refactor
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