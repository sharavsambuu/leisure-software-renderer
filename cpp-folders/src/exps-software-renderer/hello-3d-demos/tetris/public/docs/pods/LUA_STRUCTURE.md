# LUA STRUCTURE - Unity concepts mapped + the four-file level standard

> Answers three questions from the P4+ session (2026-08-25): can this
> architecture reach Unity-style Entity/Prefab concepts? Can a whole level
> be implemented purely in Lua? How should Lua files be structured as the
> project grows? Short answers: yes (already true for entities), yes (one
> step away - mission/goal scripting), and the four-layer convention below.

---

## 1 - Unity concepts mapped to Domain PODs

| Unity concept | Our equivalent | Status |
| --- | --- | --- |
| GameObject (entity ID) | row index into pod arrays / entity ID (FPS_EXAMPLE.md section 2) | concept shipped; SoA world pods arrive with the first FPS-style game |
| Component | one pod's per-entity row (health[], posX[], ...) | same thing, different name |
| Prefab | prefab table: pure-data Lua file describing components | reachable; needs an entity factory (`spawn_from_prefab`) on the next game |
| Scene | GameWorld snapshot + stage def | shipped |
| Prefab instantiation | `spawn_from_prefab(world, prefab, params)` - pure fn writing new rows | future game work (~1 week) |
| ScriptableObject | rules/level Lua tables | SHIPPED (campaign.lua, level data) |

Caveat vs Unity: their editor does visual composition. We author prefabs in
Lua/text instead - diffable, greppable, LLM-editable.

## 2 - Pure-Lua levels: what is already true, what remains

Already Lua-authored: blitz economy, canyon generation, cyber-storm
mechanics, encore orchestration, campaign manifest, all level rule
overrides (P3).

The last gap for a FULLY Lua level is the goal/mission layer:
- domains/mission/ pod skeleton (progress + fact-chained sequencing)
- event batch exposed to Lua: goal.test(events, snapshot)
- predicate DSL: when / count_where / during (~60 lines of Lua)

Tracked as TODOS Part 7 G1-G4. After that, one .lua file authors an entire
level: board shape, rules overrides, spawn cadence, victory conditions -
zero C++ changes.

## 3 - The four-file level standard

    assets/levels/cyber_storm/
    |-- init.lua          entry point; returns the Level table (DATA ONLY)
    |-- rules.lua         rule hooks: get_config, decide_spawn, on_special_lock
    |-- goals.lua         mission predicates using the DSL
    `-- presentation.lua  optional: mood/floater/audio recipe choices

Rules:

1. init.lua is DECLARATIVE. Data only. If it needs logic, it belongs in a
   sibling file.
2. All functions are PURE - same sandbox gates as today (no io/os/random),
   enforced by the script_purity ctest.
3. ONE global table per file (existing pattern: Encounter, BlitzRules,
   CanyonGen) so the host looks up hooks by name.
4. Events in, values out. Every hook signature takes plain data and returns
   a plain table. Never mutate anything the host passed.
5. File names are conventional; the host resolves them relative to the
   level directory. Missing goals.lua/presentation.lua = level simply has
   no missions/custom presentation (graceful degradation).

Benefits: hot-reloadable levels, designer-friendly tuning, LLM-editable
content, determinism intact, pod purity untouched.
