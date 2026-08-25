# Domain PODs — portable architecture docs

> Framework/genre-agnostic knowledge distilled from the Tetris port and its
> JS twin (hello-ember-tetris). Start here when designing a new system,
> a new level, or a new GAME on this skeleton.

## Index

| Doc | What it gives you |
| --- | --- |
| [PLANNING.md](PLANNING.md) | 7-step design workflow: event storming -> pod boundaries -> state machines -> producers table -> RED tests -> build |
| [EVENT_FLOW.md](EVENT_FLOW.md) | generated fact->consumer map + per-pod trap tables (Tetris instance; regenerate for other games) |
| [FPS_EXAMPLE.md](FPS_EXAMPLE.md) | genre study: entities as IDs in SoA arrays; entity VIEWS vs objects (never store refs); aim-latency clock split |
| [MISSIONS.md](MISSIONS.md) | goals/quests as a pod: owns PROGRESS not truth; event traps vs snapshot queries; fact-chained sequencing |
| [SCRIPTING.md](SCRIPTING.md) | composable goal conditions: predicate DSL over event batches (JS reference impl; Lua port guidance inside) |
| [AI_PODS.md](AI_PODS.md) | NPC brains as intent generators; perception fairness law; difficulty = brain data |
| [STATE_SAVE.md](STATE_SAVE.md) | serializing pod snapshots; save vs replay; versioned migrations |
| [BALANCING.md](BALANCING.md) | difficulty-as-data tiers; telemetry metrics; the tuning loop |
| [PERFORMANCE.md](PERFORMANCE.md) | frame budgets; zero-allocation discipline; profiling workflow |

---

## The architecture in one page

**Three movement types** (all control flow is one of these):

1. **Commands** — intent entering from edges (player input edge, AI pods)
2. **Facts** — outcomes announced by owner pods (past tense, plain data)
3. **Rulings** — deferred intents landing NEXT tick (level script decisions)

**The iron rules:**

1. Reducers are pure: `(state_t, commands, dt) -> (state_{t+1}, events)`
2. Only the owner-pod mutates its own state
3. Events are dumb data — past tense, no behavior
4. All platform contact lives in edges (SDL, audio, rasterizer, lua, ui)
5. State lives in pods as flat arrays/structs; borrow views per function call
6. Order flows through events, not call stacks
7. One event, one signal
8. Juice scales with impact

**The planning law:** storm the facts, draw the graph, fill the producers
table, write failing tests first — THEN implement.

---

## Porting checklist (new game on this skeleton)

1. Keep edges: input/audio/rasterizer/ui/lua patterns
2. Swap domain-specific pods; keep session/spatial_fx/environment skeletons
3. New `config/campaign` manifest + `config/levels/<name>/`
4. Regenerate the event-flow doc for the new fact set
5. Keep gates: pod purity greps, determinism runs, script sandbox checks
