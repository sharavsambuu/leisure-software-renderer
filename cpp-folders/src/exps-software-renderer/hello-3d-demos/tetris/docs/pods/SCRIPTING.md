# SCRIPTING - composable mission conditions (predicate DSL)

> The third layer of game-goal programming. Layer 1: hand-written traps
> (L4 reducers). Layer 2: declarative mission DATA interpreted generically
> (MISSIONS.md). Layer 3 - THIS doc: level authors compose CONDITIONS as pure
> predicate functions using a tiny DSL, so new missions are content, not code.

---

## 1 - The DSL: domains/lib/event-queries.js

A predicate is a plain function: (events[], snapshot) -> boolean.
Combinators build bigger predicates from smaller ones.

| Builder                    | Meaning                                                    |
| -------------------------- | ---------------------------------------------------------- |
| when(...conds)             | ALL must hold for this batch (alias allOf)                 |
| anyOf(...conds)            | at least one holds                                         |
| noneOf(...conds)           | none hold                                                  |
| not(cond)                  | inversion                                                  |
| countWhere(TYPE[, filter]) | counter with .atLeast(n) / .exactly(n) / .fired            |
| during(flag)               | snapshot window gate (snap.overdrive etc.)                 |
| once(cond)                 | fires at most on first satisfaction (short-circuits after) |

Purity law: same events + same snapshot give the same answer. No Date.now,
no Math.random, no reads outside the arguments. Same lint gates apply to
level scripts using this DSL.

## 2 - Authoring a mission (level scripter view)

```js
// levels/cyber-storm/missions.js - scripter home, pure functions only
import {
  when,
  during,
  countWhere,
} from "game/domains/lib/event-queries.js";

export const missions = [
  // cumulative counter style: relevant(ev) + target
  {
    id: "m1",
    relevant: (e) => e.type === "ENTITY_KILLED" && e.killerTeam === 0,
    target: 8,
    hintKey: "mission.eliminate",
  },
  // batch predicate style: test(events, snapshot)
  {
    id: "m2",
    test: when(during("overdrive"), countWhere("LINES_CLEARED").atLeast(3)),
    hintKey: "mission.overdriveRush",
  },
];
```

Two goal STYLES by design:

- cumulative (relevant + target): progress accumulates across ticks;
  completion when progress >= target
- batch predicate (test): true in a single tick completes instantly

## 3 - The interpreter: domains/mission/mission.reducer.js

Generic, ~40 lines, never changes per mission:

- evaluates active goal against the tick batch + read-only snapshot
- cumulative goals add weight(ev) ?? 1 per relevant event
- completion pushes MISSION_COMPLETE{id} and fact-chains to the next goal
- timed goals (timeLimit) decay by dt; expiry emits MISSION_FAILED
- stage machine: pending -> active -> complete | failed

The ENGINE derives snapshot flags from matrix state (e.g.
snapshot.overdrive = gravityFreeze > 0) - the interpreter itself stays
framework- and domain-agnostic.

## 4 - Composition examples

```js
// kill 6 enemies BY THE PLAYER while overdriven
when(
  during("overdrive"),
  countWhere("ENTITY_KILLED", (e) => e.killerTeam === 0).atLeast(6),
);

// survive 30s without letting stack reach 15 (negation of a bad state)
noneOf(countWhere("GAME_OVER").fired);

// EITHER a tetris OR three chained chain-clears
anyOf(
  countWhere("LINES_CLEARED", (e) => e.count >= 4).atLeast(1),
  countWhere("CHAIN_CLEAR").atLeast(3),
);
```

Each node is one predicate function - the expression tree IS the graph you
would have drawn (PLANNING.md section 2).

## 5 - Guardrails (learned from L4)

1. Predicates stay PURE: no Date.now/Math.random/side effects; seeded RNG
   lives in pods, not scripts.
2. One-tick latency preserved: predicates evaluate against facts already
   emitted this tick - determinism untouched.
3. Signal budget applies to rewards: completing a mission fires ONE
   celebration (section 10.7).
4. The DSL stays TINY: conditions only. Loops/state/behavior = write a
   tier-3 sub-pod instead (environment-overseer pattern).
5. Cumulative vs predicate: counting across ticks needs the cumulative
   style - a batch predicate only sees THIS tick's events.

## 6 - Test pins (implemented - GREEN)

- D1 when/allOf/anyOf/noneOf/not combinators over synthetic batches
- D2 countWhere filter + atLeast/exactly variants
- D3 during() snapshot gating
- D4 once() short-circuit semantics
- M1 cumulative kill-count increments only on relevant events
- M5 completion emits MISSION_COMPLETE exactly once, chains to next goal
- M7 same stream gives identical progress (determinism)
- M8 timed expiry emits MISSION_FAILED
