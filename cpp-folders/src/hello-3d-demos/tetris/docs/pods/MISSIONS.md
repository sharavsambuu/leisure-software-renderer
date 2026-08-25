# CASE STUDY — Missions & Higher-Level Goals in Domain PODs

> Second genre-portability study (with FPS_EXAMPLE.md). Answers: how are
> game goals — objectives, quests, mission chains — programmed in the pod
> architecture? Short answer: the mission/objective system is another pod,
> with one twist — it owns only MISSION progress, never duplicated game
> truth. Conditions are checked by consuming the same facts every other
> listener consumes.

---

## 1 · The key insight: missions own PROGRESS, not TRUTH

Common mistake — duplicating other pods' state inside the mission system:

```js
// ❌ WRONG — copies of facts other pods already own
mission = {
  linesCleared: 0, // progression owns this → now two truths disagree
  enemiesAlive: 12, // world owns this → staleness bugs guaranteed
};
```

The scorch-residue lesson (§10.8, Session 28) is exactly this disease:
copied/guessed data silently drifting from reality.

Correct shape — the mission pod owns ONLY mission-specific truth:

```js
// mission.contract.js
export function createMissionState(defs = []) {
  return {
    defs, // static definitions (from level.js)
    activeIndex: 0, // which mission is current (-1 = none)
    stage: "pending", // pending | active | complete | failed
    progress: 0, // THIS goal's counter — mission-owned
    timeLeft: null, // for timed goals only
  };
}
```

Everything else ("did a kill happen?", "is the player frozen?") is derived
by consuming facts or reading snapshots passed in — never stored.

---

## 2 · The hierarchy: campaign → missions → ticks

Missions slot between your existing layers:

```
campaign.js      "which mission SETS exist, unlock order"   (CAMPAIGN pattern)
    ↓ picks
mission pod      "current objective + live progress"        ← NEW POD
    ↓ emits facts
MISSION_COMPLETE{id} / MISSION_FAILED{id} / OBJECTIVE_UPDATED{progress}
    ↓ consumed by
session          RESULTS screen · next-mission transition
progression      completion bonus scoring (SPECIAL_SCORE seam reusable)
fx/hud           celebration, new-objective banner, progress bar
```

Structurally identical to the L1→L5 campaign unlock chain already shipped —
missions are just a FINER-GRAINED version of the same progression idea.

---

## 3 · Two condition patterns

### Pattern A — Event traps (most missions)

Exactly like hud-pod floaters or CHAIN_CLEAR:

```js
// reduceMission(state, events) — trap wall
for (const ev of events) {
  if (state.stage !== "active") break;
  const goal = state.defs[state.activeIndex];

  switch (goal.type) {
    case "kill_count":
      if (ev.type === "ENTITY_KILLED" && ev.killerTeam === 0) state.progress++;
      break;
    case "clear_count":
      if (ev.type === "LINES_CLEARED") state.progress += ev.count;
      break;
    case "chain_clear":
      if (ev.type === "CHAIN_CLEAR") state.progress++;
      break;
  }
  checkComplete(state, goal);
}
```

No polling. No coupling to who caused the kill. Adding a new mission TYPE is
one more case — listeners elsewhere untouched.

### Pattern B — Snapshot queries (state-shape goals)

For conditions no single event expresses: "reach the extraction zone",
"survive with health above 50%". These read a read-only SNAPSHOT argument:

```js
// reducer signature grows a snapshot param (read-only views)
export function reduceMission(state, events, snapshot, dt) {
  ...
  case 'reach_zone': {
    const dx = snapshot.playerX - goal.x;
    const dz = snapshot.playerZ - goal.z;
    state.progress = Math.hypot(dx, dz) < goal.r ? 1 : 0;
  }
  case 'survive_healthy':
    state.progress = snapshot.healthRatio >= 0.5 ? 1 : 0;
}
```

**Purity note**: READS from a passed-in snapshot do not break purity; WRITES
to other pods would. Same rule as everywhere else.

Precedent: the L5 environment pod already does multi-input decisions —
`decidePhase(phase, phaseTime, lines, danger)` reads several inputs and
emits phase facts.

---

## 4 · Mission definitions live in levels/ — scripter ladder applies

Same three tiers as L1–L5:

| Tier          | Example                                   | Implementation                                                          |
| ------------- | ----------------------------------------- | ----------------------------------------------------------------------- |
| Native        | "clear 8 lines"                           | generic counter trap, configured by NUMBERS only                        |
| Scripted hook | "kills count only while overdrive active" | `checkGoal(goal, ev, snapshot)` function in level rules.js              |
| Orchestration | multi-stage chains w/ custom logic        | dedicated sub-pod emitting its own facts (environment-overseer pattern) |

Most missions = DATA in level.js, zero engine changes:

```js
// levels/prison-break/level.js
export default {
  id: 2,
  name: 'PRISON BREAK',
  makeRules() { ... },
  missions: [
    { id: 'm1', type: 'kill_count', target: 8,
      hintKey: 'mission.eliminate' },
    { id: 'm2', type: 'survive_time', seconds: 45 },
    { id: 'm3', type: 'reach_zone', x: 40, z: 60, r: 5 },
  ],
};
```

New mission = new level entry. The mission pod interprets generically —
the same way L2/L3/L4 were "data + listeners."

---

## 5 · Sequencing: order flows through FACTS

"Missions come in order m1 → m2 → m3. Isn't that a sequence?" Yes — and pods
handle it via fact chaining, not a controller loop:

```
checkComplete(m1) → emit MISSION_COMPLETE{id:'m1'}
                 → own trap on that fact activates m2 (activeIndex++)
```

The pod's trap wall catches its OWN completion fact and advances. Order
through events, not through an imperative runner. Failure path symmetric:
`MISSION_FAILED` → either retry (reset progress, stay on index) or jump to a
fail mission defined in level.js.

Timed goals decay `timeLeft -= dt` in the same reducer; `timeLeft ≤ 0`
before completion → MISSION_FAILED.

---

## 6 · Presentation follows §10.7 signal budget

- ONE objective banner (text + progress bar), persistent while active
- ONE completion moment: banner swap + sound (+ fx celebration if earned)
- Progress bar updates are quiet; completion is loud
- Mission text via locale keys (`mission.eliminate`, ...) — i18n parity gate

Anti-example avoided deliberately: five overlapping signals per objective
(the L4 pre-S1 problem).

---

## 7 · Event flow additions (generator PRODUCERS table)

| Fact              | Producer    |
| ----------------- | ----------- |
| MISSION_COMPLETE  | mission pod |
| MISSION_FAILED    | mission pod |
| OBJECTIVE_UPDATED | mission pod |

Consumers: session (transitions), progression (bonuses), hud (banner/bar),
audio (jingle). Add these rows when implementing — the ⚠️ undeclared-
producer warning will enforce registration automatically.

---

## 8 · Test pins (write RED first)

- M1 generic counters: kill_count increments on ENTITY_KILLED(team=0) only
- M2 completion emits MISSION_COMPLETE exactly once
- M3 sequencing: completing m1 activates m2 (fact-chained, not indexed by
  external controller)
- M4 overdrive-gated goal counts only flagged clears (scripted tier)
- M5 snapshot goal: reach_zone completes inside radius, not outside
- M6 timed goal: expiry emits MISSION_FAILED before completion possible
- M7 determinism: same event stream → identical mission progress
