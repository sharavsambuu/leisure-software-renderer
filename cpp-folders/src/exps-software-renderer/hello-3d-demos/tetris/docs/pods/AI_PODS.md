# AI PODS - NPC brains in the Domain POD architecture

> Companion to FPS_EXAMPLE.md (section 2: intent sources are interchangeable)
> and MISSIONS.md. This doc covers how NPC artificial intelligence is
> programmed as pure pods: AI decides INTENTS, never applies outcomes -
> identical to the player's input edge, just a different author.

---

## 1 - The core law

> **An AI pod is an intent GENERATOR.** It reads world snapshots and emits
> the SAME command shapes the input edge emits for the player. It never
> mutates other pods, never raycasts, never damages - it only asks.

```
Player path:  keyboard edge --> MOVE{vec}, FIRE --> movement/weapon pods
NPC path:     AI reducer -----> MOVE{vec}, FIRE --> same pods
```

Downstream cannot tell them apart. That interchangeability is the design
goal (FPS_EXAMPLE.md section 2) - and it makes AI testable without
rendering: feed facts, assert intents.

---

## 2 - AI tiers (same scripter ladder as levels)

| Tier            | Behavior                          | Implementation                                             | Example                        |
| --------------- | --------------------------------- | ---------------------------------------------------------- | ------------------------------ |
| 1 Scripted      | fixed patterns, no decisions      | data tables: patrol routes, spawn timers                   | turret with sweep pattern      |
| 2 Utility       | score possible actions, pick best | pure scoring functions over snapshot                       | enemy chooses cover vs advance |
| 3 State machine | phase behavior per situation      | XState-style machine per archetype (PLANNING.md section 4) | boss: enrage below 30% hp      |

Rule of thumb: start tier 1; climb only where playtesting shows dumb
behavior. Most believable "smart" enemies are tier 2 with good numbers.

---

## 3 - The perception problem - what may the AI READ?

Fairness law: **an AI pod may read only what a player could plausibly know.**

- ALLOWED: own position/health, last-known player position, publicly
  broadcast facts (gunshot events carry shooter position), team membership,
  timers
- FORBIDDEN: current player position through walls, player input buffer,
  future RNG draws

Implementation: the ENGINE passes each AI pod a **perception snapshot**
(filtered view), not the raw world state:

```js
// engine tick, per AI entity:
const perception = buildPerception(world, aiId); // pure filter
const [brainNext, intents] = reduceAI(aiBrain[id], perception, dt);
commands.push(...intents.map((i) => ({ ...i, id: aiId })));
```

`buildPerception` is itself a PURE function - testable: given world + fog
rules, assert exactly which facts leak. Cheating AIs are a perception bug,
not a tuning issue.

---

## 4 - Anatomy of an AI reducer

```js
// ai/reducer.js - pure (brainState, perception, dt) -> intents[]
export function reduceAI(brain, p, dt) {
  // brain = per-NPC memory: target memory, alertness, cooldowns
  const b = { ...brain };

  // TRAP: heard a gunshot recently -> become alert, remember direction
  if (p.heardShot && p.sinceShot < 3) {
    b.alertness = Math.min(1, b.alertness + dt * 2);
    b.lastKnownPlayerPos = p.shotOrigin;
  }

  // DECIDE (tier-2 utility): score options, emit best as intents
  const intents = [];
  if (b.alertness > 0.4 && p.canSeePlayer) {
    intents.push({ type: "AIM", id: p.id, at: p.playerPos });
    if (p.inRange && b.fireCooldown <= 0)
      intents.push({ type: "FIRE", id: p.id });
  } else {
    intents.push({ type: "MOVE", id: p.id, toward: b.lastKnownPlayerPos });
  }
  return [b, intents];
}
```

The shape matches every other pod: `(state, inputs) -> next + outputs`.
The only novelty: its OUTPUTS are commands aimed back into the same pipeline
the player feeds.

---

## 5 - Determinism and difficulty

- Same laws as everything else: seeded randomness only (`aiRngState` inside
  brain), no Date.now / Math.random.
- Difficulty = DIFFERENT BRAIN DATA, not new code paths:
  `{ accuracy: 0.8, reactionMs: 400, aggression: 0.6 }` vs an easy variant.
  Levels select brain configs like they select rules (scripter ladder).
- Replay/kill-cam free: AI intents ride the recorded command stream.

---

## 6 - Performance notes

- N AIs ticking every frame is wasteful; most need decisions at 5-10 Hz.
  Stagger: `if ((tick + entityId) % 6 === 0) think()` - spreads cost while
  keeping determinism (staggering is a pure function of tick + id).
- Perception filtering is the expensive part (visibility checks). Cache
  last-known results per pair; invalidate on relevant FACTS, not on time.

---

## 7 - Test pins (RED-first when implementing)

- A1: AI emits only command types the input edge also produces (contract)
- A2: perception filter hides player position behind walls (no cheating)
- A3: same seed + same fact stream gives identical intent sequence
- A4: difficulty config changes behavior without code-path changes
- A5: staggered thinking - AI reacts within N ticks worst case
- A6: dead AI emits nothing
