# BALANCING & DIFFICULTY - tuning numbers without guessing

> Pods make tuning SAFE (all knobs live in rules.js / level files) but safe
> is not the same as CORRECT. This doc records how to find good values
> systematically instead of ad-hoc edits (see STATUS Session 9: pacing tuned
> by feel; it worked, but only because a playtest said "too slow" - telemetry
> would have shown it earlier).

---

## 1 - Difficulty = data, not code

Same principle as AI difficulty (AI_PODS.md section 5): difficulty levels are
DIFFERENT RULE OBJECTS, selected by level or player choice:

```js
// config/difficulty.js
export const DIFFICULTY = {
  easy: { gravityBase: 0.6, lockDelayMs: 700, specialCadence: 10 },
  normal: { gravityBase: 0.8, lockDelayMs: 500, specialCadence: 8 },
  hard: { gravityBase: 1.0, lockDelayMs: 350, specialCadence: 7 },
};
```

makeRules() merges the chosen tier over base rules. No code paths change -
same law as AI brain configs.

## 2 - The balancing loop

```
1. DEFINE what "balanced" means per metric (target values + tolerance)
2. INSTRUMENT: emit facts already carry the data (score, clears, deaths,
   time-to-clear); add a telemetry edge that aggregates them
3. PLAYTEST with real humans at target skill
4. MEASURE against targets; look for outliers
5. ADJUST ONE KNOB at a time; re-run
```

## 3 - Metrics worth tracking (per run)

| Metric                   | Healthy signal                             |
| ------------------------ | ------------------------------------------ |
| time-to-first-clear      | under ~20s on stage 1                      |
| deaths per stage attempt | decreasing across retries                  |
| special survival rate    | players use >50% of specials (L4)          |
| completion rate          | 40-60% first session (harder stages lower) |
| quit point               | where players stop = frustration spike     |

The L4 lesson maps directly: "combo never fires" was invisible until traced;
telemetry surfaces these without waiting for verbal feedback.

## 4 - Tuning knobs inventory (keep updated)

rules.js owns most knobs: targetScore, linesPerLevel, gravityDecay,
baseScores table, b2bMultiplier, comboBonus, attackLines... Levels override
via makeRules(). AI brains own accuracy/reaction/aggression. Mission defs own
targets/timeouts.

Rule: every new numeric constant in a reducer is A SMELL - move it to rules
and give it a default.

## 5 - Storm-ramp example (B3 enablement workflow)

decideCadence(score) shipped returning undefined (OFF). Enabling it properly:

1. Define intent: endgame pressure arc for score chase
2. Instrument current cadence-8 runs; collect average score curves
3. Pick milestones from DATA (e.g. median player reaches 6000 at ~90s):
   return 7 at 6000, 6 at 9000
4. Playtest; compare completion rates before/after
5. Keep or revert - one knob, measurable
