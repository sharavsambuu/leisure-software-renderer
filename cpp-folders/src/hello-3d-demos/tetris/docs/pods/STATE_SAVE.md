# STATE & SAVE - serializing pod snapshots (checkpoints, saves, replays)

> Every pod snapshot in this architecture is PLAIN DATA - JSON-serializable
> by construction. That makes save/load nearly free, but versioning and
> edge-cases deserve their own doc.

---

## 1 - Why it is almost free

Pod state is already: plain objects, typed arrays or number arrays, no
closures, no class instances, no DOM references. `JSON.stringify(snapshot)`
just works for most pods. The architecture's purity laws ARE the save-system
design.

---

## 2 - What a save contains

A full run checkpoint:

```json
{
  "version": 3,
  "stageId": 4,
  "seed": 137,
  "tick": 4821,
  "pods": {
    "matrix": { "...full matrix snapshot..." },
    "progression": { "...scoreState..." },
    "powerups": { "...cadence/cycles..." },
    "mission": { "...active goal + progress..." }
  }
}
```

- fx/hud are NOT saved - they are presentation/transient state; rebuilt fresh
  on load (same as C++ games not saving particles).
- environment brain IS saved if the stage uses one.
- `version` is mandatory - see section 4.

## 3 - Save vs replay (two different mechanisms)

|            | Save file                 | Replay                                   |
| ---------- | ------------------------- | ---------------------------------------- |
| Contains   | pod snapshots at tick T   | seed + full command stream               |
| Size       | large-ish                 | tiny                                     |
| Use        | resume where you left off | kill-cam, bug reproduction, leaderboards |
| Fragile to | pod shape changes         | ANY game logic change                    |

Both are enabled by determinism; they solve different problems. A replay
reconstructs any tick from tick 0; a save jumps directly to tick T.

## 4 - Versioning (the real work)

The only genuinely hard part of save/load: pod shapes evolve (L4 added
fields to powerups state twice). Rules:

1. Each save carries `version`; a migration function chain upgrades old
   saves step by step (`v1→v2`, `v2→v3`) rather than one big transform.
2. Migrations live in ONE module per pod (`world/migrations.js`), tested -
   migration bugs corrupt player progress silently.
3. Unknown fields are DROPPED on load (forward compatibility); missing
   fields get factory defaults via `{...createXState(), ...loaded}` spread.
4. Typed arrays serialize as regular arrays; restore with
   `Float32Array.from(saved)`.

## 5 - Checkpoint policy (when to save)

Saving every tick is waste. Natural checkpoints:

- mission complete / failed (MISSIONS.md lifecycle edges)
- stage start / restart
- explicit player save (menus)

Mid-run autosave only if death penalty design demands it - and then throttle
(e.g. every 30s max).

## 6 - Edge ownership

Like everything else: serialization lives in an EDGE (storage-edge sibling).
Pods never know JSON exists. Engine exposes `serialize()` / `deserialize()`
that walk pod contracts explicitly (not blind deep-copy) so adding a field
forces a conscious save-format decision.

## 7 - Test pins

- S1: serialize → deserialize → deep-equal original snapshot
- S2: v(n) save migrates to v(n+1); no data loss of surviving fields
- S3: unknown extra fields dropped without error
- S4: replay from seed reproduces the same tick-T state the save holds
  (save/replay equivalence - the strongest consistency pin)
