# PERFORMANCE - budgets, profiling, and the zero-allocation discipline

> The architecture is performance-friendly BY DESIGN (fixed-step sim,
> imperative edges, pooled InstancedMeshes, zero per-frame allocation). This
> doc records how to KEEP it that way: budgets, how to profile, and the
> rules that protect the discipline.

---

## 1 - Budgets (per 60 Hz frame = 16.6 ms)

| Segment             | Budget | Notes                                            |
| ------------------- | ------ | ------------------------------------------------ |
| simulation tick     | < 4 ms | reducers are cheap; watch combat raycasts in FPS |
| fx step             | < 1 ms | particle counts bounded by pool caps             |
| draw edge (JS side) | < 2 ms | instance matrix composition                      |
| GPU (draw + post)   | < 9 ms | bloom is the biggest cost; quality-flagged       |

If a segment exceeds budget twice in a row, profile BEFORE optimizing -
guesses are wrong more often than right.

## 2 - The one rule that protects everything

**Zero allocation inside the tick/frame loop.**

- scratch objects (m4, q, pos, scl, col) live at module scope in draw-frame
- particles/rings/scorch/beams use in-place compact loops, not filter/map
- event arrays DO allocate per tick - acceptable; they escape into listeners
  and are short-lived young-gen objects

Watch for regressions: any `.map`/`.filter`/spread inside `#tick` or
`drawFrame` deserves a justification comment.

## 3 - Profiling workflow (Chrome DevTools)

1. Performance tab, record 5 s of gameplay incl. a detonation
2. Look for: long tasks > 16 ms; GC spikes (sawtooth memory); layout thrash
3. Frame Timing / requestAnimationFrame timing for dropped frames
4. Memory tab: allocation instrumentation on timeline - heap snapshots before/
   during gameplay; growing sawtooth baseline = leak

Common findings mapped:

- GC spikes -> an allocation crept into the loop (search .map/.filter/spread)
- Long JS tasks -> reducer doing too much per tick (raycasts? stagger AI)
- GPU bound -> drop bloom first (quality flag exists)

## 4 - Quality scaling ladder

Already shipped: bloom/afterimage skipped when devicePixelRatio > 2.
Extend the same pattern: particle count multiplier, glow-pool cap, shadow
toggles - each behind ONE check, degrading gracefully rather than stuttering.

## 5 - Determinism vs perf exceptions

The only accepted purity/perf exception so far: camera rotation at frame rate
(FPS_EXAMPLE.md section 7). Any future exception needs the same treatment: a
comment, a doc entry, and a test proving the sim path stays deterministic.
