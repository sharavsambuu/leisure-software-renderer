# DEBUG & DEVELOPMENT - loading any level without unlocking

> How to boot straight into any campaign stage for debugging/tuning, how the
> unlock system works in regular play, and what dev flags exist.

---

## 1 · Regular (player) flow

- Boot → TITLE → LEVEL_SELECT. The carousel shows only
  `session.unlocked_stages` entries.
- `unlocked_stages` starts at **1** (MARATHON only). Beating a stage emits
  VICTORY → RESULTS auto-advances and bumps unlocks by one
  (`unlocked_stages = max(unlocked, current_stage + 2)`).
- So yes: regular mode requires beating stages in order.

## 2 · Dev shortcuts (already built in)

| Flag | Effect |
| --- | --- |
| `--stage=N` | windowed: boots STRAIGHT into stage N fully unlocked for that session (`unlocked_stages = all`). No menu needed. Headless (`--frame=N`): same + skips menus to PLAYING |
| `--script=<file>` | override the stage's Lua script with any file (hot-swapping rules for experiments) |
| `--seed=N` | fixed seed for L3-style generators (reproducibility) |
| `--screenshot --frame=N` | headless run, dump frame N to BMP (determinism gates) |
| `--autodrive-harddrop` | inject a synthetic HARD_DROP at frame 30 (input-pipeline smoke) |

Examples:

```bash
# jump into Cyber Storm, windowed, no unlocking needed:
./build_vcpkg/src/hello-3d-demos/tetris/Hello3DTetris --stage=4

# same but swap in an experimental rule script:
./...Hello3DTetris --stage=4 --script=assets/levels/cyber_storm/rules_experimental.lua

# headless determinism probe of Encore Finale at frame 45:
SDL_VIDEODRIVER=dummy ./...Hello3DTetris --stage=5 --screenshot /tmp/x.bmp --frame=45
```

Note: `--stage` unlock-forcing is SESSION-scoped — it never writes progress;
a normal relaunch returns to whatever the player has legitimately unlocked.

## 3 · Level content iteration loop (P3 data-driven)

1. Edit `assets/levels/<id>/level.lua` (rules_overrides) or
   `assets/campaign/campaign.lua` (order/unlocks/scripts)
2. Relaunch with `--stage=N` — no recompile; Lua is loaded at boot
3. If the level fails to load: fallback = MARATHON defaults + stderr note
   `[campaign] campaign.lua unavailable`

Rule-script iteration is even faster: edit the script, relaunch — each stage
gets a FRESH sandbox per load (no global leaks).

## 4 · Testing hooks

- `ctest --test-dir cpp-folders/build_vcpkg/tetris` — unit/whole-frame/
  input-harness/purity suites (5 targets, ~60 checks)
- `bash verify.sh` — full battery: determinism byte-compares, script-economy
  smokes, purity greps, UNIT gate
- `tests/goal_bridge_tests.cpp` — proves scripted goals evaluate through the
  sandbox (G1)

## 5 · Planned dev conveniences (not yet built)

- [ ] `--unlock-all` persisted flag (currently --stage covers it per-session)
- [ ] `--list-stages` printing id/name/script from campaign.lua
- [ ] Hot-reload key (F5) re-running load_stage() without process restart
- [ ] Event-log dump flag: print every fact per tick (mission debugging)
