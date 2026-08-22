# Tetris Domain-Pod Refactor — STATUS

**Date:** 2026-08-22 · **Branch state:** migration P0–P4 complete, all gates green

## Build

| Item | Result |
|---|---|
| Configure | `cmake .` in `cpp-folders/build_vcpkg` — OK |
| Target | `cmake --build . --target Hello3DTetris -j22` — **0 errors** |
| Binary | `build_vcpkg/src/hello-3d-demos/tetris/Hello3DTetris` (~8.8 MB) |
| Flags | gcc `-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion -O3 -flto -march=native`; MSVC `/GL /LTCG /MP` ready |

## Headless verification (reproducible)

Run: `bash verify.sh` (project root) — uses `SDL_VIDEODRIVER=dummy`.

| Check | Command shape | Result |
|---|---|---|
| Determinism | two idle runs `--screenshot --frame=45`, byte-compare | **PASS** (identical BMPs) |
| Behavioral delta | idle vs `--autodrive-harddrop --frame=45` | **PASS** (frames differ) |
| Pod purity: platform refs under `domains/` | `grep -rl SDL domains/` | **NONE** |
| Pod purity: scoring refs under `domains/matrix/` | `grep -rn 'score\|combo' domains/matrix/` | **NONE** |
| Main edge size | `wc -l hello_3d_tetris.cpp` | **331 lines** (was 833) |

## Definition-of-done checklist

- [x] Only one definition of every type/function (root monolith headers deleted)
- [x] Zero SDL/shs includes under `domains/`
- [x] Zero scoring arithmetic under `domains/matrix/`
- [x] Main < ~350 lines, contains no synth/raster/font code
- [x] Root `tetris.*.hpp` deleted
- [x] Deterministic headless run byte-identical
- [x] Autodrive hard-drop frame differs from idle
- [x] gcc clean; MSVC Release flags wired

## Notes / pitfalls encountered (for future migrations)

1. **CRLF defeated every `$`-anchored sed/awk extraction** from original files.
   Always `sed 's/\r$//'` (or strip `[ \t\r]*$`) before anchored matching.
2. **Ubuntu default awk is mawk**: `\b` word boundaries silently no-op. Use GNU
   `sed -E` for renames, or plain longest-first `gsub` chains.
3. **cmd.exe→wsl.exe inline `$var`s are unreliable**; prefer script files.
4. **Aggregator leaks sibling demo roots globally** (`fps` before `tetris`);
   fixed locally via target-scoped `INCLUDE_DIRECTORIES` on Hello3DTetris with
   the tetris root first. Consider scoping all four demos the same way later.
5. `rand()` debris velocities replaced by seeded xorshift inside `FxState` —
   required for determinism; visual behavior equivalent.