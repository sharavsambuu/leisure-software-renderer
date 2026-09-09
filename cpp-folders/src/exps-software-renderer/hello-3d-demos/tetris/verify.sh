#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

# Optional passthrough flags: ./verify.sh [--stage=N] [--script=<file>]
EXTRA_ARGS=()
for a in "$@"; do EXTRA_ARGS+=("$a"); done

echo "=== binary ==="
BIN="/home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/build_vcpkg/src/hello-3d-demos/tetris/Hello3DTetris"
echo "BIN=$BIN"
[ -x "$BIN" ] || { echo "NO_BINARY"; exit 1; }
ls -la "$BIN"

export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy

echo "=== run A (idle) ==="
"$BIN" --screenshot /tmp/t_idle_a.bmp --frame=45 "${EXTRA_ARGS[@]}" || echo "RUN_A_FAILED"
echo "=== run B (idle repeat) ==="
"$BIN" --screenshot /tmp/t_idle_b.bmp --frame=45 "${EXTRA_ARGS[@]}" || echo "RUN_B_FAILED"
echo "=== run C (autodrive harddrop) ==="
"$BIN" --autodrive-harddrop --screenshot /tmp/t_drop.bmp --frame=45 "${EXTRA_ARGS[@]}" || echo "RUN_C_FAILED"

echo "=== determinism (A vs B) ==="
if cmp -s /tmp/t_idle_a.bmp /tmp/t_idle_b.bmp; then echo DETERMINISM=PASS; else echo DETERMINISM=FAIL; fi

echo "=== behavioral delta (A vs C) ==="
if cmp -s /tmp/t_idle_a.bmp /tmp/t_drop.bmp; then echo DELTA=FAIL; else echo DELTA=PASS; fi

echo "=== smoke: wired script economy override reaches config (stage 2) ==="
"$BIN" --stage=2 --expect-target-score=20000 || echo "SMOKE_TARGET_SCORE_FAILED"

echo "=== blitz stage 2 boots + deterministic WITH scripting ==="
"$BIN" --stage=2 --screenshot /tmp/t_blitz_a.bmp --frame=45 || echo "BLITZ_RUN_FAILED"
"$BIN" --stage=2 --screenshot /tmp/t_blitz_b.bmp --frame=45 || echo "BLITZ_RUN_B_FAILED"
if cmp -s /tmp/t_blitz_a.bmp /tmp/t_blitz_b.bmp; then echo BLITZ_DETERMINISM=PASS; else echo BLITZ_DETERMINISM=FAIL; fi

echo "=== smoke: L3 generator objective override reaches config (stage 3) ==="
"$BIN" --stage=3 --expect-target-lines=20 || echo "SMOKE_TARGET_LINES_FAILED"

echo "=== canyon stage 3 boots + deterministic WITH board generation ==="
"$BIN" --stage=3 --screenshot /tmp/t_canyon_a.bmp --frame=45 || echo "CANYON_RUN_FAILED"
"$BIN" --stage=3 --screenshot /tmp/t_canyon_b.bmp --frame=45 || echo "CANYON_RUN_B_FAILED"
if cmp -s /tmp/t_canyon_a.bmp /tmp/t_canyon_b.bmp; then echo CANYON_DETERMINISM=PASS; else echo CANYON_DETERMINISM=FAIL; fi

echo "=== seed determinism: same seed ⇒ identical board, other seed ⇒ differs ==="
"$BIN" --stage=3 --seed=777 --screenshot /tmp/t_seed777_a.bmp --frame=30 || echo "SEED_A_FAILED"
"$BIN" --stage=3 --seed=777 --screenshot /tmp/t_seed777_b.bmp --frame=30 || echo "SEED_B_FAILED"
"$BIN" --stage=3 --seed=778 --screenshot /tmp/t_seed778.bmp   --frame=30 || echo "SEED_C_FAILED"
if cmp -s /tmp/t_seed777_a.bmp /tmp/t_seed777_b.bmp; then echo SEED_SAME_PASS; else echo SEED_SAME_FAIL; fi
if cmp -s /tmp/t_seed777_a.bmp /tmp/t_seed778.bmp;   then echo SEED_DIFF_FAIL; else echo SEED_DIFF_PASS; fi

echo "=== smoke: L4 cyber mechanics override reaches config (stage 4) ==="
"$BIN" --stage=4 --expect-special-every-n=5 || echo "SMOKE_SPECIAL_EVERY_N_FAILED"

echo "=== cyber stage 4 boots + deterministic WITH mechanics scripting ==="
"$BIN" --stage=4 --screenshot /tmp/t_cyber_a.bmp --frame=45 || echo "CYBER_RUN_FAILED"
"$BIN" --stage=4 --screenshot /tmp/t_cyber_b.bmp --frame=45 || echo "CYBER_RUN_B_FAILED"
if cmp -s /tmp/t_cyber_a.bmp /tmp/t_cyber_b.bmp; then echo CYBER_DETERMINISM=PASS; else echo CYBER_DETERMINISM=FAIL; fi

echo "=== smoke: L5 encounter config override reaches main (stage 5) ==="
"$BIN" --stage=5 --expect-encounter-config=8 || echo "SMOKE_ENCOUNTER_CONFIG_FAILED"

echo "=== encore stage 5 boots + deterministic WITH overseer scripting ==="
"$BIN" --stage=5 --screenshot /tmp/t_encore_a.bmp --frame=45 || echo "ENCORE_RUN_FAILED"
"$BIN" --stage=5 --screenshot /tmp/t_encore_b.bmp --frame=45 || echo "ENCORE_RUN_B_FAILED"
if cmp -s /tmp/t_encore_a.bmp /tmp/t_encore_b.bmp; then echo ENCORE_DETERMINISM=PASS; else echo ENCORE_DETERMINISM=FAIL; fi

echo "=== script purity: generator must be deterministic (no RNG/os/io/print) ==="
sed 's/--.*//' domains/matrix/scripts/*.lua domains/powerups/scripts/*.lua domains/environment/scripts/*.lua \
    | grep -nE 'math\.random|os\.|io\.|print\(' || echo SCRIPT_PURITY=PASS

cd /home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/hello-3d-demos/tetris
echo "=== purity: files under domains/ mentioning SDL (expect none) ==="
grep -rl 'SDL' domains/ || echo NONE
echo "=== purity: score/combo refs under domains/matrix/ C++ code (expect none) ==="
grep -rn 'score\|combo' domains/matrix/ --include='*.hpp' --include='*.cpp' || echo NONE
echo "=== purity: raw Lua C-API refs outside edges/lua (comments stripped; expect none) ==="
grep -rnE 'lua_State|luaL_|lua_pcall|lua_push|lua_pop|lua_getglobal|lua_setglobal|luaopen' \
    --include='*.hpp' --include='*.cpp' . | grep -v 'edges/lua/' \
    | sed 's://.*::' \
    | awk -F: '$3 ~ /[^ \t]/ { print }' | grep -E '.' || echo NONE
echo "=== unit tests (P0 behavioral pins) ==="
TBIN="/home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/build_vcpkg/src/hello-3d-demos/tetris/tetris_reducer_tests"
if [ -x "$TBIN" ]; then
  "$TBIN" && echo "UNIT=PASS" || echo "UNIT=FAIL"
else
  echo "UNIT=SKIPPED (tests not built)"
fi

echo "=== main size ==="
wc -l hello_3d_tetris.cpp