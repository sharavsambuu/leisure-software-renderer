#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

# Optional passthrough flags: ./verify.sh [--stage=N] [--script=<file>]
EXTRA_ARGS=()
for a in "$@"; do EXTRA_ARGS+=("$a"); done

echo "=== binary ==="
BIN=$(find /home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/build_vcpkg -maxdepth 5 -path "*tetris/Hello3DTetris" | head -1)
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

cd /home/sharavsambuu/src/dev/leisure-software-renderer/cpp-folders/src/hello-3d-demos/tetris
echo "=== purity: files under domains/ mentioning SDL (expect none) ==="
grep -rl 'SDL' domains/ || echo NONE
echo "=== purity: score/combo refs under domains/matrix/ (expect none) ==="
grep -rn 'score\|combo' domains/matrix/ || echo NONE
echo "=== purity: raw Lua C-API refs outside edges/lua (comments stripped; expect none) ==="
grep -rnE 'lua_State|luaL_|lua_pcall|lua_push|lua_pop|lua_getglobal|lua_setglobal|luaopen' \
    --include='*.hpp' --include='*.cpp' . | grep -v 'edges/lua/' \
    | sed 's://.*::' \
    | awk -F: '$3 ~ /[^ \t]/ { print }' | grep -E '.' || echo NONE
echo "=== main size ==="
wc -l hello_3d_tetris.cpp