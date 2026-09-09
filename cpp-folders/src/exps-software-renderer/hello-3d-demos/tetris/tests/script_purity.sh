#!/usr/bin/env bash
# P2c + G4: script purity - every domains/**/scripts/*.lua AND assets/**/*.lua
# must be free of nondeterministic / side-effecting globals (comments stripped
# first). The sandbox nils these at runtime; this gate catches them at rest so
# a script never ships relying on an escape hatch.
FAIL=0
FORBIDDEN='\bos\.|\bio\.|\brequire\b|\bdofile\b|\bloadstring\b|\bcollectgarbage\b|math\.random|print[[:space:]]*\('
GLOB_DIR="$(dirname "$0")"
for f in "$GLOB_DIR"/../domains/*/scripts/*.lua $(find "$GLOB_DIR/../assets" -name '*.lua' 2>/dev/null | sort); do
  [ -e "$f" ] || continue
  # strip Lua comment lines, then look for forbidden constructs
  BAD=$(grep -v '^[[:space:]]*--' "$f" | grep -E "$FORBIDDEN")
  if [ -n "$BAD" ]; then
    echo "FAIL $f"
    echo "$BAD" | head -3
    FAIL=1
  else
    echo "PASS $(basename "$f")"
  fi
done
exit $FAIL