#!/usr/bin/env bash
# P2c: script purity - every domains/**/scripts/*.lua must be free of
# nondeterministic / side-effecting globals (comments stripped first).
FAIL=0
for f in "$(dirname "$0")/../domains"/*/scripts/*.lua; do
  [ -e "$f" ] || continue
  # strip Lua comment lines, then look for forbidden constructs
  BAD=$(grep -v '^[[:space:]]*--' "$f" | grep -E '\bos\.|\bio\.|math\.random|print[[:space:]]*\(' )
  if [ -n "$BAD" ]; then
    echo "FAIL $f"
    echo "$BAD" | head -3
    FAIL=1
  else
    echo "PASS $(basename "$f")"
  fi
done
exit $FAIL
