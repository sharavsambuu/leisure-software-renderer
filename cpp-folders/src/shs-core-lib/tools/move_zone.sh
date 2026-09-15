#!/usr/bin/env bash
# P0.5 pod-first tree restructure — zone mover with facade generation.
# Usage: move_zone.sh <zone> <destination>   e.g. move_zone.sh frame domains/frame
# Moves include/shs/<zone> -> include/<destination>, then writes a facade header
# at every old path that #pragma-message's the deprecation and forwards.
set -euo pipefail

zone="$1"
dst="$2"
lib_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$lib_root"

if [[ ! -d "include/shs/${zone}" ]]; then
  echo "move_zone: no such zone include/shs/${zone}" >&2
  exit 1
fi
if [[ -e "include/${dst}" ]]; then
  echo "move_zone: destination include/${dst} already exists — refusing" >&2
  exit 1
fi

mkdir -p "include/$(dirname "${dst}")"
mv "include/shs/${zone}" "include/${dst}"

moved=0
while IFS= read -r rel; do
  rel="${rel#./}"
  old="shs/${zone}/${rel}"
  new="${dst}/${rel}"
  mkdir -p "$(dirname "include/${old}")"
  printf '#pragma message("%s is deprecated: include %s (P0.5 pod-first migration)")\n#include "%s"\n' \
    "${old}" "${new}" "${new}" > "include/${old}"
  moved=$((moved + 1))
done < <(cd "include/${dst}" && find . -name '*.hpp')

echo "move_zone: ${zone} -> ${dst} (${moved} headers, ${moved} facades)"
