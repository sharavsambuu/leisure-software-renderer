#!/usr/bin/env bash
set -euo pipefail

# G2.1 (governance review 2026-09-18, P1 / Tension 1): Pure Domain Value
# Library classification gate — Constitution II §6.1 as amended.
#
# Constitution II §6.1 recognizes exactly TWO module classes:
#   1. Stateful DVO Pods  — own applied state; full Core 4 file law
#      (contract/command/gateway/event), policed by the Core 4 gates in
#      check_kdba_boundaries.sh.
#   2. Pure Domain Value Libraries — stateless value leaves (math, generic
#      containers, value utilities): value contract = plain types + pure
#      transform functions; EXEMPT from the monostate command/event/gateway
#      scaffolding. If such a library grows applied state it MUST be
#      reclassified as a Stateful DVO Pod through a §2.2 law amendment.
#
# This gate polices the classification mechanically:
#   - a classified library's home must EXIST and carry at least one .hpp
#     (non-vacuity: an empty classification enforces nothing);
#   - a classified library must NOT carry <lib>.command.hpp / <lib>.event.hpp /
#     <lib>.gateway.hpp — stateful machinery in a pure library is a
#     misclassification and fails the gate (add state => become a pod).
#
# Usage:
#   check_pure_value_libraries.sh <shs-include-root>
# Overrides (negative-test hooks):
#   SHS_PURE_VALUE_LIBRARIES  — space-separated "name:home" pairs
#                               (default: core:core containers:containers memory:memory)

root="${1:?usage: check_pure_value_libraries.sh <shs-include-root>}"
classification="${SHS_PURE_VALUE_LIBRARIES:-core:core containers:containers memory:memory}"

failed=0
verified=0

if [[ -z "${classification//[[:space:]]/}" ]]; then
  echo "[pure-value-library] FAIL: classification list is empty — the gate would enforce nothing"
  exit 1
fi

for pair in ${classification}; do
  name="${pair%%:*}"
  home="${pair##*:}"
  dir="${root}/${home}"
  if [[ ! -d "${dir}" ]]; then
    echo "[pure-value-library] FAIL: classified pure library '${name}' home missing: ${dir}"
    failed=1
    continue
  fi
  n_hpp="$(find "${dir}" -maxdepth 1 -name '*.hpp' | wc -l)"
  if [[ "${n_hpp}" -eq 0 ]]; then
    echo "[pure-value-library] FAIL: pure library '${name}' home ${home}/ carries no headers (vacuous classification)"
    failed=1
    continue
  fi
  for role in command event gateway; do
    if [[ -f "${dir}/${name}.${role}.hpp" ]]; then
      echo "[pure-value-library] FAIL: pure library '${name}' carries ${name}.${role}.hpp — stateful machinery in a stateless leaf; reclassify as a Stateful DVO Pod (Constitution II §6.1 amendment, §2.2 law amendment required)"
      failed=1
    fi
  done
  verified=$((verified + 1))
done

if [[ "${failed}" -eq 0 ]]; then
  echo "[pure-value-library] OK: ${verified} pure domain value libraries conform (contract = plain types + pure transforms; no command/event/gateway scaffolding, §6.1 amendment)"
fi
exit "${failed}"
