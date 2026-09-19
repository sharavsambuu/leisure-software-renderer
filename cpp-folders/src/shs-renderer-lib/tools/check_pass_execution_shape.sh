#!/usr/bin/env bash
set -euo pipefail

# Pass-execution shape gate (ROP-2.4, owner ruling 2026-09-18).
#
# Constitution II §8 bans the `(payload, bool valid)` shape. The pass rim used
# to be exactly that — a `bool executed` beside the output bits — and it
# collapsed three distinct facts (invalid request, unmet prerequisites,
# deliberate decline) into one `false`. The ruled shape (R2, option A) stores a
# closed `PassOutcome` and exposes `executed()` as a derived query, so a refusal
# cannot be expressed without naming its fact.
#
#   R1 no stored `bool executed` member in `PassExecutionResult`; the derived
#      query `executed()` is the legal form and must exist;
#   R2 no `bool valid` member inside `PassExecutionResult` (§8's clause);
#   R3 the reason-free `not_executed()` factory must not come back — every
#      refusal names its fact;
#   R4 `PassOutcome` stays a closed, string-free `enum class` carrying exactly
#      the three refusal facts plus `Executed` (P5: adding a case is a
#      deliberate, reviewed act, never a silent one).
#
# Comments are stripped before scanning (lesson 9.10: a gate scans code, not
# prose — the header's banner legitimately names `not_executed()` as history).
# Scan root defaults to the lib's include tree; overridable for the negative
# fixture (same pattern as check_gateway_rails.sh).

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lib_root="$(cd "${script_dir}/.." && pwd)"
scan_root="${PASS_SHAPE_SCAN_ROOT:-${lib_root}/include/shs}"

strip_comments() {
  perl -0777 -pe 's{/\*.*?\*/}{}gs; s{//[^\n]*}{}g' "$1"
}

failed=0

header="${scan_root}/renderpath/execution/render_pass.hpp"
if [[ ! -f "${header}" ]]; then
  echo "[pass-shape] FAIL: ${header} not found under ${scan_root}"
  exit 1
fi

code="$(strip_comments "${header}")"

# --- R1/R2: the result struct must not store a validity bit ------------------
result_body="$(printf '%s\n' "${code}" | perl -0777 -ne 'print $1 if /struct\s+PassExecutionResult\s*\{(.*?)\n\s*\};/s')"
if [[ -z "${result_body}" ]]; then
  echo "[pass-shape] FAIL: struct PassExecutionResult not found in ${header}"
  exit 1
fi

stored_executed="$(printf '%s\n' "${result_body}" | grep -nE '\bbool[[:space:]]+executed[[:space:]]*[=;]' || true)"
if [[ -n "${stored_executed}" ]]; then
  echo "[pass-shape] FAIL: R1 — PassExecutionResult stores a 'bool executed' again (Constitution II §8 bans the (payload, bool valid) shape; keep the closed PassOutcome and the derived executed() query)"
  echo "${stored_executed}"
  failed=1
else
  echo "[pass-shape] OK: R1 no stored 'bool executed' in PassExecutionResult"
fi

if ! printf '%s\n' "${code}" | grep -qE 'constexpr[[:space:]]+bool[[:space:]]+executed\(\)'; then
  echo "[pass-shape] FAIL: R1 — the derived query 'constexpr bool executed() const' is missing; the outcome must stay queryable"
  failed=1
fi

stored_valid="$(printf '%s\n' "${result_body}" | grep -nE '\bbool[[:space:]]+valid[[:space:]]*[=;]' || true)"
if [[ -n "${stored_valid}" ]]; then
  echo "[pass-shape] FAIL: R2 — PassExecutionResult carries a 'bool valid' (§8: a payload must not sit beside its own validity bit)"
  echo "${stored_valid}"
  failed=1
else
  echo "[pass-shape] OK: R2 no 'bool valid' in PassExecutionResult"
fi

# --- R3: no reason-free refusal factory anywhere in the scanned tree ---------
refusal_hits="$(while IFS= read -r f; do
  strip_comments "${f}" | grep -n 'not_executed' | sed "s|^|${f#${scan_root}/}:|" || true
done < <(find "${scan_root}" -name '*.hpp' | sort))"
refusal_hits="$(printf '%s\n' "${refusal_hits}" | sed '/^$/d')"
if [[ -n "${refusal_hits}" ]]; then
  echo "[pass-shape] FAIL: R3 — 'not_executed()' is back; a refusal must name its fact (invalid_request / prerequisites_unmet / declined)"
  echo "${refusal_hits}"
  failed=1
else
  echo "[pass-shape] OK: R3 every pass refusal names its fact (no not_executed())"
fi

# --- R4: the closed vocabulary stays closed and string-free ------------------
enum_body="$(printf '%s\n' "${code}" | perl -0777 -ne 'print $1 if /enum\s+class\s+PassOutcome[^{]*\{(.*?)\}/s')"
if [[ -z "${enum_body}" ]]; then
  echo "[pass-shape] FAIL: R4 — 'enum class PassOutcome' not found; the outcome vocabulary must stay a closed scoped enum"
  failed=1
else
  r4_failed=0
  for case_name in InvalidRequest PrerequisitesUnmet Declined Executed; do
    if ! printf '%s\n' "${enum_body}" | grep -qE "\b${case_name}\b"; then
      echo "[pass-shape] FAIL: R4 — PassOutcome lost the '${case_name}' case"
      r4_failed=1
    fi
  done
  if printf '%s\n' "${enum_body}" | grep -qE 'std::string|const char\*'; then
    echo "[pass-shape] FAIL: R4 — PassOutcome carries a string (§3: a closed error/outcome vocabulary stays string-free)"
    r4_failed=1
  fi
  if [[ "${r4_failed}" -eq 0 ]]; then
    echo "[pass-shape] OK: R4 PassOutcome is a closed string-free enum class with all four facts"
  else
    failed=1
  fi
fi

if [[ "${failed}" -ne 0 ]]; then
  echo "[pass-shape] pass-execution shape violations detected"
  exit 1
fi

echo "[pass-shape] all checks passed"
