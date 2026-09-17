#!/usr/bin/env bash
set -euo pipefail

# Gateway failure-rail + exhaustiveness gate (W-C owner rulings, 2026-09-17).
#
#   P2 (expected-rail exclusivity, Constitution II §8 codification): a
#     `*_gateway` rim returns a plain Step value or rides the per-command
#     std::expected rail absorbed into events — never `bool`, integral
#     status codes, or exceptions (`throw`) as failure rails.
#   - internal `inline bool` helper arrows inside a gateway are NOT rim
#     signatures and stay legal (the per-command rail's legs);
#   - `operator==` (Step identity) is not a failure rail.
#
# P5 (closed-variant exhaustiveness): command variants are closed; a gateway
# dispatch must (a) never `default:`-swallow and (b) carry a trailing
# `static_assert` tail per `std::visit` so a NEW alternative fails to compile
# instead of silently doing nothing (monostate is not in these variants).
#
# Scan root is `include/shs` of the lib tree; overridable for the negative
# fixture (same pattern as check_contract_placement.sh, SHS_PLACEMENT_SCAN_ROOT).

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lib_root="$(cd "${script_dir}/.." && pwd)"
scan_root="${GATEWAY_RAILS_SCAN_ROOT:-${lib_root}/include/shs}"

failed=0

gateway_files=()
while IFS= read -r f; do gateway_files+=("${f}"); done < <(find "${scan_root}" -name '*.gateway.hpp' | sort)

if [[ "${#gateway_files[@]}" -eq 0 ]]; then
  echo "[gateway-rails] FAIL: no *.gateway.hpp found under ${scan_root}"
  exit 1
fi

# --- P2: expected-rail exclusivity at the gateway rim ------------------------
# (a) no exceptions as failure rails
throw_hits="$(grep -n '\bthrow\b' "${gateway_files[@]}" 2>/dev/null || true)"
if [[ -n "${throw_hits}" ]]; then
  echo "[gateway-rails] FAIL: throw in a gateway (P2: failures ride the std::expected per-command rail, never exceptions)"
  echo "${throw_hits}"
  failed=1
else
  echo "[gateway-rails] OK: no exceptions as failure rails in gateway seams (P2)"
fi

# (b) no bool / integral status return on the rim function itself
rim_bool_hits="$(grep -nE '\b(bool|int|short|long|unsigned|u?int[0-9]+_t)[[:space:]]+[A-Za-z_][A-Za-z0-9_]*_gateway[[:space:]]*\(' "${gateway_files[@]}" 2>/dev/null || true)"
rim_bool_hits="${rim_bool_hits}
$(grep -n -- '->' "${gateway_files[@]}" 2>/dev/null | grep -- '-> *bool' || true)"
rim_bool_hits="$(printf '%s\n' "${rim_bool_hits}" | sed '/^$/d')"
if [[ -n "${rim_bool_hits}" ]]; then
  echo "[gateway-rails] FAIL: bool/status rail on a gateway rim (P2: return a Step value or std::expected, never bool/status codes)"
  echo "${rim_bool_hits}"
  failed=1
else
  echo "[gateway-rails] OK: gateway rims return Step values / expected rails (P2)"
fi

# --- P5: closed-variant exhaustiveness in gateway dispatch -------------------
# (trailing // comments are stripped first — prose may legitimately mention
# the forbidden token; code may not contain it)
default_hits="$(for f in "${gateway_files[@]}"; do
  sed 's|//.*||' "${f}" | grep -nE '\bdefault[[:space:]]*:' | sed "s|^|${f}:|" || true
done)"
if [[ -n "${default_hits}" ]]; then
  echo "[gateway-rails] FAIL: 'default:' swallow in a gateway dispatch (P5: closed variants are handled exhaustively)"
  echo "${default_hits}"
  failed=1
else
  echo "[gateway-rails] OK: no 'default:' swallow in gateway dispatches (P5)"
fi

for f in "${gateway_files[@]}"; do
  visits="$(grep -c 'std::visit' "${f}" || true)"
  sasserts="$(grep -c 'static_assert' "${f}" || true)"
  if [[ "${visits}" -gt 0 && "${sasserts}" -lt "${visits}" ]]; then
    echo "[gateway-rails] FAIL: ${f#${scan_root}/} — std::visit dispatch without a static_assert exhaustiveness tail (P5: a new alternative must fail to compile, not silently do nothing)"
    failed=1
  fi
done
if [[ "${failed}" -eq 0 ]]; then
  echo "[gateway-rails] OK: every gateway dispatch carries a static_assert exhaustiveness tail (P5)"
fi

if [[ "${failed}" -ne 0 ]]; then
  echo "[gateway-rails] gateway rail violations detected"
  exit 1
fi

echo "[gateway-rails] all checks passed"
