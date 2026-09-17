#!/usr/bin/env bash
set -euo pipefail

# Gate 8 — guardrail placement by file role (Constitution II Rule 17 as
# amended; dvo_semantics_enforcement_proposal.md P1). Semantics: invariants
# live in the *type*, edge law lives at the *seam*.
#
#   SHS_PRE / SHS_POST       — edge law; only *.gateway.hpp / *.contract.hpp
#                              (the seam files).
#   SHS_CONTRACT_ASSERT      — value invariants; only *.contract.hpp,
#                              *.command.hpp, *.event.hpp, and explicit
#                              pure-leaf-value allowlist entries below.
# Any other macro use under the scan root is a gate failure. The bridge header
# itself (core/contract_guardrails.hpp) defines the macros and is exempt — the
# law governs use sites, not definitions. Tests/ and non-installed trees are
# out of scope (the gate is a scan of the installed tree root only).
#
# Overridable scan root for negative fixtures:
#   SHS_PLACEMENT_SCAN_ROOT=<dir> tools/check_contract_placement.sh

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lib_root="$(cd "${script_dir}/.." && pwd)"
scan_root="${SHS_PLACEMENT_SCAN_ROOT:-${lib_root}/include/shs}"

# Pure leaf value headers (P1: invariants of the value itself live beside the
# value). Each entry must be pure-tier computation (no IO, no platform), named
# here with its justification:
leaf_value_allowlist=(
  # Paths are scan-root relative (scan root include/shs, so no shs/ prefix).
  # renderpath/planning/render_path_compiler.hpp — pure planner compiler
  # (pod_purity_dirs tier): states the compiled-plan value invariants
  # (C2.2 technique-mode transition table).
  "renderpath/planning/render_path_compiler.hpp"
)

if [[ ! -d "${scan_root}" ]]; then
  echo "[contract-placement] FAIL: scan root does not exist: ${scan_root}"
  exit 1
fi

failed=0
scanned=0
while IFS= read -r f; do
  rel="${f#"${scan_root}/"}"
  # Bridge exemption: the macro definition site is not an annotation site.
  [[ "${rel}" == "core/contract_guardrails.hpp" ]] && continue
  scanned=$((scanned + 1))
  edge_ok=0
  value_ok=0
  case "${rel}" in
    *.gateway.hpp)                  edge_ok=1 ;;
    *.contract.hpp)                 edge_ok=1; value_ok=1 ;;
    *.command.hpp | *.event.hpp)    value_ok=1 ;;
  esac
  for _a in "${leaf_value_allowlist[@]}"; do
    [[ "${rel}" == "${_a}" ]] && value_ok=1
  done

  edge_hits="$(grep -nE '\bSHS_(PRE|POST)[[:space:]]*\(' "${f}" 2>/dev/null || true)"
  if [[ -n "${edge_hits}" && "${edge_ok}" -ne 1 ]]; then
    echo "[contract-placement] FAIL: SHS_PRE/SHS_POST (edge law) outside *.gateway.hpp / *.contract.hpp"
    echo "${rel}:${edge_hits}"
    failed=1
  fi

  value_hits="$(grep -nE '\bSHS_CONTRACT_ASSERT[[:space:]]*\(' "${f}" 2>/dev/null || true)"
  if [[ -n "${value_hits}" && "${value_ok}" -ne 1 ]]; then
    echo "[contract-placement] FAIL: SHS_CONTRACT_ASSERT (value invariant) outside *.contract/command/event.hpp + pure-leaf allowlist"
    echo "${rel}:${value_hits}"
    failed=1
  fi
done < <(find "${scan_root}" -name '*.hpp' | sort)

if [[ "${scanned}" -eq 0 ]]; then
  echo "[contract-placement] FAIL: no headers found under scan root: ${scan_root}"
  exit 1
fi

if [[ "${failed}" -ne 0 ]]; then
  echo "[contract-placement] placement violations detected (P1: invariants live in the type, edge law lives at the seam)"
  exit 1
fi

echo "[contract-placement] OK: guardrail placement law holds across ${scanned} header(s) (gate 8)"
