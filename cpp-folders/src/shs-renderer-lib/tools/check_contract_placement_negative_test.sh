#!/usr/bin/env bash
set -euo pipefail

# Negative fixture for gate 8 (P1 placement law, C-W-A DoD):
#   - a macro in a math leaf (SHS_CONTRACT_ASSERT in shs/math/*.hpp) FAILs;
#   - edge macros in a non-seam header (SHS_PRE in *.intent-ish file) FAIL;
#   - the same macros in their legal homes (gateway/contract/command/event)
#     PASS.
# Fixture trees are minimal fakes — the gate is a text scan, no compile.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
checker="${script_dir}/check_contract_placement.sh"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

# --- fixture A: illegal placements must FAIL -------------------------------
mkdir -p "${tmp}/bad/shs/math" "${tmp}/bad/shs/renderpath"
cat > "${tmp}/bad/shs/math/math_value.hpp" <<'EOF'
#pragma once
#include "shs/core/contract_guardrails.hpp"
inline int math_leaf_negate(int x) { SHS_CONTRACT_ASSERT(x > 0); return -x; }
EOF
cat > "${tmp}/bad/shs/renderpath/renderpath.intent.hpp" <<'EOF'
#pragma once
#include "shs/core/contract_guardrails.hpp"
inline bool edge_outside_seam(bool ok) { SHS_PRE(ok); return ok; }
EOF
if SHS_PLACEMENT_SCAN_ROOT="${tmp}/bad/shs" "${checker}" >/dev/null 2>&1; then
  echo "[contract-placement-negative] FAIL: illegal placements passed gate 8 (math leaf / non-seam edge)"
  exit 1
fi

# --- fixture B: legal placements must PASS ----------------------------------
mkdir -p "${tmp}/good/shs/renderpath"
cat > "${tmp}/good/shs/renderpath/renderpath.gateway.hpp" <<'EOF'
#pragma once
#include "shs/core/contract_guardrails.hpp"
inline int seam_edge(int x) { SHS_PRE(x > 0); SHS_POST(x < 100); return x * 2; }
EOF
cat > "${tmp}/good/shs/renderpath/renderpath.contract.hpp" <<'EOF'
#pragma once
#include "shs/core/contract_guardrails.hpp"
inline int seam_value(int x) { SHS_CONTRACT_ASSERT(x != 42); return x; }
EOF
if ! SHS_PLACEMENT_SCAN_ROOT="${tmp}/good/shs" "${checker}" >/dev/null 2>&1; then
  echo "[contract-placement-negative] FAIL: legal placements rejected by gate 8"
  exit 1
fi

echo "[contract-placement-negative] all tests passed"
