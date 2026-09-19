#!/usr/bin/env bash
set -euo pipefail

# Negative fixture for the claimed-primitive gate (ROP-1.3):
#   - an unclaimed core/ primitive with no recorded disposition FAILs (R1);
#   - an allow-list entry whose header is gone FAILs (R2, stale record);
#   - an allow-list entry with an empty disposition FAILs (R3);
#   - a claimed primitive PASSES, and an allow-listed unclaimed primitive
#     PASSES while still being reported as KNOWN (never silent).

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
checker="${script_dir}/check_claimed_primitives.py"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

# --- fixture A: R1 — unclaimed and unrecorded → FAIL --------------------------
mkdir -p "${tmp}/a/include/shs/core" "${tmp}/a/include/shs/demo"
cat > "${tmp}/a/include/shs/core/orphan.hpp" <<'EOF'
#pragma once
namespace shs::core { inline int orphan() { return 0; } }
EOF
echo '{"entries":[]}' > "${tmp}/a/empty.json"
if CLAIMED_PRIMITIVES_LIB_ROOT="${tmp}/a" CLAIMED_PRIMITIVES_ALLOWLIST="${tmp}/a/empty.json" \
   "${checker}" >/dev/null 2>&1; then
  echo "[claimed-primitives-negative] FAIL: unclaimed primitive passed R1"
  exit 1
fi

# --- fixture B: R3 — allow-listed with an empty disposition → FAIL -----------
mkdir -p "${tmp}/b/include/shs/core"
cat > "${tmp}/b/include/shs/core/orphan.hpp" <<'EOF'
#pragma once
namespace shs::core { inline int orphan() { return 0; } }
EOF
cat > "${tmp}/b/allow.json" <<'EOF'
{"entries":[{"header":"shs/core/orphan.hpp","disposition":"   "}]}
EOF
if CLAIMED_PRIMITIVES_LIB_ROOT="${tmp}/b" CLAIMED_PRIMITIVES_ALLOWLIST="${tmp}/b/allow.json" \
   "${checker}" >/dev/null 2>&1; then
  echo "[claimed-primitives-negative] FAIL: empty disposition passed R3"
  exit 1
fi

# --- fixture C: R2 — stale record whose header is gone → FAIL ---------------
mkdir -p "${tmp}/c/include/shs/core"
cat > "${tmp}/c/include/shs/core/claimed.hpp" <<'EOF'
#pragma once
namespace shs::core { inline int claimed() { return 0; } }
EOF
cat > "${tmp}/c/allow.json" <<'EOF'
{"entries":[{"header":"shs/core/vanished.hpp","disposition":"was pending a move that already happened"}]}
EOF
if CLAIMED_PRIMITIVES_LIB_ROOT="${tmp}/c" CLAIMED_PRIMITIVES_ALLOWLIST="${tmp}/c/allow.json" \
   "${checker}" >/dev/null 2>&1; then
  echo "[claimed-primitives-negative] FAIL: stale allow-list entry passed R2"
  exit 1
fi

# --- fixture D: a claimed primitive and a recorded unclaimed one must PASS ---
mkdir -p "${tmp}/d/include/shs/core" "${tmp}/d/include/shs/demo"
cat > "${tmp}/d/include/shs/core/claimed.hpp" <<'EOF'
#pragma once
namespace shs::core { inline int claimed() { return 0; } }
EOF
cat > "${tmp}/d/include/shs/core/recorded.hpp" <<'EOF'
#pragma once
namespace shs::core { inline int recorded() { return 0; } }
EOF
cat > "${tmp}/d/include/shs/demo/user.hpp" <<'EOF'
#pragma once
#include "shs/core/claimed.hpp"
namespace shs::demo { inline int use() { return shs::core::claimed(); } }
EOF
cat > "${tmp}/d/allow.json" <<'EOF'
{"entries":[{"header":"shs/core/recorded.hpp","disposition":"pending the app-context proposal that owns it"}]}
EOF
out="$(CLAIMED_PRIMITIVES_LIB_ROOT="${tmp}/d" CLAIMED_PRIMITIVES_ALLOWLIST="${tmp}/d/allow.json" \
       "${checker}" 2>&1)" || {
  echo "[claimed-primitives-negative] FAIL: conformant tree rejected"
  echo "${out}"
  exit 1
}
if ! printf '%s\n' "${out}" | grep -q 'KNOWN: shs/core/recorded.hpp'; then
  echo "[claimed-primitives-negative] FAIL: the recorded primitive was not reported as KNOWN (debt must stay visible)"
  exit 1
fi

echo "[claimed-primitives-negative] all tests passed"
