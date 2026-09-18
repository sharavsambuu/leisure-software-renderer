#!/usr/bin/env bash
set -euo pipefail

# Negative fixture for the Pure Domain Value Library classification gate
# (governance todo G2.1, Constitution II §6.1 amendment):
#   - fixture A: a classified pure library carrying <lib>.command.hpp
#     (stateful machinery in a stateless leaf) must FAIL;
#   - fixture B: a conforming pure library (contract types + pure transforms
#     only) must PASS;
#   - fixture C: a classified library with an EMPTY home must FAIL
#     (vacuous classification);
#   - fixture D: a classified library whose home directory is MISSING must
#     FAIL (classification points nowhere);
#   - fixture E: an EMPTY classification list must FAIL loudly (the gate
#     would silently enforce nothing).
# Fixture trees are minimal fakes — the gate is a text scan, no compile.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
checker="${script_dir}/check_pure_value_libraries.sh"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

# --- fixture A: stateful machinery in a pure library must FAIL -------------
mkdir -p "${tmp}/bad/shs/mylib"
cat > "${tmp}/bad/shs/mylib/mylib.contract.hpp" <<'EOF'
#pragma once
#include <cstdint>
namespace mylib { struct MyValue { std::uint32_t a = 0; }; }
EOF
cat > "${tmp}/bad/shs/mylib/mylib.command.hpp" <<'EOF'
#pragma once
#include <variant>
namespace mylib { using MyCommand = std::variant<std::monostate>; }
EOF
if SHS_PURE_VALUE_LIBRARIES="mylib:mylib" "${checker}" "${tmp}/bad/shs" >/dev/null 2>&1; then
  echo "[pure-value-library-negative] FAIL: stateful machinery (command file) passed the pure-library gate"
  exit 1
fi

# --- fixture B: conforming pure library must PASS ---------------------------
mkdir -p "${tmp}/good/shs/mylib"
cat > "${tmp}/good/shs/mylib/mylib.contract.hpp" <<'EOF'
#pragma once
#include <cstdint>
namespace mylib { struct MyValue { std::uint32_t a = 0; }; }
EOF
cat > "${tmp}/good/shs/mylib/mylib.transforms.hpp" <<'EOF'
#pragma once
#include "shs/mylib/mylib.contract.hpp"
namespace mylib { inline std::uint32_t double_value(std::uint32_t x) { return x * 2u; } }
EOF
if ! SHS_PURE_VALUE_LIBRARIES="mylib:mylib" "${checker}" "${tmp}/good/shs" >/dev/null 2>&1; then
  echo "[pure-value-library-negative] FAIL: conforming pure library rejected by the gate"
  exit 1
fi

# --- fixture C: empty home must FAIL ----------------------------------------
mkdir -p "${tmp}/vac/shs/emptlib"
if SHS_PURE_VALUE_LIBRARIES="emptlib:emptlib" "${checker}" "${tmp}/vac/shs" >/dev/null 2>&1; then
  echo "[pure-value-library-negative] FAIL: empty classified home passed the gate (vacuous classification)"
  exit 1
fi

# --- fixture D: missing home must FAIL --------------------------------------
if SHS_PURE_VALUE_LIBRARIES="ghost:ghost" "${checker}" "${tmp}/good/shs" >/dev/null 2>&1; then
  echo "[pure-value-library-negative] FAIL: classification with a missing home directory passed"
  exit 1
fi

# --- fixture E: empty classification must FAIL loudly -----------------------
if SHS_PURE_VALUE_LIBRARIES=" " "${checker}" "${tmp}/good/shs" >/dev/null 2>&1; then
  echo "[pure-value-library-negative] FAIL: empty classification list passed (gate enforces nothing)"
  exit 1
fi

echo "[pure-value-library-negative] all tests passed"
