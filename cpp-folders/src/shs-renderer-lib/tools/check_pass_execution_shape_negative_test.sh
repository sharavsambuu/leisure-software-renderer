#!/usr/bin/env bash
set -euo pipefail

# Negative fixture for the pass-execution shape gate (ROP-2.4): a stored
# `bool executed` FAILs (R1), a `bool valid` in the result FAILs (R2), a
# resurrected `not_executed()` FAILs (R3), a non-scoped / string-carrying /
# incomplete `PassOutcome` FAILs (R4), and the ruled shape PASSES.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
checker="${script_dir}/check_pass_execution_shape.sh"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

fixture() { mkdir -p "${tmp}/$1/renderpath/execution"; }

# --- fixture A: R1 — a stored `bool executed` is back → FAIL ------------------
fixture a
cat > "${tmp}/a/renderpath/execution/render_pass.hpp" <<'EOF'
#pragma once
#include <cstdint>
namespace shs { inline namespace renderpath {
enum class PassOutcome : uint8_t { InvalidRequest = 0, PrerequisitesUnmet = 1, Declined = 2, Executed = 3 };
struct PassExecutionResult
{
    bool executed = false;
    PassOutcome outcome = PassOutcome::InvalidRequest;
    constexpr bool executed() const { return outcome == PassOutcome::Executed; }
};
}}
EOF
if PASS_SHAPE_SCAN_ROOT="${tmp}/a" "${checker}" >/dev/null 2>&1; then
  echo "[pass-shape-negative] FAIL: stored bool executed passed R1"
  exit 1
fi

# --- fixture B: R2 — a `bool valid` inside the result → FAIL ------------------
fixture b
cat > "${tmp}/b/renderpath/execution/render_pass.hpp" <<'EOF'
#pragma once
#include <cstdint>
namespace shs { inline namespace renderpath {
enum class PassOutcome : uint8_t { InvalidRequest = 0, PrerequisitesUnmet = 1, Declined = 2, Executed = 3 };
struct PassExecutionResult
{
    PassOutcome outcome = PassOutcome::InvalidRequest;
    bool valid = false;
    constexpr bool executed() const { return outcome == PassOutcome::Executed; }
};
}}
EOF
if PASS_SHAPE_SCAN_ROOT="${tmp}/b" "${checker}" >/dev/null 2>&1; then
  echo "[pass-shape-negative] FAIL: bool valid passed R2"
  exit 1
fi

# --- fixture C: R3 — a reason-free refusal factory returns → FAIL -------------
fixture c
cat > "${tmp}/c/renderpath/execution/render_pass.hpp" <<'EOF'
#pragma once
#include <cstdint>
namespace shs { inline namespace renderpath {
enum class PassOutcome : uint8_t { InvalidRequest = 0, PrerequisitesUnmet = 1, Declined = 2, Executed = 3 };
struct PassExecutionResult
{
    PassOutcome outcome = PassOutcome::InvalidRequest;
    static constexpr PassExecutionResult not_executed() { return PassExecutionResult{}; }
    constexpr bool executed() const { return outcome == PassOutcome::Executed; }
};
}}
EOF
if PASS_SHAPE_SCAN_ROOT="${tmp}/c" "${checker}" >/dev/null 2>&1; then
  echo "[pass-shape-negative] FAIL: not_executed() passed R3"
  exit 1
fi

# --- fixture D: R4 — an unscoped enum → FAIL ---------------------------------
fixture d
cat > "${tmp}/d/renderpath/execution/render_pass.hpp" <<'EOF'
#pragma once
namespace shs { inline namespace renderpath {
enum PassOutcome { InvalidRequest = 0, PrerequisitesUnmet = 1, Declined = 2, Executed = 3 };
struct PassExecutionResult
{
    PassOutcome outcome = InvalidRequest;
    constexpr bool executed() const { return outcome == Executed; }
};
}}
EOF
if PASS_SHAPE_SCAN_ROOT="${tmp}/d" "${checker}" >/dev/null 2>&1; then
  echo "[pass-shape-negative] FAIL: unscoped PassOutcome passed R4"
  exit 1
fi

# --- fixture E: R4 — a closed vocabulary that lost a fact → FAIL --------------
fixture e
cat > "${tmp}/e/renderpath/execution/render_pass.hpp" <<'EOF'
#pragma once
#include <cstdint>
namespace shs { inline namespace renderpath {
enum class PassOutcome : uint8_t { InvalidRequest = 0, PrerequisitesUnmet = 1, Executed = 3 };
struct PassExecutionResult
{
    PassOutcome outcome = PassOutcome::InvalidRequest;
    constexpr bool executed() const { return outcome == PassOutcome::Executed; }
};
}}
EOF
if PASS_SHAPE_SCAN_ROOT="${tmp}/e" "${checker}" >/dev/null 2>&1; then
  echo "[pass-shape-negative] FAIL: PassOutcome missing 'Declined' passed R4"
  exit 1
fi

# --- fixture F: the ruled shape must PASS ------------------------------------
fixture f
cat > "${tmp}/f/renderpath/execution/render_pass.hpp" <<'EOF'
#pragma once
#include <cstdint>
namespace shs { inline namespace renderpath {
enum class PassOutcome : uint8_t
{
    InvalidRequest = 0,
    PrerequisitesUnmet = 1,
    Declined = 2,
    Executed = 3
};
struct PassExecutionResult
{
    PassOutcome outcome = PassOutcome::InvalidRequest;
    bool produced_depth = false;

    static constexpr PassExecutionResult declined() { return PassExecutionResult{PassOutcome::Declined, false}; }
    static constexpr PassExecutionResult executed_no_outputs() { return PassExecutionResult{PassOutcome::Executed, false}; }

    constexpr bool executed() const { return outcome == PassOutcome::Executed; }
};
}}
EOF
cat > "${tmp}/f/renderpath/execution/pass_adapters.hpp" <<'EOF'
#pragma once
// A conformant adapter: every refusal names its fact.
namespace shs { inline namespace renderpath {
inline PassExecutionResult example(bool ready)
{
    if (!ready) return PassExecutionResult::prerequisites_unmet();
    return PassExecutionResult::declined();
}
}}
EOF
if ! PASS_SHAPE_SCAN_ROOT="${tmp}/f" "${checker}" >/dev/null 2>&1; then
  echo "[pass-shape-negative] FAIL: the ruled pass-execution shape was rejected"
  PASS_SHAPE_SCAN_ROOT="${tmp}/f" "${checker}" || true
  exit 1
fi

echo "[pass-shape-negative] all tests passed"
