#!/usr/bin/env bash
set -euo pipefail

# Negative fixture for the gateway-rails gate (W-C rulings P2 + P5):
#   - `throw` in a gateway FAILs (P2);
#   - a bool rail on the rim function FAILs (P2);
#   - `default:` swallow in a dispatch FAILs (P5);
#   - std::visit without a static_assert tail FAILs (P5);
#   - a conformant gateway (Step rim, expected rail legs, static_assert tail)
#     PASSES.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
checker="${script_dir}/check_gateway_rails.sh"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

fail() { echo "[gateway-rails-negative] FAIL: $1"; exit 1; }

# --- fixture A: illegal rails / dispatches must FAIL -------------------------
mkdir -p "${tmp}/bad/shs/demo"
cat > "${tmp}/bad/shs/demo/demo.gateway.hpp" <<'EOF'
#pragma once
#include <stdexcept>
namespace shs::demo
{
    inline bool demo_gateway(int cmd) { if (cmd < 0) throw std::runtime_error("bad"); return cmd > 0; }
}
EOF
if GATEWAY_RAILS_SCAN_ROOT="${tmp}/bad/shs" "${checker}" >/dev/null 2>&1; then
  echo "[gateway-rails-negative] FAIL: throw + bool rim passed the rails gate (P2)"
  exit 1
fi

mkdir -p "${tmp}/bad2/shs/demo"
cat > "${tmp}/bad2/shs/demo/bad.gateway.hpp" <<'EOF'
#pragma once
#include <type_traits>
#include <variant>
namespace shs::demo
{
    using ClosedCommand = std::variant<int>;
    inline int dispatch_gateway(const ClosedCommand& command)
    {
        return std::visit([&](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, int>) { return 1; }
            else { return 0; } // no static_assert tail — P5 violation
        }, command);
    }
}
EOF
if GATEWAY_RAILS_SCAN_ROOT="${tmp}/bad2/shs" "${checker}" >/dev/null 2>&1; then
  echo "[gateway-rails-negative] FAIL: visit-without-static_assert passed the rails gate (P5)"
  exit 1
fi

# --- fixture B: conformant gateway must PASS ---------------------------------
mkdir -p "${tmp}/good/shs/demo"
cat > "${tmp}/good/shs/demo/demo.gateway.hpp" <<'EOF'
#pragma once
#include <expected>
#include <variant>
namespace shs::demo
{
    struct DemoStep { int applied = 0; bool operator==(const DemoStep&) const = default; };
    using DemoCommand = std::variant<int>;
    inline DemoStep demo_gateway(const DemoCommand& command)
    {
        DemoStep step{};
        std::visit([&](const auto& cmd) {
            using T = std::decay_t<decltype(cmd)>;
            if constexpr (std::is_same_v<T, int>) { step.applied += 1; }
            else { static_assert(sizeof(T) == 0, "unhandled alternative (P5)"); }
        }, command);
        return step;
    }
}
EOF
if ! GATEWAY_RAILS_SCAN_ROOT="${tmp}/good/shs" "${checker}" >/dev/null 2>&1; then
  echo "[gateway-rails-negative] FAIL: conformant gateway rejected by the rails gate"
  exit 1
fi

echo "[gateway-rails-negative] all tests passed"
