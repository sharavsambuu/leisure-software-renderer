#pragma once
// tetris/config/levels/blitz_120.hpp — LEVEL DEFINITION (tetris::config::Blitz120)
// L2 · Blitz 120 — Tier 2 hybrid: C++ engine + Lua-authored economy.
// The wired script (domains/progression/scripts/blitz_mode.lua) may retune
// target/time at boot via BlitzRules.get_config(); these values are the
// fallback when no script is loaded (graceful degradation, ARCHITECTURE §4.2).
#include <config/rules.hpp>

namespace tetris::config {

    struct Blitz120 {
        static constexpr const char* NAME = "BLITZ 120";

        static Rules make_rules() {
            Rules r;
            r.mode_id      = MODE_BLITZ_120;
            r.time_limit   = 120.0f;   // 2-minute sprint
            r.target_score = 20000;    // aggressive economy (script-gated)
            return r;
        }
    };

} // namespace tetris::config