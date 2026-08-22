#pragma once
// tetris/config/levels/garbage_canyon.hpp — LEVEL DEFINITION (tetris::config::GarbageCanyon)
// L3 · Garbage Canyon — Tier 3: the board itself is authored in Lua.
// The generator script (domains/matrix/scripts/garbage_canyon.gen.lua) stamps
// a pre-ruined board at boot via CanyonGen.generate(difficulty, seed) and may
// retune target_lines/time_limit via CanyonGen.get_config(); these values are
// the fallback when no script is loaded (graceful degradation, ARCHITECTURE §4.2).
#include <config/rules.hpp>

namespace tetris::config {

    struct GarbageCanyon {
        static constexpr const char* NAME = "GARBAGE CANYON";

        static Rules make_rules() {
            Rules r;
            r.mode_id      = MODE_GARBAGE_CANYON;
            r.time_limit   = 180.0f;   // excavation sprint clock (script-gated)
            r.target_lines = 20;       // win = excavate 20 lines (script-gated)
            r.target_score = 0;        // unused in excavation mode
            // Camera: wide diorama shot — mesas/torches stay visible around
            // the well while the full board keeps its clearance.
            r.camera.eye     = { 0.0f, 13.5f, -28.0f };
            r.camera.target  = { 0.0f, 10.2f,   0.0f };
            r.camera.fov_deg = 58.0f;
            return r;
        }
    };

} // namespace tetris::config