#pragma once
// tetris/config/levels/cyber_storm.hpp — LEVEL DEFINITION (tetris::config::CyberStorm)
// L4 · Cyber Storm — Tier 3: gameplay MECHANICS authored in Lua. Special pieces
// (Bomb / Laser / Freeze) drop on a scripted cadence; their lock effects are
// decided by CyberRules.on_special_lock() in
// domains/powerups/scripts/cyber_storm.lua. These values are the native
// fallback when no script is loaded (graceful degradation, ARCHITECTURE §4.2).
#include <config/rules.hpp>

namespace tetris::config {

    struct CyberStorm {
        static constexpr const char* NAME = "CYBER STORM";

        static Rules make_rules() {
            Rules r;
            r.mode_id         = MODE_CYBER_STORM;
            r.time_limit      = 0.0f;   // untimed — the storm sets the pace
            r.target_lines    = 24;     // win = 24 lines with special-piece help
            r.target_score    = 0;      // unused in lines mode
            r.special_every_n = 5;      // every 5th spawn is a special (script-gated)
            r.freeze_seconds  = 5.0f;
            // Camera: low dramatic angle — eye below board center looking up,
            // neon horizon bar and floor strips read strongly in frame.
            r.camera.eye     = { 0.0f,  9.8f, -23.5f };
            r.camera.target  = { 0.0f, 11.2f,   0.0f };
            r.camera.fov_deg = 56.0f;
            return r;
        }
    };

} // namespace tetris::config