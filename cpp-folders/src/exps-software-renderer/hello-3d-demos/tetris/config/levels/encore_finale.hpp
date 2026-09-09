#pragma once
// tetris/config/levels/encore_finale.hpp — LEVEL DEFINITION (tetris::config::EncoreFinale)
// L5 · Encore Finale — Tier 3: encounter ORCHESTRATION authored in Lua. The
// 4-phase boss show (CALM → RAIN → BLACKOUT → CRESCENDO) is decided by
// Encounter.decide_phase() in domains/environment/scripts/encounter_overseer.lua.
// These values are the native fallback when no script is loaded.
#include <config/rules.hpp>

namespace tetris::config {

    struct EncoreFinale {
        static constexpr const char* NAME = "ENCORE FINALE";

        static Rules make_rules() {
            Rules r;
            r.mode_id         = MODE_ENCORE_FINALE;
            r.time_limit      = 0.0f;   // untimed — the encounter sets the pace
            r.target_lines    = 20;     // win = survive & dig 20 lines through the show
            r.target_score    = 0;      // unused in lines mode
            r.special_every_n = 0;      // no L4 specials here (pure encounter stage)
            r.freeze_seconds  = 0.0f;
            // Camera: wide cinematic diorama shot. The eye sits back and
            // slightly high, aimed at mid-well height so BOTH the board and
            // the floor-level set dressing (crowd rows, pedestal rings)
            // stay inside the vertical frustum at diorama distance.
            r.camera.eye     = {  1.5f, 14.0f, -30.0f };
            r.camera.target  = {  0.0f,  8.5f,   0.0f };
            r.camera.fov_deg = 62.0f;
            return r;
        }
    };

} // namespace tetris::config
