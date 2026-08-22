#pragma once
// tetris/domains/progression/progression.contract.hpp — SCORE STATE (tetris::progression)
// Plain-data schema only. Mode fields (L2 Blitz 120) are plain values; the
// blitz clock is stepped by the reducer, urgency flags come from the active
// rule source (Lua script via main-injected hooks, or native C++ fallback).
#include <cstdint>

#include <config/rules.hpp>

namespace tetris::progression {

    using tetris::config::MODE_MARATHON;
    using tetris::config::MODE_BLITZ_120;

    struct ScoreState {
        int  score         = 0;
        int  high_score    = 0;
        int  lines_cleared = 0;
        int  level         = 1;
        int  combo_count   = 0;
        int  max_combo     = 0;
        int  target_score  = 12000;
        bool victory       = false;

        // --- Mode identity + blitz clock (L2) ---
        int   mode_id      = MODE_MARATHON;
        float time_left    = 0.0f;    // seconds remaining; 0 = untimed mode
        bool  clock_danger = false;    // urgency flag from the active rule source
        bool  clock_hurry  = false;
        bool  time_up      = false;   // clock expired (main freezes the run)

        // --- RESULTS breakdown accumulators ---
        int   score_clears     = 0;    // points from line clears
        int   score_drops      = 0;    // points from soft/hard drops
        float time_bonus_total = 0.0f; // bonus seconds collected
    };

} // namespace tetris::progression