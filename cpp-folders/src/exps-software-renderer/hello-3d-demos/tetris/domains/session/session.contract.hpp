#pragma once
// tetris/domains/session/session.contract.hpp — SESSION STATE VOCABULARY
// (tetris::session) Pure value types for the meta game-state machine:
// which screen is up, menu cursors, unlock progress, sound preference,
// and the last-run results latch. No platform I/O, no rendering, no gameplay.
#include <cstdint>

namespace tetris::session {

    enum class Screen : uint8_t {
        TITLE        = 0,   // start menu
        LEVEL_SELECT = 1,   // stage carousel
        PLAYING      = 2,   // live run (main steps the sim pods)
        PAUSED       = 3,   // frozen board + pause menu
        RESULTS      = 4    // end-of-run breakdown + next/retry/title
    };

    struct SessionSnapshot {
        Screen screen          = Screen::TITLE;
        int    cursor          = 0;    // active row on the current screen's menu
        int    stage_cursor    = 0;    // level-select carousel index (0-based)
        int    current_stage   = 0;    // 0-based index of the running/last stage
        int    unlocked_stages = 1;    // stages [0, unlocked_stages) are playable
        int    stage_count     = 2;    // manifest size (main wires this)
        bool   sound_enabled   = true;

        // Last-run results latch (plain values — no cross-pod type coupling).
        bool   run_victory     = false;
        bool   run_time_up     = false;
        int    final_score     = 0;
        int    final_lines     = 0;
        int    final_max_combo = 0;
        float  final_seconds   = 0.0f;

        float  anim_time       = 0.0f; // menu animation clock (drift/pulse)
    };

} // namespace tetris::session