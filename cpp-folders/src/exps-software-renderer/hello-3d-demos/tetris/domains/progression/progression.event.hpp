#pragma once
// tetris/domains/progression/progression.event.hpp — DERIVED OCCURRENCES (tetris::progression)
// Downstream consumers (spatial_fx, ui edge, audio map) listen to these;
// nobody polls progression state directly (Rule 8.1).
#include <cstdint>

namespace tetris::progression {

    enum class ProgressionEventType : uint8_t {
        SCORE_CHANGED,
        COMBO_STREAK,         // emitted at combo_count >= 2 (FX/GUI feed)
        LEVEL_UP,
        OBJECTIVE_COMPLETED,
        CLOCK_TICK,           // blitz: crossed a 30-second boundary
        TIME_BONUS,           // blitz: rule source granted bonus seconds
        TIME_UP               // blitz: clock expired
    };

    struct ProgressionEvent {
        ProgressionEventType type;
        int   score_delta = 0;
        int   combo       = 0;
        int   new_level   = 0;
        float seconds     = 0.0f;   // TIME_BONUS payload
    };

} // namespace tetris::progression