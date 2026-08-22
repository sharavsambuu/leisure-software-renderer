#pragma once
// tetris/domains/progression/progression.event.hpp — DERIVED OCCURRENCES (tetris::progression)
#include <cstdint>

namespace tetris::progression {

    enum class ProgressionEventType : uint8_t {
        SCORE_CHANGED,
        COMBO_STREAK,        // reserved for Lua rule layer (Phase: lua.edge)
        LEVEL_UP,
        OBJECTIVE_COMPLETED
    };

    struct ProgressionEvent {
        ProgressionEventType type;
        int score_delta = 0;
        int combo       = 0;
        int new_level   = 0;
    };

} // namespace tetris::progression
