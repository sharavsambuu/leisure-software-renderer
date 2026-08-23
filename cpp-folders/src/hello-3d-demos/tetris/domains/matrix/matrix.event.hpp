#pragma once
// tetris/domains/matrix/matrix.event.hpp — RAW FACTS (tetris::matrix)
// Discrete occurrences emitted by the grid rulebook. NO point deltas here —
// scoring is derived by the progression pod from these facts (Rule 8.1).
#include <cstdint>
#include <glm/glm.hpp>

namespace tetris::matrix {

    enum class MatrixEventType : uint8_t {
        PIECE_SPAWNED,
        PIECE_MOVED,
        PIECE_ROTATED,
        PIECE_LOCK_IMPACT,
        HARD_DROP_SLAM,
        SOFT_DROP,
        LINES_CLEARED,
        HOLD_SWAPPED,
        GAME_OVER,
        SPECIAL_LOCKED,  // L4: a Bomb/Laser/Freeze piece reached its lock point;
                         // the grid is untouched — the scripted ruling decides
                         // the mutation (raw fact for the powerups pod)
        GARBAGE_RAINED   // L5: an AddGarbageRowsIntent volley landed (rows
                         // shifted up, garbage filled in from below). Raw fact
                         // for FX/audio; rain cadence lives upstream.
    };

    struct MatrixEvent {
        MatrixEventType type;
        uint8_t         lines_cleared_count = 0;
        uint8_t         cleared_rows[4]{ 0, 0, 0, 0 };
        glm::vec3       world_position{ 0.0f };
        int             cells = 0;   // dropped (hard) / stepped (soft) cell count

        // L3 raw fact: how many Garbage cells were involved (cleared mass for
        // LINES_CLEARED; 1/0 adjacency flag for lock/slam impacts). FX and the
        // dig-feel audio map scale off this — no grid peeking downstream.
        uint8_t         garbage_cells = 0;

        // L4 raw facts for SPECIAL_LOCKED: which special locked and the grid
        // cell of its anchor block (plain values — no snapshot coupling).
        uint8_t         special_type = 0;
        int16_t         lock_x       = 0;
        int16_t         lock_y       = 0;

        // L5 raw fact for GARBAGE_RAINED: how many rows the volley added.
        uint8_t         rain_rows    = 0;
    };

} // namespace tetris::matrix
