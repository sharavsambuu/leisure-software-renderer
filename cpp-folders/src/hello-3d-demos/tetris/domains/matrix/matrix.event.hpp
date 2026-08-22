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
        GAME_OVER
    };

    struct MatrixEvent {
        MatrixEventType type;
        uint8_t         lines_cleared_count = 0;
        uint8_t         cleared_rows[4]{ 0, 0, 0, 0 };
        glm::vec3       world_position{ 0.0f };
        int             cells = 0;   // dropped (hard) / stepped (soft) cell count
    };

} // namespace tetris::matrix
