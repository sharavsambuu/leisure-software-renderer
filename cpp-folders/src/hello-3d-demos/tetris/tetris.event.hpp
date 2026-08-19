#pragma once

#include <cstdint>
#include <glm/glm.hpp>

namespace tetris {

    enum class TetrisEventType : uint8_t {
        PIECE_SPAWNED,
        PIECE_MOVED,
        PIECE_ROTATED,
        PIECE_LOCK_IMPACT,
        HARD_DROP_SLAM,
        LINES_CLEARED,
        COMBO_STREAK,
        HOLD_SWAPPED,
        LEVEL_UP,
        GAME_OVER,
        OBJECTIVE_COMPLETED
    };

    struct TetrisEvent {
        TetrisEventType type;
        uint8_t         lines_cleared_count = 0;
        uint8_t         cleared_rows[4]{ 0, 0, 0, 0 };
        glm::vec3       world_position{ 0.0f };
        int             score_delta = 0;
        int             combo = 0;
    };

} // namespace tetris