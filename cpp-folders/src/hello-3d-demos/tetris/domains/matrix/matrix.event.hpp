#pragma once

#include "matrix.contract.hpp"

namespace tetris {
    namespace matrix {

        // ============================================================================
        // Discrete Event Types: Immutable records emitted by reduce_matrix().
        // Consumed by progression (scoring) and spatial_fx (particles/camera shake).
        // ============================================================================

        enum class MatrixEventType : uint8_t {
            PIECE_SPAWNED,           // New piece spawned at spawn gate
            PIECE_MOVED,             // Horizontal wall kick or soft move
            PIECE_ROTATED,           // SRS rotation succeeded (with optional kick)
            HOLD_SWAPPED,            // Hold + swap completed
            HARD_DROP_SLAM,          // Instant lock with score delta
            GRAVITY_STEP,            // Piece moved down one row by gravity
            PIECE_LOCK_IMPACT,       // Active piece locked into grid
            LINES_CLEARED,           // Rows cleared — triggers shatter particles, camera shake
            COMBO_STREAK            // Consecutive lines without zeroing
        };

        struct MatrixEvent {
            MatrixEventType type;
            int32_t score_delta = 0;          // For HARD_DROP_SLAM, LINES_CLEARED
            glm::vec3 world_position{0.0f};   // World-space position of impact
            uint8_t lines_cleared_count = 0;
            std::array<int, 4> cleared_rows;  // Row indices (top-to-bottom) for TETRIS FIVE
        };

    } // namespace matrix
} // namespace tetris
