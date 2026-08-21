#pragma once

#include <cstdint>
#include "matrix.contract.hpp"

namespace tetris {
    namespace matrix {

        // ============================================================================
        // Intent Tokens: Caller signals desired state change to the reducer.
        // Collected from input.edge and passed as std::span<const MatrixCommand>.
        // ============================================================================

        /// Move piece one cell left (with SRS wall kick)
        struct MoveLeftIntent {};

        /// Move piece one cell right (with SRS wall kick)
        struct MoveRightIntent {};

        /// Rotate clockwise (SRS: up to 5-point wall kicks attempted)
        struct RotateCWIntent {};

        /// Rotate counter-clockwise (SRS: up to 5-point wall kicks attempted)
        struct RotateCCWIntent {};

        /// Soft drop: move down one cell immediately
        struct SoftDropIntent {};

        /// Hard drop: instantly lock piece at ghost position
        struct HardDropIntent {};

        /// Hold current active piece, spawn next from queue
        struct HoldPieceIntent {};

        /// Reset game: clear grid and spawn new piece
        struct RestartIntent {};

    } // namespace matrix
} // namespace tetris
