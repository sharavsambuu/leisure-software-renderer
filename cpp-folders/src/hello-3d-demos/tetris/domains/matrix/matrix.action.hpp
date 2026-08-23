#pragma once
// tetris/domains/matrix/matrix.action.hpp — INTENT TOKENS (tetris::matrix)
#include <domains/matrix/matrix.contract.hpp>
#include <span>
#include <type_traits>
#include <variant>

namespace tetris::matrix {

    struct MoveLeftIntent    {};
    struct MoveRightIntent   {};
    struct RotateCWIntent    {};
    struct RotateCCWIntent   {};
    struct SoftDropIntent    {};
    struct HardDropIntent    {};
    struct HoldPieceIntent   {};
    struct RestartIntent     {};

    // L3 injection seam: plain-data initial-board stamp. The generator script
    // decides WHERE blocks go (raw facts only); the schema just applies them.
    // No generation logic lives here — main bridges script output to this.
    struct StampInitialBoardIntent {
        CellGrid cells{};
    };

    // L4 powerup seams (same privilege model as the stamp): the scripted
    // ruling decides WHAT mutates; these commands just carry raw facts.
    struct QueueSpecialIntent {
        uint8_t special_type = 0;   // PieceType of the next pulled piece
    };
    struct ClearCellsIntent {
        static constexpr int MAX_CELLS = 32;
        uint8_t     count = 0;
        glm::ivec2  cells[MAX_CELLS]{};
    };
    struct FreezeGravityIntent {
        float seconds = 0.0f;
    };

    // L5 encounter seam: a garbage-rain volley (encounter overseer cadence).
    // The script decides WHEN/HOW MUCH; this command carries the raw facts —
    // up to 4 rows per volley, each with its hole column (0..GRID_W-1).
    struct AddGarbageRowsIntent {
        static constexpr int MAX_ROWS = 4;
        uint8_t rows   = 0;
        uint8_t hole_x[MAX_ROWS]{ 0, 0, 0, 0 };
    };

    using TetrisCommand = std::variant<
        MoveLeftIntent, MoveRightIntent, RotateCWIntent, RotateCCWIntent,
        SoftDropIntent, HardDropIntent, HoldPieceIntent, RestartIntent,
        StampInitialBoardIntent,
        QueueSpecialIntent, ClearCellsIntent, FreezeGravityIntent,
        AddGarbageRowsIntent
    >;

    struct TetrisCommandFrame {
        int  move_x        = 0;     // -1 (Left), +1 (Right)
        int  rotate_dir    = 0;     // +1 (CW), -1 (CCW)
        bool soft_drop     = false;
        bool hard_drop     = false;
        bool hold_pressed  = false;
        bool reset_pressed = false;
    };

    static inline TetrisCommandFrame reduce_tetris_commands(std::span<const TetrisCommand> commands) {
        TetrisCommandFrame out{};
        for (const auto& cmd : commands) {
            std::visit([&out](auto&& c) {
                using T = std::decay_t<decltype(c)>;
                if constexpr (std::is_same_v<T, MoveLeftIntent>)        out.move_x       -= 1;
                else if constexpr (std::is_same_v<T, MoveRightIntent>)  out.move_x       += 1;
                else if constexpr (std::is_same_v<T, RotateCWIntent>)   out.rotate_dir   += 1;
                else if constexpr (std::is_same_v<T, RotateCCWIntent>)  out.rotate_dir   -= 1;
                else if constexpr (std::is_same_v<T, SoftDropIntent>)   out.soft_drop     = true;
                else if constexpr (std::is_same_v<T, HardDropIntent>)   out.hard_drop     = true;
                else if constexpr (std::is_same_v<T, HoldPieceIntent>)  out.hold_pressed  = true;
                else if constexpr (std::is_same_v<T, RestartIntent>)    out.reset_pressed = true;
                // Payload-carrying intents (StampInitialBoardIntent and the L4
                // powerup trio) are handled directly by reduce_matrix — they
                // are not folded into the frame.
            }, cmd);
        }
        return out;
    }

} // namespace tetris::matrix
