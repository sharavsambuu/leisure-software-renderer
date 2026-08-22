#pragma once
// tetris/domains/matrix/matrix.action.hpp — INTENT TOKENS (tetris::matrix)
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

    using TetrisCommand = std::variant<
        MoveLeftIntent, MoveRightIntent, RotateCWIntent, RotateCCWIntent,
        SoftDropIntent, HardDropIntent, HoldPieceIntent, RestartIntent
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
            }, cmd);
        }
        return out;
    }

} // namespace tetris::matrix
