#pragma once

// snake action: folds command tokens into a movement frame. Pure function, no side effects.
#include <span>
#include "snake.contract.hpp"

namespace snake::matrix {

    // Fold intent tokens (left/right/up/down) into a single grid-aligned movement delta.
    inline auto reduce_snake_commands(std::span<const SnakeCommand> commands) -> glm::vec2 {
        int dx = 0, dy = 0;
        for (const auto& cmd : commands) {
            switch (cmd.type) {
                case SnakeCommandType::LEFT:   dx -= 1; break;
                case SnakeCommandType::RIGHT:  dx += 1; break;
                case SnakeCommandType::UP:     dy -= 1; break;
                case SnakeCommandType::DOWN:   dy += 1; break;
            }
        }
        return { static_cast<float>(dx), static_cast<float>(dy) };
    }

} // namespace snake::matrix
