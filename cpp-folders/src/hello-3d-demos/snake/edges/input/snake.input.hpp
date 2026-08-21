#pragma once

// snake input edge — translates raw device events into matrix::SnakeCommand intents. Pure mapping;
// never touches game state. SDL polling lives in the main entry (SDL2 API).
#include <glm/glm.hpp>
#include "../../../domains/matrix/snake.contract.hpp"   // SnakeCommand, SnakeCommandType

namespace snake::input {

    struct InputState {
        bool turn_left = false;
        bool turn_right = false;
        bool strafe_up  = false;   // +Y (up)
        bool strafe_down = false;  // -Y (down)
    };

    inline std::pmr::vector<matrix::SnakeCommand> reduce_input(const InputState& s, std::pmr::memory_resource* mr) {
        using matrix::SnakeCommandType;

        std::pmr::vector<matrix::SnakeCommand> cmds(mr);
        if (s.turn_left)  cmds.push_back({0u, SnakeCommandType::LEFT, 1.0f});
        if (s.turn_right) cmds.push_back({0u, SnakeCommandType::RIGHT, 1.0f});
        if (s.strafe_up)  cmds.push_back({0u, SnakeCommandType::UP, 1.0f});
        if (s.strafe_down)cmds.push_back({0u, SnakeCommandType::DOWN, 1.0f});
        return cmds;
    }

} // namespace snake::input
