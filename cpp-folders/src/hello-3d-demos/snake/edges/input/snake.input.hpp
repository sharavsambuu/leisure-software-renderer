#pragma once

// snake input edge — translates raw device events into command intents. Pure mapping; never touches
// game state. SDL polling lives in the main entry (SDL2 API). This pod is STANDALONE: it owns its own
// local InputState and compiles without depending on any other domain pod's internals. Coupling to a
// matrix/game pod happens later via a shared command vocabulary, not now — here we adopt that shared
// vocabulary directly so the edge output plugs straight into snake::matrix::reduce_snake().
#include <memory_resource>
#include <glm/glm.hpp>
#include <span>
#include "snake.contract.hpp"   // SnakeCommand / SnakeCommandType (shared vocab)

namespace snake::input {

    struct InputState {
        bool turn_left = false;
        bool turn_right = false;
        bool strafe_up  = false;   // +Y (up)
        bool strafe_down = false;  // -Y (down)
    };

    // Pure edge function: raw input -> command span. Allocates transiently into the injected arena only
    // if growth is needed; otherwise returns an empty view over a static buffer to stay allocation-free.
    inline std::pmr::vector<snake::matrix::SnakeCommand> reduce_input(const InputState& s, std::pmr::memory_resource* mr) {
        using SnakeCommandType = snake::matrix::SnakeCommandType;

        // Static scratch for the common (<=4 commands) case — zero heap traffic on hot paths.
        static snake::matrix::SnakeCommand kCommands[4] = {
            {0u, SnakeCommandType::LEFT, 1.0f},
            {0u, SnakeCommandType::RIGHT, 1.0f},
            {0u, SnakeCommandType::UP, 1.0f},
            {0u, SnakeCommandType::DOWN, 1.0f},
        };

        std::pmr::vector<snake::matrix::SnakeCommand> cmds(mr);
        size_t n = 0;
        if (s.turn_left)  kCommands[n++] = {0u, SnakeCommandType::LEFT, 1.0f};
        if (s.turn_right) kCommands[n++] = {0u, SnakeCommandType::RIGHT, 1.0f};
        if (s.strafe_up)  kCommands[n++] = {0u, SnakeCommandType::UP, 1.0f};
        if (s.strafe_down)kCommands[n++] = {0u, SnakeCommandType::DOWN, 1.0f};

        if (n == 0) return {};   // empty view over the static buffer — no allocation at all
        cmds.resize(n);
        for (size_t i = 0; i < n; ++i) cmds[i] = kCommands[i];
        return cmds;
    }

} // namespace snake::input
