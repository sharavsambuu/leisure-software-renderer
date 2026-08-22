#pragma once
// tetris/edges/input/tetris.input.hpp — SDL INPUT BOUNDARY (tetris::input)
// Sole owner of SDL event polling; emits pure intent tokens.
#include <SDL2/SDL.h>
#include <memory_resource>

#include <domains/matrix/matrix.action.hpp>

namespace tetris::input {

    struct InputState {
        bool quit = false;
        std::pmr::vector<matrix::TetrisCommand> commands;

        explicit InputState(std::pmr::memory_resource* mr) : commands(mr) {}
    };

    static inline InputState poll_input(std::pmr::memory_resource* mr) {
        InputState in(mr);
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) in.quit = true;
            if (e.type == SDL_KEYDOWN) {
                if (e.key.keysym.sym == SDLK_ESCAPE) in.quit = true;
                if (e.key.keysym.sym == SDLK_r)     in.commands.push_back(matrix::RestartIntent{});
                if (e.key.keysym.sym == SDLK_LEFT  || e.key.keysym.sym == SDLK_a) in.commands.push_back(matrix::MoveLeftIntent{});
                if (e.key.keysym.sym == SDLK_RIGHT || e.key.keysym.sym == SDLK_d) in.commands.push_back(matrix::MoveRightIntent{});
                if (e.key.keysym.sym == SDLK_UP    || e.key.keysym.sym == SDLK_w) in.commands.push_back(matrix::RotateCWIntent{});
                if (e.key.keysym.sym == SDLK_z)     in.commands.push_back(matrix::RotateCCWIntent{});
                if (e.key.keysym.sym == SDLK_DOWN  || e.key.keysym.sym == SDLK_s) in.commands.push_back(matrix::SoftDropIntent{});
                if (e.key.keysym.sym == SDLK_SPACE) in.commands.push_back(matrix::HardDropIntent{});
                if (e.key.keysym.sym == SDLK_c     || e.key.keysym.sym == SDLK_LSHIFT) in.commands.push_back(matrix::HoldPieceIntent{});
            }
        }
        return in;
    }

} // namespace tetris::input
