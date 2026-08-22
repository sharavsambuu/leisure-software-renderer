#pragma once
// tetris/edges/input/tetris.input.hpp — SDL INPUT BOUNDARY (tetris::input)
// Sole owner of SDL event polling; emits pure intent tokens for BOTH the
// gameplay pods (matrix commands) and the meta layer (session commands).
#include <SDL2/SDL.h>
#include <memory_resource>

#include <domains/matrix/matrix.action.hpp>
#include <domains/session/session.action.hpp>

namespace tetris::input {

    struct InputState {
        bool quit = false;
        std::pmr::vector<matrix::TetrisCommand>   commands;
        std::pmr::vector<session::SessionCommand> session_commands;

        explicit InputState(std::pmr::memory_resource* mr)
            : commands(mr), session_commands(mr) {}
    };

    static inline InputState poll_input(std::pmr::memory_resource* mr) {
        InputState in(mr);
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) in.quit = true;
            if (e.type == SDL_KEYDOWN) {
                // Menu keys ignore key-repeat (holding a key should not scroll rows).
                const bool repeat = (e.key.repeat != 0);

                if (e.key.keysym.sym == SDLK_r)
                    in.commands.push_back(matrix::RestartIntent{});
                if (e.key.keysym.sym == SDLK_z)
                    in.commands.push_back(matrix::RotateCCWIntent{});
                if (e.key.keysym.sym == SDLK_c     || e.key.keysym.sym == SDLK_LSHIFT)
                    in.commands.push_back(matrix::HoldPieceIntent{});
                if (e.key.keysym.sym == SDLK_LEFT)
                    in.commands.push_back(matrix::MoveLeftIntent{});
                if (e.key.keysym.sym == SDLK_RIGHT)
                    in.commands.push_back(matrix::MoveRightIntent{});
                if (e.key.keysym.sym == SDLK_DOWN)
                    in.commands.push_back(matrix::SoftDropIntent{});
                if (e.key.keysym.sym == SDLK_SPACE)
                    in.commands.push_back(matrix::HardDropIntent{});

                // --- session intents (menus + pause; repeat-guarded) ---
                switch (e.key.keysym.sym) {
                case SDLK_UP:
                    if (!repeat) in.session_commands.push_back(session::NavUpIntent{});
                    break;
                case SDLK_DOWN:
                    if (!repeat) in.session_commands.push_back(session::NavDownIntent{});
                    break;
                case SDLK_a:
                    if (!repeat) in.session_commands.push_back(session::NavLeftIntent{});
                    break;
                case SDLK_d:
                    if (!repeat) in.session_commands.push_back(session::NavRightIntent{});
                    break;
                case SDLK_RETURN:
                case SDLK_KP_ENTER:
                    if (!repeat) in.session_commands.push_back(session::ConfirmIntent{});
                    break;
                case SDLK_ESCAPE:
                    if (!repeat) in.session_commands.push_back(session::BackIntent{});
                    break;
                case SDLK_p:
                    if (!repeat) in.session_commands.push_back(session::TogglePauseIntent{});
                    break;
                case SDLK_m:
                    if (!repeat) in.session_commands.push_back(session::ToggleSoundIntent{});
                    break;
                default: break;
                }
            }
        }
        return in;
    }

} // namespace tetris::input
