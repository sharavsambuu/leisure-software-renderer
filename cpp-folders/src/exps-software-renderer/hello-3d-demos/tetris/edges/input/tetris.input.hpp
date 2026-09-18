#pragma once
// tetris/edges/input/tetris.input.hpp - SDL INPUT BOUNDARY (tetris::input)
//
// STATEFUL INPUT EDGE (Part 6 input-feel fix):
//  - tracks PHYSICAL held state via SDL_KEYDOWN + SDL_KEYUP; OS key-repeat
//    is never trusted (late ~500ms, OS-controlled rate)
//  - horizontal movement: DAS/ARR scheduling at fixed game rates
//      DAS 150 ms initial delay, ARR 40 ms auto-repeat (guideline values)
//    - a fresh press shifts ONCE immediately (tap responsiveness)
//    - last-press-wins when both directions are held
//  - soft drop is a HELD FLAG the reducer reads each tick
//  - discrete actions (rotate/hard drop/hold/restart) are edge-triggered:
//    one intent per physical press, released on KEYUP
//  - all held state releases on window focus loss (alt-tab safety)
//
// Usage per frame:
//   input_edge.begin_frame(dt);
//   InputState in = input_edge.poll(arena);
#include <SDL3/SDL.h>
#include <algorithm>
#include <memory_resource>

#include <domains/matrix/matrix.action.hpp>
#include <domains/session/session.action.hpp>

namespace tetris::input {

    struct FeelParams {
        float das_delay_s  = 0.150f; // delay before auto-shift starts
        float arr_repeat_s = 0.040f; // auto-shift repeat period
    };

    struct InputState {
        bool quit = false;

        std::pmr::vector<matrix::TetrisCommand>   commands;
        std::pmr::vector<session::SessionCommand> session_commands;

        // Continuous held-state reads for the matrix reducer:
        int  move_x         = 0;     // -1/0/+1 (DAS/ARR output this frame)
        bool soft_drop_held = false; // gravity multiplier flag

        explicit InputState(std::pmr::memory_resource* mr)
            : commands(mr), session_commands(mr) {}
    };

    class InputEdge {
    public:
        explicit InputEdge(const FeelParams& params = FeelParams{})
            : p_(params) {}

        void begin_frame(float dt_seconds) { dt_ = dt_seconds; }

        InputState poll(std::pmr::memory_resource* mr, float dt_frame = 0.0f) {
            if (dt_frame > 0.0f) dt_ = dt_frame;
            InputState in(mr);

            // --- SDL event pump -------------------------------------------------
            SDL_Event e;
            while (SDL_PollEvent(&e)) {
                switch (e.type) {
                case SDL_QUIT:
                    in.quit = true;
                    break;

                case SDL_WINDOWEVENT:
                    if (e.window.event == SDL_WINDOWEVENT_FOCUS_LOST)
                        release_all();
                    break;

                case SDL_KEYDOWN: {
                    const auto k = e.key.keysym.sym;
                    if (e.key.repeat != 0) break; // OS repeat never trusted

                    // gameplay: discrete intents latch until KEYUP
                    if (k == SDLK_UP || k == SDLK_x) {
                        if (!l_.rot_cw)   { l_.rot_cw   = true; in.commands.push_back(matrix::RotateCWIntent{}); }
                    } else if (k == SDLK_z) {
                        if (!l_.rot_ccw)  { l_.rot_ccw  = true; in.commands.push_back(matrix::RotateCCWIntent{}); }
                    } else if (k == SDLK_SPACE) {
                        if (!l_.hard)     { l_.hard     = true; in.commands.push_back(matrix::HardDropIntent{}); }
                    } else if (k == SDLK_c || k == SDLK_LSHIFT) {
                        if (!l_.hold)     { l_.hold     = true; in.commands.push_back(matrix::HoldPieceIntent{}); }
                    } else if (k == SDLK_r) {
                        if (!l_.restart)  { l_.restart  = true; in.commands.push_back(matrix::RestartIntent{}); }
                    }

                    // horizontal: direction press -> immediate TAP shift +
                    // reset DAS timers (last-press-wins)
                    if (k == SDLK_LEFT && !h_.left) {
                        h_.left = true;
                        das_t = 0.0f; arr_t = 0.0f; das_shifted = false;
                        dir = -1;
                        in.commands.push_back(matrix::MoveLeftIntent{});
                    } else if (k == SDLK_RIGHT && !h_.right) {
                        h_.right = true;
                        das_t = 0.0f; arr_t = 0.0f; das_shifted = false;
                        dir = +1;
                        in.commands.push_back(matrix::MoveRightIntent{});
                    } else if (k == SDLK_DOWN) {
                        h_.down = true;
                    }

                    // session intents (repeat-guarded latches)
                    session_keydown(k, in);
                    break;
                }

                case SDL_KEYUP: {
                    const auto k = e.key.keysym.sym;
                    if (k == SDLK_LEFT)  h_.left = false;
                    if (k == SDLK_RIGHT) h_.right = false;
                    if (k == SDLK_DOWN)  h_.down = false;
                    if (k == SDLK_UP || k == SDLK_x)          l_.rot_cw  = false;
                    if (k == SDLK_z)                          l_.rot_ccw = false;
                    if (k == SDLK_SPACE)                      l_.hard    = false;
                    if (k == SDLK_c || k == SDLK_LSHIFT)      l_.hold    = false;
                    if (k == SDLK_r)                          l_.restart = false;

                    // direction release: hand control to the other key if held,
                    // otherwise stop moving and clear DAS state
                    if (k == SDLK_LEFT || k == SDLK_RIGHT) {
                        if (h_.left)  { dir = -1; das_t = 0.0f; arr_t = 0.0f; das_shifted = false; }
                        else if (h_.right) { dir = +1; das_t = 0.0f; arr_t = 0.0f; das_shifted = false; }
                        else dir = 0;
                    }
                    session_keyup(k);
                    break;
                }

                default: break;
                }
            }

            // --- DAS/ARR scheduler ----------------------------------------------
            const float dt = dt_frame > 0.0f ? dt_frame : dt_;
            if (dir != 0) {
                das_t += dt;
                if (das_t >= p_.das_delay_s) {
                    arr_t += dt;
                    while (arr_t >= p_.arr_repeat_s) {
                        arr_t -= p_.arr_repeat_s;
                        in.commands.push_back(
                            dir < 0 ? matrix::TetrisCommand(matrix::MoveLeftIntent{})
                                    : matrix::TetrisCommand(matrix::MoveRightIntent{}));
                    }
                    if (!das_shifted) {
                        in.commands.push_back(dir < 0
                            ? matrix::TetrisCommand(matrix::MoveLeftIntent{})
                            : matrix::TetrisCommand(matrix::MoveRightIntent{}));
                        das_shifted = true;
                    }
                }
            }

            in.move_x         = dir;
            in.soft_drop_held = h_.down;
            return in;
        }

        int  current_move_dir()  const { return dir; }
        bool soft_drop_held()    const { return h_.down; }

    private:
        struct Held { bool left=false, right=false, down=false; } h_{};
        struct Latch {
            bool rot_cw=false, rot_ccw=false, hard=false, hold=false,
                 restart=false, nav_up=false, nav_down=false, nav_left=false,
                 nav_right=false, confirm=false, back=false, pause=false,
                 sound=false;
        } l_{};

        int   dir = 0;           // -1 left / +1 right / 0 none
        float das_t = 0.0f;
        float arr_t = 0.0f;
        bool  das_shifted = false;
        float dt_ = 1.0f / 60.0f;
        FeelParams p_;

        void release_all() {
            h_ = {};
            l_ = {};
            dir = 0; das_t = arr_t = 0.0f; das_shifted = false;
        }

        void session_keydown(SDL_Keycode k, InputState& in) {
            switch (k) {
            case SDLK_UP:    if (!l_.nav_up)    { l_.nav_up    = true; in.session_commands.push_back(session::NavUpIntent{}); } break;
            case SDLK_DOWN:  if (!l_.nav_down)  { l_.nav_down  = true; in.session_commands.push_back(session::NavDownIntent{}); } break;
            case SDLK_a:     if (!l_.nav_left)  { l_.nav_left  = true; in.session_commands.push_back(session::NavLeftIntent{}); } break;
            case SDLK_d:     if (!l_.nav_right) { l_.nav_right = true; in.session_commands.push_back(session::NavRightIntent{}); } break;
            case SDLK_RETURN: case SDLK_KP_ENTER:
                if (!l_.confirm) { l_.confirm = true; in.session_commands.push_back(session::ConfirmIntent{}); } break;
            case SDLK_ESCAPE: if (!l_.back)   { l_.back   = true; in.session_commands.push_back(session::BackIntent{}); } break;
            case SDLK_p:     if (!l_.pause)   { l_.pause   = true; in.session_commands.push_back(session::TogglePauseIntent{}); } break;
            case SDLK_m:     if (!l_.sound)   { l_.sound   = true; in.session_commands.push_back(session::ToggleSoundIntent{}); } break;
            default: break;
            }
        }
        void session_keyup(SDL_Keycode k) {
            switch (k) {
            case SDLK_UP:    l_.nav_up    = false; break;
            case SDLK_DOWN:  l_.nav_down  = false; break;
            case SDLK_a:     l_.nav_left  = false; break;
            case SDLK_d:     l_.nav_right = false; break;
            case SDLK_RETURN: case SDLK_KP_ENTER: l_.confirm = false; break;
            case SDLK_ESCAPE: l_.back   = false; break;
            case SDLK_p:      l_.pause  = false; break;
            case SDLK_m:      l_.sound  = false; break;
            default: break;
            }
        }
    };

} // namespace tetris::input
