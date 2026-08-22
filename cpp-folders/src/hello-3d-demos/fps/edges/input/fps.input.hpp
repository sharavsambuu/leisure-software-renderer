#pragma once

// ============================================================================
// fps/edges/input/fps.input.hpp — SDL INPUT BOUNDARY (fps::input)
// Main polls SDL into InputState; reduce_input translates it into a frame's
// UserCommands (applying sensitivity + key-look rates from Difficulty).
// This is the ONLY pod that knows about SDL input APIs.
// ============================================================================

#include <SDL2/SDL.h>

#include <glm/glm.hpp>
#include <memory_resource>

#include <domains/matrix/fps.contract.hpp>
#include <config/difficulty.hpp>

namespace fps::input {

    struct InputState {
        bool      quit_requested = false;
        bool      fire_pressed   = false;
        bool      jump_pressed   = false;
        bool      reset_pressed  = false;
        float     mouse_dx       = 0.0f;
        float     mouse_dy       = 0.0f;
        float     key_yaw        = 0.0f;   // -1..1 raw axis; dt scaling applied in reduce_input
        float     key_pitch      = 0.0f;   // -1..1 raw axis
        glm::vec3 move_axis{ 0.0f };       // x=strafe, z=forward, each -1..1
    };

    // Poll SDL events + device state into a fresh InputState for this frame.
    inline InputState poll_input() {
        InputState in{};

        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) in.quit_requested = true;

            if (e.type == SDL_KEYDOWN) {
                switch (e.key.keysym.sym) {
                case SDLK_ESCAPE: in.quit_requested = true; break;
                case SDLK_r:      in.reset_pressed  = true; break;
                case SDLK_SPACE:  in.jump_pressed   = true; break;
                case SDLK_f:
                case SDLK_LCTRL:
                case SDLK_RCTRL:
                case SDLK_RETURN: in.fire_pressed   = true; break;
                default: break;
                }
            }

            if (e.type == SDL_MOUSEBUTTONDOWN) {
                SDL_SetRelativeMouseMode(SDL_TRUE);
                if (e.button.button == SDL_BUTTON_LEFT) in.fire_pressed = true;
            }
        }

        int mdx = 0, mdy = 0;
        SDL_GetRelativeMouseState(&mdx, &mdy);
        in.mouse_dx = static_cast<float>(mdx);
        in.mouse_dy = static_cast<float>(mdy);

        const Uint8* keys = SDL_GetKeyboardState(nullptr);
        if (keys[SDL_SCANCODE_LEFT])  in.key_yaw   -= 1.0f;
        if (keys[SDL_SCANCODE_RIGHT]) in.key_yaw   += 1.0f;
        if (keys[SDL_SCANCODE_UP])    in.key_pitch += 1.0f;
        if (keys[SDL_SCANCODE_DOWN])  in.key_pitch -= 1.0f;

        if (keys[SDL_SCANCODE_W]) in.move_axis.z += 1.0f;
        if (keys[SDL_SCANCODE_S]) in.move_axis.z -= 1.0f;
        if (keys[SDL_SCANCODE_D]) in.move_axis.x += 1.0f;
        if (keys[SDL_SCANCODE_A]) in.move_axis.x -= 1.0f;

        return in;
    }

    // Translate raw input into UserCommands (pure given the InputState).
    inline std::pmr::vector<matrix::UserCommand> reduce_input(
        const InputState&             in,
        const config::Difficulty&     diff,
        float                         dt,
        std::pmr::memory_resource*    frame_arena
    ) {
        std::pmr::vector<matrix::UserCommand> commands(frame_arena);

        if (in.reset_pressed) commands.push_back(matrix::ResetIntent{});
        if (in.jump_pressed)  commands.push_back(matrix::JumpIntent{});
        if (in.fire_pressed)  commands.push_back(matrix::FireIntent{});

        if (in.mouse_dx != 0.0f || in.mouse_dy != 0.0f) {
            commands.push_back(matrix::LookIntent{
                in.mouse_dx * diff.mouse_sensitivity,
                in.mouse_dy * diff.mouse_sensitivity
            });
        }

        if (in.key_yaw != 0.0f || in.key_pitch != 0.0f) {
            commands.push_back(matrix::LookIntent{
                in.key_yaw * diff.key_look_rate_yaw * dt,
                -(in.key_pitch * diff.key_look_rate_pitch * dt)
            });
        }

        if (glm::length(in.move_axis) > 0.01f) {
            commands.push_back(matrix::MoveIntent{ in.move_axis });
        }

        return commands;
    }

} // namespace fps::input