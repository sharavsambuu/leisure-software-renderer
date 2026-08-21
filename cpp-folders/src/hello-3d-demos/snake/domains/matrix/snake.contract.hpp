#pragma once

// snake contract: defines the core gameplay vocabulary and state machine for the 3D snake demo.
#include <cstdint>
#include <array>
#include <glm/glm.hpp>

namespace snake::matrix {

    // SnakeCommandType: raw input token (key press), one of four grid directions.
    enum class SnakeCommandType : uint8_t { LEFT, RIGHT, UP, DOWN };

    // SnakeCommand: a single queued intent for the current tick.
    struct SnakeCommand {
        uint32_t tick = 0;
        SnakeCommandType type = SnakeCommandType::NONE;   // NONE is invalid (guarded by input edge)
        float strength = 1.0f;
    };

    // BodySoA: snake body as a structure-of-arrays over grid cells (z == 0). Allocation-free growth.
    struct BodySoA {
        std::pmr::vector<glm::vec3> position;   // each entry is a world-space cell center (x,y,z)
    };

    // FoodState: live food position on the arena grid (valid when pos != {-1,-1}).
    struct FoodState {
        glm::ivec2 pos{ -1, -1 };               // empty until first spawn
    };

    // SnakeSnapshot: full game state at one tick — head + body + direction + food.
    struct SnakeSnapshot {
        glm::ivec2 head_pos{ 0, 0 };            // grid cell of the head
        glm::vec2  head_dir{ 1, 0 };            // unit facing (grid-aligned)
        FoodState  food;                        // live food position
        BodySoA    body;                        // includes the head as position[0]
    };

    // SnakeEventType: discrete outcomes emitted by the reducer each tick.
    enum class SnakeEventType : uint8_t {
        HEAD_MOVED,      // head advanced one cell this tick
        SELF_COLLISION,  // head hit a wall or body — game over
        FOOD_EATEN       // ate food; tail won't drop this tick (snake grows)
    };

    // SnakeEvent: an event with its scoring payload.
    struct SnakeEvent {
        SnakeEventType type = SnakeEventType::HEAD_MOVED;
        int score_delta = 0;   // +10 per food eaten
    };

    // SnakeStepResult: output of one reducer tick — next state + emitted events.
    struct SnakeStepResult {
        SnakeSnapshot next_state;
        std::pmr::vector<SnakeEvent> events;
        bool alive = true;     // false once the snake has died (game over)
    };

} // namespace snake::matrix
