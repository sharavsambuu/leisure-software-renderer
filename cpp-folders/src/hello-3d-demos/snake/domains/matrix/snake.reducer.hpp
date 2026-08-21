#pragma once

// snake reducer: advances the head, resolves collisions + food, emits events. Pure state machine;
// never touches rendering or SDL. Works in grid cells, converts to world space for the body SoA.
#include "snake.contract.hpp"
#include "snake.action.hpp"   // reduce_snake_commands — fold commands into a movement delta
#include "../../../config/levels/snake_level_01.hpp"   // arena bounds (GRID_W/H) + food table

namespace snake::matrix {

    // Map a grid cell (x,y) to its world-space center on the arena plane.
    inline glm::vec3 cell_to_world(const SnakeLevel01& level, int x, int y) {
        return glm::vec3(
            level.arena_center.x + level.arena_half_w * float(x),
            level.arena_center.y + level.arena_half_h * float(y),
            0.0f);
    }

    // Advance the head one cell, resolve collisions/food, and return next state + events.
    inline SnakeStepResult reduce_snake(
        const SnakeSnapshot& snap,
        std::span<const SnakeCommand> commands,
        config::Difficulty difficulty,
        const SnakeLevel01& level)
    {
        // 1. Fold commands into a single grid-aligned movement delta (dx,dy).
        glm::vec2 delta = reduce_snake_commands(commands);

        // 2. Reject reversing directly into the neck (would collide with own body).
        if (delta.x == -snap.head_dir.x && delta.y == -snap.head_dir.y) { delta = {0, 0}; }

        glm::ivec2 head_x(snap.head_pos.x + static_cast<int>(delta.x), snap.head_pos.y + static_cast<int>(delta.y));

        // 3. Wall collision — arena boundary. Solid walls are lethal; soft walls bounce (stay put).
        bool out_of_bounds =
            head_x.x < 0 || head_x.x >= level.GRID_W ||
            head_x.y < 0 || head_x.y >= level.GRID_H;

        if (out_of_bounds) {
            if (difficulty.solid_walls) {
                SnakeEvent ev{ SnakeEventType::SELF_COLLISION, 0 };
                return { snap, std::pmr::vector<SnakeEvent>{ev}, false };
            }
            // Soft wall: reverse direction and hold position this tick.
            delta = {-snap.head_dir.x, -snap.head_dir.y};
        }

        // Recompute head from the (possibly reversed) delta.
        glm::ivec2 new_head(snap.head_pos.x + static_cast<int>(delta.x), snap.head_pos.y + static_cast<int>(delta.y));

        // 4. Food: eat, grow (tail does not vacate this tick). Growth returns early below.
        if (new_head == snap.food.pos) {
            SnakeEvent ev{ SnakeEventType::FOOD_EATEN, 10 };
            return { snap, std::pmr::vector<SnakeEvent>{ev}, true };   // alive + grew — tail stays below
        }

        const int n = static_cast<int>(snap.body.position.size());

        // 5. Self-collision against the body. The tail vacates every tick (growth handled above),
        //    so only segments [0..n-2] are lethal to collide with.
        for (int i = 0; i < n - 1; ++i) {
            glm::ivec2 seg(snap.body.position[i].x, snap.body.position[i].y);
            if (new_head == seg) {
                SnakeEvent ev{ SnakeEventType::SELF_COLLISION, 0 };
                return { snap, std::pmr::vector<SnakeEvent>{ev}, false };
            }
        }

        // Build next body: [new_head] + current[0..n-2], dropping the vacating tail.
        BodySoA next_body;
        next_body.position.reserve(n);
        for (int i = 0; i < n - 1; ++i) {
            glm::vec3 w = cell_to_world(level, snap.body.position[i].x, snap.body.position[i].y);
            next_body.position.push_back(w);
        }
        next_body.position.push_back(cell_to_world(level, new_head.x, new_head.y));

        SnakeSnapshot next{ new_head, delta, snap.food, next_body };
        return { next, std::pmr::vector<SnakeEvent>{}, true };
    }

} // namespace snake::matrix
