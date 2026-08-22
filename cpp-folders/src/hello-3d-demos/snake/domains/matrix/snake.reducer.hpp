#pragma once

// snake reducer: advances the head, resolves collisions + food, emits events. Pure state machine;
// never touches rendering or SDL. Works in grid cells; BodySoA stores grid-cell coordinates as
// vec3(x, y, 0) — one grid cell == one world unit, so grid space IS world space for this demo
// (the plan draws tiles at the same coordinates and orbits the camera around level.arena_center).
#include "snake.contract.hpp"
#include "snake.action.hpp"   // reduce_snake_commands — fold commands into a movement delta
#include "difficulty.hpp"     // snake::config::Difficulty (resolved via global include dir: <snake>/config)
#include "snake_level_01.hpp" // arena bounds (GRID_W/H) + food table (resolved via global include dir: <snake>/config/levels)

namespace snake::matrix {

    // Map a grid cell (x,y) to its world-space position on the arena plane.
    // One cell == one world unit; cell (0,0) sits at the origin and the board is centered on level.arena_center.
    inline glm::vec3 cell_to_world(const SnakeLevel01& level, int x, int y) {
        (void)level;
        return glm::vec3(float(x), float(y), 0.0f);
    }

    // Deterministic food advancement: walk the level's food_table in order, wrapping around.
    // Pure — no RNG, identical snapshots always yield identical next-food positions.
    inline FoodState advance_food(const FoodState& current, const SnakeLevel01& level) {
        FoodState next;
        const size_t n = level.food_table.size();
        if (current.pos.x < 0 || current.pos.y < 0) {
            next.pos = level.food_table[0];                       // first spawn
            return next;
        }
        size_t idx = 0;
        for (size_t i = 0; i < level.food_table.size(); ++i) {
            if (level.food_table[i] == current.pos) { idx = i; break; }
        }
        next.pos = level.food_table[(idx + 1) % n];               // cycle the table
        return next;
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

        glm::ivec2 new_head(snap.head_pos.x + static_cast<int>(delta.x),
                            snap.head_pos.y + static_cast<int>(delta.y));

        // 3. Wall collision — arena boundary. Solid walls are lethal; soft walls hold position this tick.
        const bool out_of_bounds =
            new_head.x < 0 || new_head.x >= level.GRID_W ||
            new_head.y < 0 || new_head.y >= level.GRID_H;

        if (out_of_bounds) {
            if (difficulty.solid_walls) {
                SnakeEvent ev{ SnakeEventType::SELF_COLLISION, 0 };
                return { snap, std::pmr::vector<SnakeEvent>{ev}, false };
            }
            // Soft wall: hold position — state fully UNCHANGED this tick.
            // (Returning snap verbatim matters: falling through to the normal-move path with
            // new_head == head_pos would vacate the tail and duplicate the head, silently
            // shrinking the snake one segment per tick while pressed against a wall.)
            return { snap, std::pmr::vector<SnakeEvent>{}, true };
        }

        // No movement intent (no keys, or input neutralized by the reverse-rejection above):
        // state fully unchanged. Same tail-vacate trap as the soft-wall hold — without this
        // early-out an IDLE snake decayed to a single segment within seconds.
        if (new_head == snap.head_pos) {
            return { snap, std::pmr::vector<SnakeEvent>{}, true };
        }

        const size_t n = snap.body.position.size();

        // 4. Food: eat → grow (tail does not vacate this tick) + advance to the next food table entry.
        if (new_head == snap.food.pos) {
            BodySoA grown_body;
            grown_body.position.reserve(n + 1);
            for (size_t i = 0; i < n; ++i) grown_body.position.push_back(snap.body.position[i]);
            grown_body.position.push_back(cell_to_world(level, new_head.x, new_head.y));

            SnakeSnapshot next{ new_head, delta, advance_food(snap.food, level), grown_body };
            SnakeEvent ev{ SnakeEventType::FOOD_EATEN, 10 };
            return { next, std::pmr::vector<SnakeEvent>{ev}, true };
        }

        // 5. Self-collision against the body. The tail vacates every tick (growth handled above),
        //    so only segments [0..n-2] are lethal to collide with. (i+1<n avoids underflow at n==0.)
        for (size_t i = 0; i + 1 < n; ++i) {
            glm::ivec2 seg(static_cast<int>(snap.body.position[i].x),
                           static_cast<int>(snap.body.position[i].y));
            if (new_head == seg) {
                SnakeEvent ev{ SnakeEventType::SELF_COLLISION, 0 };
                return { snap, std::pmr::vector<SnakeEvent>{ev}, false };
            }
        }

        // 6. Normal move: build next body as [head..n-2] + new_head, dropping the vacating tail.
        BodySoA next_body;
        next_body.position.reserve(n);
        for (size_t i = 0; i + 1 < n; ++i) next_body.position.push_back(snap.body.position[i]);
        next_body.position.push_back(cell_to_world(level, new_head.x, new_head.y));

        SnakeSnapshot next{ new_head, delta, snap.food, next_body };
        return { next, std::pmr::vector<SnakeEvent>{}, true };
    }

} // namespace snake::matrix