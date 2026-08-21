#pragma once

// snake level data: pure, replay-ready grid layout. Zero logic — consumed by the reducer + plan at runtime.
#include <cstdint>
#include <array>
#include <glm/glm.hpp>

namespace snake {

    struct SnakeLevel01 {
        static constexpr int GRID_W = 20;   // grid width (cells)
        static constexpr int GRID_H = 20;   // grid height (cells)

        glm::ivec2 head_spawn = { 9, 9 };              // initial head cell (bottom-left origin)
        std::array<glm::ivec2, 3> body_spawn = {
            { 8, 9 }, { 7, 9 }                          // initial body segments
        };

        glm::ivec2 dir_spawn = { 1, 0 };               // facing right (+X)

        glm::ivec2 food_spawn = { -1, -1 };            // empty until first spawn
        std::array<glm::ivec2, 8> food_table = {         // deterministic food spawn positions (x,y pairs)
            { 3, 3 }, { 16, 3 }, { 10, 16 }, { 4, 15 },
            { 17, 12 }, { 6, 8 }, { 13, 6 }, { 9, 18 }
        };

        glm::vec3 arena_center = { GRID_W * 0.5f - 0.5f, GRID_H * 0.5f - 0.5f, 0.0f }; // world-space center
        float     arena_half_w   = (float)GRID_W * 0.5f;
        float     arena_half_h   = (float)GRID_H * 0.5f;

        uint32_t rng_state = 0x9e3779b9u;               // deterministic seed for food spawn table
    };

} // namespace snake
