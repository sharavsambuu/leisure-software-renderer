#pragma once

// snake config: difficulty settings and level meta-table (replay-ready).
#include <cstdint>
#include <array>

namespace snake::config {

    struct Difficulty {
        bool solid_walls = false;   // true = walls are impenetrable (die on touch); false = soft walls (bounce)
        uint32_t speed_ticks = 15;  // ticks per second of game loop (~6.67 Hz at default)
    };

    struct LevelMeta {
        const char* name;
        Difficulty difficulty;
        int food_count;      // number of food items to spawn on level start
    };

} // namespace snake::config

// Canonical difficulty table — replay-ready, deterministic per level index.
namespace snake::config {

    constexpr std::array<LevelMeta, 8> levels = {{
        LevelMeta{"Easy",   Difficulty{false, 15}, 3},
        LevelMeta{"Normal", Difficulty{false, 10}, 6},
        LevelMeta{"Hard",   Difficulty{true, 7},  9},
        LevelMeta{"Nightmare",Difficulty{true,4}, 12},
        LevelMeta{"Insane", Difficulty{true, 3}, 15},
        LevelMeta{"Chaos",  Difficulty{false, 20}, 8},
        LevelMeta{"Zen",    Difficulty{false, 6},  4},
        LevelMeta{"Speedrun",Difficulty{true,1},   20},
    }};

} // namespace snake::config
