#pragma once
// tetris/config/levels/marathon_01.hpp — LEVEL DEFINITION (tetris::config::Marathon01)
#include <config/rules.hpp>

namespace tetris::config {

    struct Marathon01 {
        static constexpr const char* NAME = "MARATHON";

        // Classic marathon: the shared default framing (config/camera.hpp) —
        // full 22-row board visible with HUD clearance on both sides.
        static Rules make_rules() { return Rules{}; }
    };

} // namespace tetris::config
