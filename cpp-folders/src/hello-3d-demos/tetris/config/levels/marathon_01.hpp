#pragma once
// tetris/config/levels/marathon_01.hpp — LEVEL DEFINITION (tetris::config::Marathon01)
#include <config/rules.hpp>

namespace tetris::config {

    struct Marathon01 {
        static constexpr const char* NAME = "MARATHON";

        static Rules make_rules() { return Rules{}; }   // defaults = classic marathon
    };

} // namespace tetris::config
