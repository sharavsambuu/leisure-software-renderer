#pragma once
// tetris/domains/progression/progression.contract.hpp — SCORE STATE (tetris::progression)
namespace tetris::progression {

    struct ScoreState {
        int  score         = 0;
        int  high_score    = 0;
        int  lines_cleared = 0;
        int  level         = 1;
        int  combo_count   = 0;
        int  target_score  = 12000;
        bool victory       = false;
    };

} // namespace tetris::progression
