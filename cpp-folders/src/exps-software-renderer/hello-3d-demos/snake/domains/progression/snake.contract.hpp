#pragma once

#include <glm/glm.hpp>
#include <cstdint>

// progression pod — scoring & speed ramp. Listens to matrix events; never touches the grid.

namespace snake::progression {

    struct ScoreState {
        int score = 0;
        int high_score = 0;
        int length = 3;               // current body segment count (incl. head)
        float speed_mult = 1.0f;      // tick-interval multiplier (lower = faster)

        static ScoreState fresh() { return {}; }
    };

} // namespace snake::progression
