#pragma once

// progression pod — pure event-driven scoring & speed ramp. Listens to matrix events only;
// never touches the grid or rendering. Allocation-free.
#include <algorithm>
#include "snake.contract.hpp"
#include "snake.event.hpp"   // shared vocabulary (SnakeEventType) — DOD reusable-vocabulary test

namespace snake::progression {

    inline ScoreState reduce_progression(
        const std::pmr::vector<matrix::SnakeEvent>& events,
        ScoreState prev)
    {
        int score_delta = 0;
        for (const auto& e : events) {
            if (e.type == matrix::SnakeEventType::FOOD_EATEN) {
                score_delta += e.score_delta;   // +10 per food eaten
            }
        }
        prev.score += score_delta;

        // Speed ramp: faster ticks as length grows. Clamped to a floor of 0.3x.
        float new_speed_mult = prev.speed_mult * (1.0f - std::min(0.75f, (float)prev.length * 0.02f));
        if (new_speed_mult < 0.3f) new_speed_mult = 0.3f;
        prev.speed_mult = new_speed_mult;

        return prev;
    }

} // namespace snake::progression
