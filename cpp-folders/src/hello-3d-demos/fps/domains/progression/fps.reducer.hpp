#pragma once

// ============================================================================
// fps/domains/progression/fps.reducer.hpp — reduce_progression (pure)
// (events, prev ScoreState) -> next ScoreState.
// ============================================================================

#include <span>

#include "fps.contract.hpp"

#include <domains/matrix/fps.event.hpp>

namespace fps::progression {

    inline ScoreState reduce_progression(std::span<const matrix::CombatEvent> events,
                                         const ScoreState& prev) {
        ScoreState next = prev;
        for (const auto& ev : events) {
            if (ev.type == matrix::EventType::BOT_KILLED) {
                next.score += 100;
                next.kills += 1;
            }
        }
        return next;
    }

} // namespace fps::progression