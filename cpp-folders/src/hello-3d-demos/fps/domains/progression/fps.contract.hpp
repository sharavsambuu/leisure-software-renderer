#pragma once

// ============================================================================
// fps/domains/progression/fps.contract.hpp — ScoreState (fps::progression)
// Pure scoring state, fed by CombatEvents. Extracted from PlayerSnapshot so
// combat logic never mutates score directly (snake lesson).
// ============================================================================

#include <cstdint>

namespace fps::progression {

    struct ScoreState {
        int32_t score = 0;
        int32_t kills = 0;

        static constexpr ScoreState fresh() {
            return ScoreState{};
        }
    };

} // namespace fps::progression