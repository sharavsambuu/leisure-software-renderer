#pragma once

#include "progression.contract.hpp"

namespace tetris {
    namespace progression {

        // ============================================================================
        // Discrete Event Types: Immutable records of occurrences emitted by the reducer.
        // These events are consumed by other domains (audio, environment) without mutation.
        // ============================================================================

        enum class ProgressionEventType : uint8_t {
            SCORE_CHANGED,
            COMBO_STREAK,
            LEVEL_UP,
            MODE_CHANGED,
            TIME_WARNING,
            OBJECTIVE_COMPLETED,
            DANGER_ALERT
        };

        struct ProgressionEvent {
            ProgressionEventType type;
            int32_t score_delta = 0;
            uint8_t new_level = 1;
            float time_remaining_seconds = -1.0f;
            ModeConfig::Type old_mode, new_mode;
            std::string_view message_tag; // "BlitzComplete", "SprintSurvived", "ComboBusted"
        };

    } // namespace progression
} // namespace tetris
