#pragma once

#include "progression.contract.hpp"

namespace tetris {
    namespace progression {

        // ============================================================================
        // Intent Tokens: Caller signals to the reducer what change is desired.
        // These are collected from input edges and fed into reduce_progression().
        // ============================================================================

        /// Switches game mode (Normal <-> Blitz <-> Sprint).
        struct ChangeModeIntent {
            ModeConfig::Type target_mode;
            float time_limit_seconds = 0.0f;
            int32_t target_score       = 12000;
        };

        /// Activates a countdown timer for Blitz mode (optional sub-mode).
        struct StartTimerIntent {
            float duration_seconds = 60.0f;
        };

        /// Pauses the timer in Blitz/Sprint modes.
        struct PauseTimerIntent {};

        /// Resumes a paused timer.
        struct ResumeTimerIntent {};

        /// Resets combo counter (e.g., after user manually resets).
        struct ResetComboIntent {};

    } // namespace progression
} // namespace tetris
