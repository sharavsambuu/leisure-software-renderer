#pragma once

#include <cstdint>
#include <array>
#include <string_view>
#include "shs_renderer.hpp"

namespace tetris {
    namespace progression {

        // ============================================================================
        // POD 2: PROGRESSION — Scoring, Combos, Game Modes, Level Curves
        // ============================================================================

        /// A game mode config (blitz/sprint/marathon). Loaded from Lua.
        struct ModeConfig {
            enum class Type : uint8_t { Normal, Blitz, Sprint40, Marathon };

            // Time limit in seconds (only used by Blitz and Sprint modes)
            float time_limit_seconds = 0.0f;

            /// Score required to win / complete level
            int32_t target_score = 12000;

            /// Minimum lines to clear before level-up
            uint8_t lines_per_level = 10;

            // Mode-specific thresholds
            bool blitz_cleared_any = false;   // Blitz: any single-line clear counts
            bool sprint_deadline_hits = false; // Sprint: hit time threshold at end

            ModeConfig() { type = Type::Normal; }
        };

        /// Discrete progression events emitted by the reducer.
        enum class ProgressionEventType : uint8_t {
            SCORE_CHANGED,   // Score delta applied (from combo or level-up)
            COMBO_STREAK,    // Combo counter incremented
            LEVEL_UP,        // Level increased; drop interval decreased
            MODE_CHANGED,    // Game mode switched (Normal -> Blitz, etc.)
            TIME_WARNING,    // Time remaining dropped below threshold
            OBJECTIVE_COMPLETED, // Mode win condition met
            DANGER_ALERT     // No points in a while / combo busted
        };

        struct ProgressionEvent {
            ProgressionEventType type;
            int32_t score_delta = 0;
            uint8_t new_level = 1;
            float time_remaining_seconds = -1.0f;
            ModeConfig::Type old_mode, new_mode;
            std::string_view message_tag; // "BlitzComplete", "SprintSurvived", etc.
        };

        /// Persistent progression state that survives across frames.
        struct ProgressionSnapshot {
            int32_t score           = 0;
            int32_t high_score      = 0;
            uint8_t level           = 1;
            uint8_t combo_count     = 0; // Consecutive lines cleared without zeroing

            ModeConfig::Type        mode_type   = ModeConfig::Type::Normal;
            float                   time_left   = -1.0f; // -1 means infinite (normal)
            bool                    blitz_complete = false;
            bool                    sprint_survived = false;

            float                   drop_interval = 0.80f; // Updated at level-up
            uint32_t               rng_state    = 0xDEADBEEFu; // For reproducible scoring tests

            int32_t                 lines_cleared_total = 0;
        };

        /// Result struct returned by the pure reducer function.
        struct ProgressionStepResult {
            ProgressionSnapshot       next_snapshot;
            std::pmr::vector<ProgressionEvent> events;
        };

    } // namespace progression
} // namespace tetris
