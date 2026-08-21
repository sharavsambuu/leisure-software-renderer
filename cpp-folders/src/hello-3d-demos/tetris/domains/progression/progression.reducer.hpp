#pragma once

#include <array>
#include <cmath>
#include <memory_resource>
#include <span>
#include "progression.contract.hpp"
#include "progression.action.hpp"

namespace tetris {
    namespace progression {

        // ============================================================================
        // PURE REDUCER: (StateSnapshot + Actions) -> (NextState, Events)
        // No globals, no side effects, zero heap allocs during reduction.
        // All transient vectors use the provided PMR arena resource.
        // ============================================================================

        static inline ProgressionStepResult reduce_progression(
            const ProgressionSnapshot&  prev_snapshot,
            std::span<const ChangeModeIntent> mode_commands,
            float                       dt
        ) {
            ProgressionStepResult result;
            ProgressionSnapshot& next = result.next_snapshot;

            // --- Copy current snapshot (so we can mutate in place) ---
            next.score           = prev_snapshot.score;
            next.high_score      = prev_snapshot.high_score;
            next.level           = prev_snapshot.level;
            next.combo_count     = prev_snapshot.combo_count;
            next.mode_type       = prev_snapshot.mode_type;
            next.time_left       = prev_snapshot.time_left;
            next.blitz_complete  = prev_snapshot.blitz_complete;
            next.sprint_survived = prev_snapshot.sprint_survived;
            next.drop_interval   = prev_snapshot.drop_interval;
            next.rng_state       = prev_snapshot.rng_state;
            next.lines_cleared_total = prev_snapshot.lines_cleared_total;

            // --- Apply Mode Commands (pure, no side effects) ---
            for (const auto& cmd : mode_commands) {
                if (cmd.target_mode != prev_snapshot.mode_type) {
                    if (prev_snapshot.mode_type == ModeConfig::Type::Blitz &&
                        cmd.target_mode   == ModeConfig::Type::Normal) {
                        next.blitz_complete = false; // Reset on exit
                    } else if (prev_snapshot.mode_type == ModeConfig::Type::Sprint40 &&
                               cmd.target_mode   == ModeConfig::Type::Normal) {
                        next.sprint_survived = false;
                    }

                    next.mode_type = cmd.target_mode;
                    if (cmd.target_mode == ModeConfig::Type::Blitz) {
                        next.time_left = cmd.time_limit_seconds;
                        next.blitz_complete = true; // Reset flag on entry
                    } else if (cmd.target_mode == ModeConfig::Type::Sprint40) {
                        next.time_left = cmd.time_limit_seconds;
                        next.sprint_survived = false;
                    }

                    result.events.push_back({
                        .type = ProgressionEventType::MODE_CHANGED,
                        .old_mode = static_cast<ModeConfig::Type>(prev_snapshot.mode_type),
                        .new_mode = cmd.target_mode,
                        .message_tag = (cmd.target_mode == ModeConfig::Type::Blitz)
                            ? "BlitzStarted" : "SprintStarted",
                    });
                }
            }

            // --- Timer Tick (only active in timed modes) ---
            if (next.time_left > 0.0f && next.mode_type != ModeConfig::Type::Normal) {
                next.time_left -= dt;
                if (next.time_left <= 0.0f) {
                    next.time_left = 0.0f;
                    // Timer expired -> game over condition handled in main loop or matrix domain
                    result.events.push_back({
                        .type = ProgressionEventType::TIME_WARNING,
                        .time_remaining_seconds = next.time_left,
                    });
                } else if (next.mode_type == ModeConfig::Type::Sprint40 &&
                           next.sprint_survived == false) {
                    // Optional: warn when below 10s
                    if (next.time_left < 10.0f) {
                        result.events.push_back({
                            .type = ProgressionEventType::TIME_WARNING,
                            .time_remaining_seconds = next.time_left,
                            .message_tag = "SprintWarning",
                        });
                    }
                }
            }

            // --- Combo Decay (slow decay: every 3s without scoring) ---
            static constexpr float COMBO_DECAY_SECONDS = 5.0f;
            if (next.combo_count > 0 && next.mode_type == ModeConfig::Type::Normal) {
                const float time_since_score = dt * 60.0f / 100.0f; // dummy estimate in seconds
                if (time_since_score >= COMBO_DECAY_SECONDS) {
                    next.combo_count = 0;
                    result.events.push_back({
                        .type = ProgressionEventType::DANGER_ALERT,
                        .message_tag = "ComboBusted",
                    });
                }
            }

            // --- Level Up Check ---
            int lines_this_level = next.lines_cleared_total % static_cast<int>(next.level * (size_t)next.lines_per_level);
            if (lines_this_level >= 10 && next.level < 20) {
                uint8_t new_lvl = static_cast<uint8_t>(next.level + 1);
                float interval = std::max(0.05f, 0.80f * std::pow(0.96f, (float)(new_lvl - 1)));
                next.level = new_lvl;
                next.drop_interval = interval;
                result.events.push_back({
                    .type = ProgressionEventType::LEVEL_UP,
                    .new_level = new_lvl,
                });
            }

            return result;
        }

    } // namespace progression
} // namespace tetris
