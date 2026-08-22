#pragma once
// tetris/domains/progression/progression.reducer.hpp — EVENT-FED SCORING (tetris::progression)
// Consumes ONLY matrix raw-fact events; never touches the grid (Rule 8.1).
//
// Rule-source routing (L2 Blitz 120): main injects pure value-in/value-out
// function pointers (ScriptHooks) that bridge to the lua edge. Domains never
// see lua_State*. Null hooks ⇒ native C++ rules (marathon baseline / fallback
// when a script is missing or errors — ARCHITECTURE.md §4.2).
#include <algorithm>
#include <memory_resource>
#include <span>
#include <vector>

#include <config/rules.hpp>
#include <domains/matrix/matrix.event.hpp>
#include <domains/progression/progression.contract.hpp>
#include <domains/progression/progression.event.hpp>

namespace tetris::progression {
using tetris::matrix::MatrixEventType;
using tetris::matrix::MatrixEvent;

    struct ProgressionStep {
        ScoreState                      next;
        std::pmr::vector<ProgressionEvent> events;

        explicit ProgressionStep(std::pmr::memory_resource* mr) : events(mr) {}
    };

    // Native C++ scoring rule (fallback when no script hook is injected).
    static inline int compute_line_clear_score(const config::Rules& rules,
                                               int lines_cleared_count,
                                               int level, int combo) {
        return rules.base_scores[lines_cleared_count] * level
             + combo * rules.combo_bonus * level;
    }

    // Value patch returned by a line-clear ruling (script or native).
    struct LineClearRuling {
        int   score_added  = 0;
        bool  level_up     = false;
        bool  danger_alert = false;
        float time_bonus   = 0.0f;   // seconds bought back (blitz economy)
    };

    // Pure value-in/value-out rule hooks injected by main (see edges/lua).
    // Plain function pointers only — no captures, no lua_State*, no closures.
    struct ScriptHooks {
        LineClearRuling (*line_clear_score)(int level, int lines, int combo,
                                            bool is_tspin)                = nullptr;
        void            (*clock_rule)(float time_left, int stack_height,
                                      bool* danger_alert, bool* hurry)  = nullptr;
    };

    static inline ProgressionStep reduce_progression(
        std::span<const MatrixEvent>      matrix_events,
        const ScoreState&                 prev,
        const config::Rules&              rules,
        std::pmr::memory_resource*        frame_mr,
        float                             dt           = 0.0f,
        int                               stack_height = 0,
        const ScriptHooks&                hooks        = ScriptHooks{}
    ) {
        ProgressionStep result(frame_mr);
        ScoreState&     s = result.next;
        s = prev;

        bool cleared_this_frame = false;

        // --- Blitz clock tick (timed modes only) ------------------------------
        if (rules.time_limit > 0.0f && !s.time_up) {
            const float t_before = s.time_left;
            s.time_left = std::max(0.0f, s.time_left - dt);

            if (hooks.clock_rule) {
                hooks.clock_rule(s.time_left, stack_height, &s.clock_danger, &s.clock_hurry);
            } else {
                s.clock_danger = (s.time_left < 30.0f) || (stack_height >= 16);
                s.clock_hurry  = (s.time_left < 10.0f);
            }

            // 30-second boundary crossing → shockwave-ring feed (spatial_fx).
            if (s.time_left > 0.0f &&
                static_cast<int>(t_before / 30.0f) != static_cast<int>(s.time_left / 30.0f)) {
                result.events.push_back({ .type = ProgressionEventType::CLOCK_TICK });
            }

            if (s.time_left <= 0.0f && !s.victory) {
                s.time_up = true;
                result.events.push_back({ .type = ProgressionEventType::TIME_UP });
            }
        }

        for (const auto& ev : matrix_events) {
            switch (ev.type) {
            case MatrixEventType::HARD_DROP_SLAM: {
                int delta = ev.cells * rules.hard_drop_score_per_cell;
                s.score += delta;
                s.score_drops += delta;
                result.events.push_back({ .type = ProgressionEventType::SCORE_CHANGED, .score_delta = delta });
                break;
            }
            case MatrixEventType::SOFT_DROP: {
                int delta = ev.cells * rules.soft_drop_score_per_cell;
                s.score += delta;
                s.score_drops += delta;
                result.events.push_back({ .type = ProgressionEventType::SCORE_CHANGED, .score_delta = delta });
                break;
            }
            case MatrixEventType::LINES_CLEARED: {
                cleared_this_frame = true;
                s.combo_count++;
                s.max_combo = std::max(s.max_combo, s.combo_count);
                s.lines_cleared += ev.lines_cleared_count;

                // Rule-source routing: script hook when wired, native otherwise.
                LineClearRuling ruling;
                if (hooks.line_clear_score) {
                    ruling = hooks.line_clear_score(s.level, ev.lines_cleared_count,
                                                    s.combo_count, /*is_tspin*/ false);
                } else {
                    ruling.score_added = compute_line_clear_score(rules, ev.lines_cleared_count,
                                                                  s.level, s.combo_count);
                    ruling.level_up    = (1 + s.lines_cleared / rules.lines_per_level) > s.level;
                }

                s.score += ruling.score_added;
                s.score_clears += ruling.score_added;
                result.events.push_back({ .type = ProgressionEventType::SCORE_CHANGED, .score_delta = ruling.score_added });

                if (s.combo_count >= 2) {
                    result.events.push_back({ .type = ProgressionEventType::COMBO_STREAK, .combo = s.combo_count });
                }

                if (ruling.time_bonus > 0.0f && rules.time_limit > 0.0f) {
                    s.time_left = std::min(rules.time_limit, s.time_left + ruling.time_bonus);
                    s.time_bonus_total += ruling.time_bonus;
                    result.events.push_back({ .type = ProgressionEventType::TIME_BONUS, .seconds = ruling.time_bonus });
                }

                if (ruling.level_up) {
                    s.level += 1;
                    result.events.push_back({ .type = ProgressionEventType::LEVEL_UP, .new_level = s.level });
                }
                break;
            }
            case MatrixEventType::PIECE_LOCK_IMPACT: {
                if (!cleared_this_frame) s.combo_count = 0;   // combo breaks on sterile lock
                break;
            }
            default:
                break;
            }
        }

        s.high_score = std::max(s.high_score, s.score);
        if (!s.victory && s.score >= s.target_score) {
            s.victory = true;
            result.events.push_back({ .type = ProgressionEventType::OBJECTIVE_COMPLETED });
        }

        return result;
    }

} // namespace tetris::progression