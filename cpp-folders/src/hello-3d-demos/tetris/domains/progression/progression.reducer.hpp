#pragma once
// tetris/domains/progression/progression.reducer.hpp — EVENT-FED SCORING (tetris::progression)
// Consumes ONLY matrix raw-fact events; never touches the grid (Rule 8.1).
// LUA HOOK (next task): compute_line_clear_score() below is the single pure
// rule function to swap for an edges/lua StatelessLuaEvaluator call returning
// { score_added, level_up, danger_alert } — signature stays value-in/value-out.
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

    // Pure scoring rule (Lua-swappable). Inputs are plain values; output is a
    // plain value. See docs/TODOS.md Part 2 for the intended blitz_mode.lua shape.
    static inline int compute_line_clear_score(const config::Rules& rules,
                                               int lines_cleared_count,
                                               int level, int combo) {
        return rules.base_scores[lines_cleared_count] * level
             + combo * rules.combo_bonus * level;
    }

    static inline ProgressionStep reduce_progression(
        std::span<const MatrixEvent>      matrix_events,
        const ScoreState&                 prev,
        const config::Rules&              rules,
        std::pmr::memory_resource*        frame_mr
    ) {
        ProgressionStep result(frame_mr);
        ScoreState&     s = result.next;
        s = prev;

        bool cleared_this_frame = false;

        for (const auto& ev : matrix_events) {
            switch (ev.type) {
            case MatrixEventType::HARD_DROP_SLAM: {
                int delta = ev.cells * rules.hard_drop_score_per_cell;
                s.score += delta;
                result.events.push_back({ .type = ProgressionEventType::SCORE_CHANGED, .score_delta = delta });
                break;
            }
            case MatrixEventType::SOFT_DROP: {
                int delta = ev.cells * rules.soft_drop_score_per_cell;
                s.score += delta;
                result.events.push_back({ .type = ProgressionEventType::SCORE_CHANGED, .score_delta = delta });
                break;
            }
            case MatrixEventType::LINES_CLEARED: {
                cleared_this_frame = true;
                s.combo_count++;
                s.lines_cleared += ev.lines_cleared_count;

                int delta = compute_line_clear_score(rules, ev.lines_cleared_count, s.level, s.combo_count);
                s.score += delta;
                result.events.push_back({ .type = ProgressionEventType::SCORE_CHANGED, .score_delta = delta });

                int new_level = 1 + (s.lines_cleared / rules.lines_per_level);
                if (new_level > s.level) {
                    s.level = new_level;
                    result.events.push_back({ .type = ProgressionEventType::LEVEL_UP, .new_level = new_level });
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
