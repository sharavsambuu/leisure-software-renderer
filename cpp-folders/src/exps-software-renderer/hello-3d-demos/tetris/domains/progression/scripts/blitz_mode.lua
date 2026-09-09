-- tetris/domains/progression/scripts/blitz_mode.lua
-- BLITZ 120 — pure stateless Lua decision rules (Constitution II Rule 8.2).
-- Zero globals mutated, zero side effects, zero os/io/math.random (the
-- evaluator strips them). Designers retune the whole blitz economy by
-- editing THIS file only — no C++ changes, no recompiles.

local BlitzRules = {}

-- Boot-time economy overrides: plain table out, merged into config::Rules
-- by main before frame 0 (smoke-gated: target_score must land in ScoreState).
function BlitzRules.get_config()
    return {
        mode_id      = 2,       -- config::MODE_BLITZ_120
        target_score = 20000,   -- aggressive sprint target
        time_limit   = 120.0    -- 2-minute clock
    }
end

-- Per-line-clear scoring rule: (level, lines, combo, is_tspin) -> ruling.
-- Pure value-in/value-out; same inputs always give the same ruling.
function BlitzRules.calculate_score(level, lines_cleared, combo_count, is_tspin)
    local base_scores = { [1] = 100, [2] = 300, [3] = 500, [4] = 800 }
    local base = base_scores[lines_cleared] or 0

    if is_tspin then
        base = base * 2
    end

    local combo_bonus = combo_count * 50 * level
    local score_added = (base * level) + combo_bonus

    -- Tetris clears buy back 5 seconds of clock (time-bonus economy).
    local time_bonus = 0.0
    if lines_cleared >= 4 then
        time_bonus = 5.0
    end

    return {
        score_added  = score_added,
        level_up     = (lines_cleared >= 4),
        danger_alert = (lines_cleared <= 1 and combo_count <= 1),
        time_bonus   = time_bonus
    }
end

-- Per-tick clock rule: (time_left, stack_height) -> urgency flags.
function BlitzRules.evaluate_clock(time_left, stack_height)
    return {
        danger_alert = (time_left < 30.0) or (stack_height >= 16),
        hurry        = (time_left < 10.0)
    }
end

return BlitzRules