-- assets/levels/cyber_storm/goals.lua - SCRIPTED GOALS (G3 demo)
--
-- Cyber Storm's victory ladder authored as pure predicates. Loaded through
-- the sandboxed evaluator; consumed by domains/mission via IScriptHost
-- evaluate_goal("Goals", events, snapshot).
--
-- Contract: Goals.test(goal_id, events, snapshot) -> truthy to complete.
-- goal_id lets one script own several conditions cleanly.

Goals = {}

local dsl = dofile(TETRIS_SOURCE_ROOT .. "/assets/levels/goals_dsl.lua")
local when      = dsl.when
local during    = dsl.during
local count_where = dsl.count_where

-- Per-goal batch predicates. Cumulative counters stay host-side
-- (mission.reducer EVENT_COUNT style) because they need cross-tick memory.
local predicates = {
    -- storm_1: clear 10 lines (host counts cumulatively; here we just gate)
    storm_1 = function(events, snap)
        return snap.lines >= 10
    end,

    -- storm_2: reach 15000 score while ANY clear happens under overdrive
    storm_2 = when(
        during("overdrive"),
        function(events, snap)
            for _, e in ipairs(events) do
                if e.type == "LINES_CLEARED" then return snap.score >= 15000 end
            end
            return false
        end
    ),

    -- storm_3: land 2 special pieces (special locks arrive as
    -- SPECIAL_LOCKED events with b=lock_y, a=special_type)
    storm_3 = function(events, snap)
        return snap.special_locks >= 2
    end,
}

-- Host entry point. Returns truthy => goal complete this tick.
function Goals.test(goal_id, events, snapshot)
    local p = predicates[goal_id]
    if not p then return false end
    return p(events, snapshot)
end

return Goals
