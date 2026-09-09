-- assets/levels/goals_dsl.lua - PREDICATE DSL (Part 7 G3)
--
-- Tiny pure-function library for composing mission goal conditions.
-- A predicate is a function (events, snapshot) -> boolean.
--   events   : array of { type="...", a=N, b=N }  (this tick's facts)
--   snapshot : { score=N, lines=N, level=N, overdrive=bool, stack_ratio=F }
--
-- Purity law: no os/io/math.random/print. Same inputs -> same answer.
-- Keep this file TINY: conditions only. Behavior belongs in C++ reducers.

GoalsDSL = {}

-- ALL sub-conditions must hold for this batch.
function GoalsDSL.when(...)
    local conds = { ... }
    return function(events, snap)
        for _, c in ipairs(conds) do
            if not c(events, snap) then return false end
        end
        return true
    end
end

-- AT LEAST ONE holds.
function GoalsDSL.any_of(...)
    local conds = { ... }
    return function(events, snap)
        for _, c in ipairs(conds) do
            if c(events, snap) then return true end
        end
        return false
    end
end

-- NONE hold.
function GoalsDSL.none_of(...)
    local conds = { ... }
    return function(events, snap)
        for _, c in ipairs(conds) do
            if c(events, snap) then return false end
        end
        return true
    end
end

-- Snapshot window gate: overdrive / any snapshot flag.
function GoalsDSL.during(flag)
    return function(events, snap)
        return snap[flag] == true
    end
end

-- Counting helper: how many events match type + optional filter.
local function count_matches(events, etype, filter_a)
    local n = 0
    for _, e in ipairs(events) do
        if e.type == etype and (filter_a == nil or e.a >= filter_a) then
            n = n + 1
        end
    end
    return n
end

-- count_where("LINES_CLEARED", 4).at_least(2): two clears of >= 4 lines.
function GoalsDSL.count_where(etype, filter_a)
    local api = {}
    function api.at_least(n)
        return function(events, snap)
            return count_matches(events, etype, filter_a) >= n
        end
    end
    function api.exactly(n)
        return function(events, snap)
            return count_matches(events, etype, filter_a) == n
        end
    end
    return api
end

return GoalsDSL
