-- tetris/domains/matrix/scripts/garbage_canyon.gen.lua
-- GARBAGE CANYON — pure stateless Lua BOARD GENERATOR (Constitution II Rule 8.2).
-- Zero globals mutated, zero side effects, zero os/io/math.random (the
-- evaluator strips them). The whole level layout is authored HERE: designers
-- reshape the canyon by editing THIS file only — no C++ changes, no recompiles.
--
-- Contract (Use Case 4 pattern, ARCHITECTURE.md Part II):
--   CanyonGen.generate(difficulty, seed)
--       -> { rows          = { "....XX....", ... },   -- BOTTOM-UP strings,
--                                                   --   'X' = garbage block,
--                                                   --   '.' = hole
--            target_lines  = 20,                    -- win = excavate N lines
--            time_limit    = 180.0,                 -- excavation sprint clock
--            mode_id       = 3,                     -- config::MODE_GARBAGE_CANYON
--            target_score  = 0,                     -- unused in excavation mode
--            seed_tag      = <int> }                -- corner tag (daily identity)
--
-- Determinism: the sandbox strips math.random, so the generator carries its
-- own Lehmer LCG (MINSTD: a=48271, m=2^31-1). All math stays integer-exact in
-- doubles (< 2^53), so same seed => byte-identical board, every run.

-- NOTE: CanyonGen is a GLOBAL (same pattern as BlitzRules in blitz_rules.lua):
-- the evaluator's loadbuffer+pcall discards chunk return values, so the bridge
-- looks the rule table up via lua_getglobal.
CanyonGen = {}

local M = 2147483647            -- 2^31 - 1 (prime modulus)
local A = 48271                 -- MINSTD multiplier

local function lcg_next(s)
    return (s * A) % M
end

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

-- Boot-time objective overrides: plain table out, merged into config::Rules
-- by main before frame 0.
function CanyonGen.get_config()
    return {
        mode_id      = 3,       -- config::MODE_GARBAGE_CANYON
        target_lines = 20,      -- excavate 20 lines to win
        time_limit   = 180.0    -- 3-minute dig clock
    }
end

-- (difficulty, seed) -> pre-ruined board + objective numbers.
-- difficulty 1..5 scales tower height and hole density; seed picks the variant.
function CanyonGen.generate(difficulty, seed)
    local s = (seed % (M - 1)) + 1        -- Lehmer rejects state 0
    local function rnd()                  -- uniform [0,1), deterministic
        s = lcg_next(s)
        return s / M
    end

    local W, H = 10, 20                   -- matches matrix GRID_W / VISIBLE_H
    local diff = clamp(math.floor(difficulty or 1), 1, 5)

    -- 1) Column heights: a random walk — staggered garbage towers with a
    --    natural skyline. Capped at row 14 so the spawn buffer stays clear.
    local lo_h = 3 + diff
    local hi_h = math.min(8 + diff * 2, 14)
    local h = lo_h + math.floor(rnd() * 3)
    local heights = {}
    for x = 1, W do
        h = h + math.floor(rnd() * 5) - 2             -- step -2..+2
        h = clamp(h, lo_h, hi_h)
        heights[x] = h
    end

    -- 2) Guaranteed climbing channel: a zig-zag shaft of holes so every seed
    --    produces an excavatable canyon (no unwinnable boards).
    local holes = {}
    for y = 1, H do holes[y] = {} end
    local channel_col = 2 + math.floor(rnd() * (W - 4))
    for y = 1, 14 do
        holes[y][channel_col] = true
        if rnd() < 0.45 then
            local dir = (rnd() < 0.5) and -1 or 1
            local nc = channel_col + dir
            if nc >= 1 and nc <= W then
                holes[y][nc] = true
                channel_col = nc
            end
        end
    end

    -- 3) Extra pockets: higher difficulty scatters more isolated holes.
    for y = 1, 12 do
        for k = 1, diff - 1 do
            if rnd() < 0.30 then
                holes[y][1 + math.floor(rnd() * W)] = true
            end
        end
    end

    -- 4) Emit rows bottom-up as strings ('X' garbage, '.' hole).
    local rows = {}
    for y = 1, H do
        local cells = {}
        for x = 1, W do
            local filled = (y <= heights[x]) and not holes[y][x]
            cells[x] = filled and "X" or "."
        end
        rows[y] = table.concat(cells, "")
    end

    return {
        rows         = rows,
        target_lines = 20,
        time_limit   = 150.0 + diff * 10.0,
        mode_id      = 3,
        target_score = 0,
        seed_tag     = seed % 100000
    }
end

return CanyonGen