-- tetris/domains/powerups/scripts/cyber_storm.lua — L4 CYBER STORM RULES
-- Tier 3: gameplay MECHANICS authored in Lua (Pod 4). Pure stateless reducers
-- only: plain values in, plain values out. No RNG, no os/io, no print
-- (determinism sandbox strips them anyway; see edges/lua/lua.edge.hpp).
--
-- Special pieces (matrix PieceType ids):
--   BOMB = 9   detonates a 3x3 blast around its anchor cell
--   LASER = 10 vaporizes the entire row it locks on
--   FREEZE = 11 suspends gravity for freeze_seconds
--
-- The engine owns the grid; this script only DECIDES what mutates. Mutations
-- arrive as raw facts (ClearCells / FreezeGravity commands) via main wiring.

CyberRules = {}

local GRID_W = 10

-- Boot-time config merge (ARCHITECTURE §4.2): known keys patch Rules.
function CyberRules.get_config()
    return {
        mode_id         = 4,
        target_lines    = 24,
        target_score    = 0,
        time_limit      = 0,
        special_every_n = 5,
        freeze_seconds  = 5.0
    }
end

-- Spawn scheduler hook: called when the cadence counter reaches the threshold.
-- pieces_since_special >= special_every_n; armed_index cycles 1..3
-- (1=Bomb, 2=Laser, 3=Freeze). Return { type = N } with a matrix piece id,
-- or { type = 0 } to skip this window.
function CyberRules.decide_spawn(pieces_since_special, armed_index)
    -- Storm pattern: Bomb opens the storm, then Laser, then Freeze — but a
    -- tall stack begs for the Laser instead of another Bomb (pure function of
    -- the inputs; no state held here).
    local t = armed_index
    if armed_index == 1 and pieces_since_special > 12 then
        t = 2   -- stack pressure: swap the opener for a row-vaporizer
    end
    return { type = 8 + t }
end

-- Lock ruling: decide the mutation for a special that just locked at (gx, gy).
-- grid is a flat row-major array indexed [y * GRID_W + x + 1], y 0 = bottom;
-- values are 0 (empty) or piece ids. Returns:
--   clear_count + cx[] + cy[] : cells to zero (raw facts for ClearCells)
--   freeze_seconds            : > 0 suspends gravity (FreezeGravity)
--   fx_id                     : 1 blast / 2 laser sweep / 3 frost
function CyberRules.on_special_lock(special_type, gx, gy, grid)
    if special_type == 9 then
        -- BOMB: 3x3 crater around the anchor, clamped to the well.
        local cx, cy, n = {}, {}, 0
        for dy = -1, 1 do
            for dx = -1, 1 do
                local x, y = gx + dx, gy + dy
                if x >= 0 and x < GRID_W and y >= 0 and y < 22 then
                    n = n + 1
                    cx[n] = x
                    cy[n] = y
                end
            end
        end
        return { clear_count = n, cx = cx, cy = cy, freeze_seconds = 0, fx_id = 1 }
    elseif special_type == 10 then
        -- LASER: sweep the full row the beam locked on.
        local cx, cy, n = {}, {}, 0
        for x = 0, GRID_W - 1 do
            n = n + 1
            cx[n] = x
            cy[n] = gy
        end
        return { clear_count = n, cx = cx, cy = cy, freeze_seconds = 0, fx_id = 2 }
    elseif special_type == 11 then
        -- FREEZE: time-stop, nothing cleared.
        return { clear_count = 0, cx = {}, cy = {}, freeze_seconds = 5.0, fx_id = 3 }
    end
    return { clear_count = 0, cx = {}, cy = {}, freeze_seconds = 0, fx_id = 0 }
end