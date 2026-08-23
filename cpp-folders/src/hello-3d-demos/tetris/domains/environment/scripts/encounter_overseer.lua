-- encounter_overseer.lua — L5 ENCORE FINALE (pure stateless boss-phase rules)
-- The whole 4-phase encounter is authored here: CALM → RAIN → BLACKOUT →
-- CRESCENDO. Pure functions of plain values only — no globals mutated, no
-- RNG, no I/O (sandbox purity gate enforces this). Designers retune the
-- show by editing thresholds/mood targets below; C++ never hardcodes them.

Encounter = {}

-- Encounter tuning: phase count, rain cadence/rows, mood targets per phase.
function Encounter.get_config()
    return {
        phase_count = 4,
        rain_every  = 8.0,   -- seconds between garbage volleys in PHASE_RAIN
        rain_rows   = 1,     -- garbage rows per volley
        mood_calm      = 0.00,   -- cyan
        mood_rain      = 0.45,   -- drifting toward crimson
        mood_blackout  = 0.70,   -- deep crimson
        mood_crescendo = 1.00    -- gold
    }
end

-- Phase machine: pure function of (phase, time-in-phase, lines, danger).
-- lines = total lines cleared this run; danger = stack near the ceiling.
function Encounter.decide_phase(phase, phase_time, lines, danger)
    if phase == 1 then
        -- CALM until 6 lines dug or 40s elapse.
        if lines >= 6 or phase_time >= 40.0 then
            return { new_phase = 2, mood_target = 0.45 }
        end
        return { new_phase = 0, mood_target = 0.00 }

    elseif phase == 2 then
        -- RAIN until 14 total lines or 50s in the rain.
        if lines >= 14 or phase_time >= 50.0 then
            return { new_phase = 3, mood_target = 0.70 }
        end
        return { new_phase = 0, mood_target = 0.45 }

    elseif phase == 3 then
        -- BLACKOUT lasts 22s (dimming is driven by the phase itself).
        if phase_time >= 22.0 then
            return { new_phase = 4, mood_target = 1.00 }
        end
        return { new_phase = 0, mood_target = 0.70 }

    end

    -- CRESCENDO holds to victory.
    return { new_phase = 0, mood_target = 1.00 }
end

-- Crowd reaction to discrete events (value-in/value-out):
-- type 1 = LINES_CLEARED (value = line count), type 2 = VICTORY/OBJECTIVE.
function Encounter.on_event(event_type, value)
    if event_type == 1 then
        return { crowd_pulse = 0.25 + 0.15 * value }
    elseif event_type == 2 then
        return { crowd_pulse = 1.0 }
    end
    return { crowd_pulse = 0.0 }
end

return Encounter