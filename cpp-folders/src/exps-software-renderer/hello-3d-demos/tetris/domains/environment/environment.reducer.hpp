#pragma once
// tetris/domains/environment/environment.reducer.hpp — PURE TRANSITION
// (tetris::environment) State_{t+1} = f(State_t, OverseerRuling, CrowdPulse,
// dt). Interpolates mood/dim toward the scripted targets, tracks phase time,
// and runs the garbage-rain cadence clock. Zero platform refs, zero Lua.
#include <algorithm>
#include <cstdint>

#include <domains/environment/environment.contract.hpp>

namespace tetris::environment {

    struct EnvironmentStepResult {
        EnvironmentSnapshot next;
        bool  rain_due      = false;   // one-shot: RAIN cadence timer elapsed
        bool  phase_changed = false;   // one-shot: ruling moved us to a new phase
    };

    // rain_every <= 0 disables the cadence clock (script owns scheduling).
    static inline EnvironmentStepResult reduce_environment(
        const EnvironmentSnapshot& prev,
        const OverseerRuling&      ruling,
        const CrowdPulse&          pulse,
        float                      dt,
        float                      rain_every
    ) {
        EnvironmentStepResult out;
        out.next = prev;

        // Phase transition (edge-triggered by the overseer's ruling).
        if (ruling.valid && ruling.new_phase > 0 && ruling.new_phase != prev.phase) {
            out.next.phase      = ruling.new_phase;
            out.next.phase_time = 0.0f;
            if (out.next.phase == PHASE_RAIN) {
                out.next.rain_timer = rain_every;   // first volley after a full interval
            }
            out.phase_changed = true;
        } else {
            out.next.phase_time += dt;
        }

        // Mood interpolation toward the scripted target (cyan → crimson → gold).
        if (ruling.valid && ruling.mood_target >= 0.0f) {
            const float rate = 0.35f * dt;   // ~3s to traverse the full range
            if (out.next.mood < ruling.mood_target) {
                out.next.mood = std::min(ruling.mood_target, out.next.mood + rate);
            } else if (out.next.mood > ruling.mood_target) {
                out.next.mood = std::max(ruling.mood_target, out.next.mood - rate);
            }
        }

        // Blackout dimming eases in/out (ghost hides past the halfway point —
        // projected downstream as fx.ghost_hidden, never stored here).
        const float dim_target = (out.next.phase == PHASE_BLACKOUT) ? 1.0f : 0.0f;
        const float dim_rate   = 0.8f * dt;
        if (out.next.dim < dim_target) {
            out.next.dim = std::min(dim_target, out.next.dim + dim_rate);
        } else if (out.next.dim > dim_target) {
            out.next.dim = std::max(dim_target, out.next.dim - dim_rate);
        }

        // Crowd energy: kicks decay smoothly.
        if (pulse.valid && pulse.kick > 0.0f) {
            out.next.crowd_pulse = std::min(1.0f, out.next.crowd_pulse + pulse.kick);
        }
        out.next.crowd_pulse = std::max(0.0f, out.next.crowd_pulse - dt * 0.45f);

        // Garbage-rain cadence (RAIN phase only; one-shot flag per volley).
        if (out.next.phase == PHASE_RAIN && rain_every > 0.0f) {
            out.next.rain_timer -= dt;
            if (out.next.rain_timer <= 0.0f) {
                out.next.rain_timer = rain_every;
                out.rain_due = true;
            }
        }

        return out;
    }

} // namespace tetris::environment