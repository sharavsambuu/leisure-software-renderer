#pragma once
// tetris/domains/environment/environment.contract.hpp — POD 5 VOCABULARY
// (tetris::environment) Pure value types for the reactive environment: the
// encounter phase state, the mood/dim interpolation targets, and the plain
// rulings the scripted overseer returns. No platform I/O, no rendering.
#include <cstdint>

namespace tetris::environment {

    // L5 Encore Finale encounter phases (script-authored choreography):
    enum : int {
        PHASE_CALM      = 1,   // normal play — learn the stage
        PHASE_RAIN      = 2,   // garbage volleys on a cadence
        PHASE_BLACKOUT  = 3,   // dimmed board, ghost hidden
        PHASE_CRESCENDO = 4    // victory push — gold mood + crowd frenzy
    };

    // Plain ruling returned by Encounter.decide_phase() (value-in/value-out).
    struct OverseerRuling {
        bool  valid       = false;
        int   new_phase   = 0;     // 0 = keep current phase
        float mood_target = -1.0f; // <0 = keep current target (0 cyan .. 1 gold)
    };

    // Plain pulse returned by Encounter.on_event() (crowd reaction).
    struct CrowdPulse {
        bool  valid = false;
        float kick  = 0.0f;        // added to crowd_pulse (clamped 0..1)
    };

    // Encounter tuning numbers from Encounter.get_config() (plain values).
    struct EncounterConfig {
        bool  valid       = false;
        int   phase_count = 4;
        float rain_every  = 8.0f;  // seconds between garbage volleys
        int   rain_rows   = 1;     // garbage rows per volley
    };

    struct EnvironmentSnapshot {
        int   phase       = PHASE_CALM;
        float phase_time  = 0.0f;  // seconds in the current phase
        float mood        = 0.0f;  // interpolated 0 (cyan) .. 1 (gold)
        float dim         = 0.0f;  // blackout dimming 0..1 (drives ghost-hidden)
        float crowd_pulse = 0.0f;  // decays; light-wave/pedestal energy
        float rain_timer  = 0.0f;  // countdown to next garbage volley (RAIN)
    };

} // namespace tetris::environment