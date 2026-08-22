#pragma once
// tetris/domains/powerups/powerups.contract.hpp — POWERUP STATE VOCABULARY
// (tetris::powerups) Pod 4: cooldown/counter state for the scripted special
// pieces, plus the plain-value ruling struct that crosses the Lua boundary.
// No platform I/O, no rendering, no grid access (Constitution II).
#include <cstdint>

namespace tetris::powerups {

    // Special-piece identity (mirrors matrix PieceType values 9..11; kept as
    // a pod-local enum so this contract stays decoupled from the grid schema).
    enum class PowerupType : uint8_t {
        None = 0, Bomb = 1, Laser = 2, Freeze = 3
    };

    // Scripted lock ruling (plain values). The stage script returns this shape
    // from CyberRules.on_special_lock(); main bridges it into matrix commands.
    // fx_id: 1 = blast, 2 = laser sweep, 3 = frost (spatial_fx recipe selector).
    static constexpr int RULING_MAX_CELLS = 32;
    struct SpecialRuling {
        bool    valid          = false;
        uint8_t clear_count    = 0;
        int16_t clear_x[RULING_MAX_CELLS]{};
        int16_t clear_y[RULING_MAX_CELLS]{};
        float   freeze_seconds = 0.0f;
        uint8_t fx_id          = 0;
    };

    // Spawn decision returned by CyberRules.decide_spawn() (0 = no special).
    struct SpawnDecision {
        bool    valid        = false;
        uint8_t special_type = 0;   // matrix PieceType value (9..11)
    };

    struct PowerupSnapshot {
        // Scheduler: every Nth spawn is special; armed_next cycles Bomb→Laser→
        // Freeze. request_open latches until main fulfills it via the script.
        int         pieces_since_special = 0;
        PowerupType armed_next           = PowerupType::Bomb;
        bool        request_open         = false;

        // Live-effect mirror for HUD projection (main wires freeze_left from
        // the matrix snapshot's gravity_freeze each frame — plain-field duty).
        float       freeze_left          = 0.0f;

        // Lifetime counters (RESULTS flavor / future stats).
        int blasts_fired = 0;
        int lasers_fired = 0;
        int freezes_used = 0;
    };

} // namespace tetris::powerups