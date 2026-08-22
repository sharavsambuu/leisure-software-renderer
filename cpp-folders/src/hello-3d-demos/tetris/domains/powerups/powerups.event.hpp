#pragma once
// tetris/domains/powerups/powerups.event.hpp — DERIVED EVENTS (tetris::powerups)
// Consumed by main for audio mapping and by the HUD/FX wiring bundles.
#include <cstdint>

namespace tetris::powerups {

    enum class PowerupEventType : uint8_t {
        POWERUP_TRIGGERED        = 0,  // .powerup = PowerupType that detonated
        SPAWN_SPECIAL_REQUESTED  = 1   // cadence reached; main asks the script
    };

    struct PowerupEvent {
        PowerupEventType type;
        uint8_t          powerup = 0;   // PowerupType value
    };

} // namespace tetris::powerups