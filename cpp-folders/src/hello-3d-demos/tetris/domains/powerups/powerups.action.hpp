#pragma once
// tetris/domains/powerups/powerups.action.hpp — INTENT TOKENS (tetris::powerups)
// The pod's input is event-fed (matrix raw facts) plus scripted rulings handed
// over by main; ApplyRulingIntent is the explicit token for "a special locked
// and its ruling arrived" so the reducer stays a pure (state, tokens) function.
#include <domains/powerups/powerups.contract.hpp>

namespace tetris::powerups {

    struct ApplyRulingIntent {
        SpecialRuling ruling{};
        int16_t       lock_x = 0;   // anchor cell of the detonated special
        int16_t       lock_y = 0;
    };

} // namespace tetris::powerups