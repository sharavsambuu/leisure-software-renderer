#pragma once
// tetris/config/rules.hpp — PURE TUNING DATA (tetris::config::Rules)
// Every gameplay number lives here; pods read, never mutate.
#include <cmath>
#include <cstdint>

namespace tetris::config {

    // Mode identity (campaign stages; see config/campaign/main_campaign.hpp)
    enum : int {
        MODE_MARATHON   = 1,   // L1: untimed target chase (pure C++ tier)
        MODE_BLITZ_120  = 2    // L2: 2-minute sprint (Lua-authored economy)
    };

    struct Rules {
        // Gravity
        float initial_drop_interval = 0.80f;
        float min_drop_interval     = 0.08f;
        float gravity_decay         = 0.85f;   // per level
        float soft_drop_factor      = 0.12f;   // fraction of interval

        // Lock discipline
        float lock_delay      = 0.5f;
        int   max_lock_resets = 15;

        // Scoring (Lua-swappable rule inputs — see progression.reducer)
        int base_scores[5]           = { 0, 100, 300, 500, 800 };
        int combo_bonus              = 50;   // x combo x level
        int hard_drop_score_per_cell = 2;
        int soft_drop_score_per_cell = 1;
        int lines_per_level          = 10;

        // Objective
        int      target_score = 12000;
        uint32_t rng_seed     = 0x9e3779b9u;

        // Mode identity + blitz clock (L2). time_limit == 0 means untimed.
        // A wired Lua script may override mode_id/target_score/time_limit at
        // boot via BlitzRules.get_config() (see edges/lua/lua.edge.hpp).
        int      mode_id     = MODE_MARATHON;
        float    time_limit  = 0.0f;

        float gravity_for_level(int level) const {
            return std::max(min_drop_interval,
                            initial_drop_interval * std::pow(gravity_decay, (float)(level - 1)));
        }
    };

} // namespace tetris::config
