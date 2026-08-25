-- assets/campaign/campaign.lua - P3 DATA-DRIVEN CAMPAIGN MANIFEST
-- Ordered stage list. Replaces config/campaign/main_campaign.hpp as the
-- source of truth for WHAT content exists and in what order.
--
-- Contract (loaded by game/include/game/stage.hpp):
--   campaign = { stages = { {...}, ... } }
-- Each stage:
--   id            stable string (future save/unlock key)
--   name          display name (HUD / carousel)
--   unlock_after  1-based stage index required first (0 = open)
--   rules         table of Rules overrides merged over config::Rules defaults
--   script        optional path (relative to TETRIS_SOURCE_ROOT) or nil
--
-- Determinism: this file is DATA ONLY - no functions, no randomness.

campaign = {
    stages = {
        {
            id           = "marathon_01",
            name         = "MARATHON",
            unlock_after = 0,
            rules        = {},
            script       = nil,
        },
        {
            id           = "blitz_120",
            name         = "BLITZ 120",
            unlock_after = 1,
            rules = {
                mode_id      = 2,      -- MODE_BLITZ_120
                time_limit   = 120.0,
                target_score = 20000,
                camera_eye   = { 0.0, 11.6, -24.5 },
                camera_target= { 0.0, 10.2, 0.0 },
                camera_fov   = 55.0,
            },
            script       = "domains/progression/scripts/blitz_mode.lua",
        },
        {
            id           = "garbage_canyon",
            name         = "GARBAGE CANYON",
            unlock_after = 2,
            rules = {
                mode_id      = 3,      -- MODE_GARBAGE_CANYON
                time_limit   = 180.0,
                target_lines = 20,
                target_score = 0,
                camera_eye   = { 0.0, 13.5, -28.0 },
                camera_target= { 0.0, 10.2, 0.0 },
                camera_fov   = 58.0,
            },
            script       = "domains/matrix/scripts/garbage_canyon.gen.lua",
        },
        {
            id           = "cyber_storm",
            name         = "CYBER STORM",
            unlock_after = 3,
            rules = {
                mode_id      = 4,      -- MODE_CYBER_STORM
                special_every_n = 5,
                freeze_seconds  = 5.0,
            },
            script       = "domains/powerups/scripts/cyber_storm.lua",
        },
        {
            id           = "encore_finale",
            name         = "ENCORE FINALE",
            unlock_after = 4,
            rules        = {},
            script       = "domains/environment/scripts/encounter_overseer.lua",
        },
    },
}
