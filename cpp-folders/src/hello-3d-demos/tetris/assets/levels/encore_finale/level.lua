-- tetris/assets/levels/ENCORE FINALE/level.lua - LEVEL DEFINITION (P3: data-driven)
-- Rules overrides merged over config::Rules defaults by the stage loader.
-- Camera fields are optional; omitted fields keep engine defaults.

Level = {
    name         = "ENCORE FINALE",
    rules_overrides = {
        mode_id          = 5,
        time_limit       = 0.0,
        target_lines     = 20,
        target_score     = 0,
        special_every_n  = 0,
        freeze_seconds   = 0.0,
        camera           = { eye = { 1.5, 14.0, -30.0 }, target = { 0.0, 8.5, 0.0 }, fov_deg = 62.0 },
    },
}
