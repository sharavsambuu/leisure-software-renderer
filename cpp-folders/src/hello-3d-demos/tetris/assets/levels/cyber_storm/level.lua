-- tetris/assets/levels/CYBER STORM/level.lua - LEVEL DEFINITION (P3: data-driven)
-- Rules overrides merged over config::Rules defaults by the stage loader.
-- Camera fields are optional; omitted fields keep engine defaults.

Level = {
    name         = "CYBER STORM",
    rules_overrides = {
        mode_id          = 4,
        time_limit       = 0.0,
        target_lines     = 24,
        target_score     = 0,
        special_every_n  = 5,
        freeze_seconds   = 5.0,
        camera           = { eye = { 0.0, 9.8, -23.5 }, target = { 0.0, 11.2, 0.0 }, fov_deg = 56.0 },
    },
}
