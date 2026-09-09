-- tetris/assets/levels/BLITZ 120/level.lua - LEVEL DEFINITION (P3: data-driven)
-- Rules overrides merged over config::Rules defaults by the stage loader.
-- Camera fields are optional; omitted fields keep engine defaults.

Level = {
    name         = "BLITZ 120",
    rules_overrides = {
        mode_id      = 2,
        time_limit   = 120.0,
        target_score = 20000,
        camera       = { eye = { 0.0, 11.6, -24.5 }, target = { 0.0, 10.2, 0.0 }, fov_deg = 55.0 },
    },
}
