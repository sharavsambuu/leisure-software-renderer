-- tetris/assets/levels/GARBAGE CANYON/level.lua - LEVEL DEFINITION (P3: data-driven)
-- Rules overrides merged over config::Rules defaults by the stage loader.
-- Camera fields are optional; omitted fields keep engine defaults.

Level = {
    name         = "GARBAGE CANYON",
    rules_overrides = {
        mode_id      = 3,
        time_limit   = 180.0,
        target_lines = 20,
        target_score = 0,
        camera       = { eye = { 0.0, 13.5, -28.0 }, target = { 0.0, 10.2, 0.0 }, fov_deg = 58.0 },
    },
}
