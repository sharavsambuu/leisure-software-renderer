#pragma once
// game/world.hpp - GameWorld: the aggregate of ALL tetris pod states.
//
// Plain data (VOP). One instance owns the whole simulation; swapping this
// struct swaps universes (rollback-ready per ARCHITECTURE.md Part I).
#include <config/rules.hpp>
#include <domains/matrix/matrix.contract.hpp>
#include <domains/progression/progression.contract.hpp>
#include <domains/powerups/powerups.contract.hpp>
#include <domains/environment/environment.contract.hpp>
#include <domains/session/session.contract.hpp>
#include <domains/spatial_fx/spatial_fx.contract.hpp>

namespace tetris::game {

    struct GameWorld {
        matrix::MatrixSnapshot               matrix_state{};
        progression::ScoreState              score{};
        powerups::PowerupSnapshot            powerups_state{};
        environment::EnvironmentSnapshot     env{};
        session::SessionSnapshot             session{};
        spatial_fx::FxState                  fx{ std::pmr::get_default_resource() };

        // Stage context (set at load; read-only during play).
        config::Rules rules{};
        int  stage_index = 0;
        int  session_high = 0;

        // Frame-scratch carried between ticks by main wiring today:
        bool boot_pending = false;   // boot queue nonempty flag mirror
    };

} // namespace tetris::game
