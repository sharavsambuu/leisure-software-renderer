#pragma once
// game/step.hpp - THE TICK CORE (P1a: deterministic sim slice)
//
// step_core() advances the DETERMINISTIC simulation one frame:
//   restart latch -> freeze gate -> boot queue -> matrix -> progression ->
//   mood wires -> run-end latch -> fx step -> hud step.
//
// Still in main() until P1b (needs ScriptHost plumbing):
//   - environment overseer scripting (lua call_decide_phase / on_event)
//   - L4 powerup script rulings (call_on_special_lock / decide_spawn)
//   - audio edge mapping (platform effect)
//
// Determinism: everything here is pure given inputs; headless runs replay
// byte-identically (verify.sh gates prove it).
#include <algorithm>
#include <memory_resource>
#include <span>
#include <vector>

#include <game/world.hpp>
#include <domains/matrix/matrix.action.hpp>
#include <domains/matrix/matrix.reducer.hpp>
#include <domains/session/session.reducer.hpp>
#include <domains/progression/progression.reducer.hpp>
#include <domains/powerups/powerups.reducer.hpp>
#include <domains/mission/mission.contract.hpp>
#include <domains/environment/environment.reducer.hpp>
#include <domains/spatial_fx/spatial_fx.reducer.hpp>
#include <edges/ui/tetris.hud.hpp>

namespace tetris::game {

    // Stack height projection (moved verbatim from main so both share it).
    inline int compute_stack_height(const matrix::MatrixSnapshot& world) {
        for (int y = 0; y < matrix::GRID_H; ++y) {
            for (int x = 0; x < matrix::GRID_W; ++x) {
                if (world.grid[y][x] != 0) {
                    return matrix::GRID_H - y;   // topmost filled row downward
                }
            }
        }
        return 0;
    }

    // Minimal audio surface: main wires this to the synth edge; tests pass a
    // no-op. Keeps the tick free of SDL/audio includes. Sound ids are the
    // audio::SND_* enum values (plain ints here by design).
    struct IAudioSink {
        virtual ~IAudioSink() = default;
        virtual void play(int sound_id) = 0;
    };

    // Script host interface: wraps the Lua evaluator edge so step_core never
    // sees lua_State*. Null pointer or valid()==false => native C++ rules.
    struct IScriptHost {
        virtual ~IScriptHost() = default;
        virtual bool valid() const = 0;

        // L4 cyber storm hooks
        virtual bool has_special_lock() const = 0;
        virtual powerups::SpecialRuling on_special_lock(
            int special_type, int lock_x, int lock_y,
            const matrix::CellGrid& grid) = 0;
        virtual bool has_decide_spawn() const = 0;
        virtual powerups::SpawnDecision decide_spawn(
            int pieces_since, int armed_next) = 0;

        // L5 encounter overseer hooks
        virtual environment::CrowdPulse on_event(int event_kind, int value) = 0;
        virtual environment::OverseerRuling decide_phase(
            int phase, float phase_time, int lines_cleared, bool danger) = 0;

        // Part 7 G1: scripted GOAL evaluation (SCRIPTING.md section 2).
        // goal_table names a global Lua table (e.g. "Goals") whose function
        // `test` receives (events, snapshot) as plain tables and returns
        // truthy to complete the goal this tick.
        virtual bool has_goal_test(const char* goal_table) const = 0;
        virtual bool evaluate_goal(const char* goal_table,
                const std::vector<tetris::mission::MissionEventView>& events,
                const tetris::mission::MissionSnapshot& snap) = 0;
    };

    struct FrameInput {
        std::span<const matrix::TetrisCommand> commands{};
        bool soft_drop_held = false;
    };

    struct StepContext {
        float dt = 1.0f / 60.0f;
        int   frame = 0;                  // deterministic rain-hole parity
        bool  headless = false;
        std::pmr::memory_resource* arena = std::pmr::get_default_resource();
        IAudioSink*  audio   = nullptr;   // null => silent (headless/tests)
        IScriptHost* scripts = nullptr;   // null/invalid => native rules
        float rain_every = 0.0f;          // L5 encounter rain cadence (0 = off)
    };

    struct CoreStepResult {
        const std::pmr::vector<matrix::MatrixEvent>*      matrix_events = nullptr;
        const std::pmr::vector<progression::ProgressionEvent>* prog_events = nullptr;
        bool restart_requested = false;
        bool phase_changed = false;   // L5 transition happened this tick
    };

    // Advances the deterministic core. boot_commands: in/out queue (L3 stamp,
    // L4 special queue, L5 rain ride here). hud: presentation transient state.
    inline void step_core(GameWorld& w,
                          const FrameInput& in,
                          StepContext& ctx,
                          std::vector<matrix::TetrisCommand>& boot_commands,
                          ui::HudState& hud,
                          CoreStepResult& out) {
        const config::Rules& rules = w.rules;
        auto* mr = ctx.arena;

        // ---- restart latch ---------------------------------------------------
        bool restart_requested = false;
        for (const auto& cmd : in.commands)
            if (std::holds_alternative<matrix::RestartIntent>(cmd))
                restart_requested = true;
        out.restart_requested = restart_requested;

        // Blitz time-up freezes the run; a restart press thaws that frame.
        const bool frozen = w.score.time_up && !restart_requested;
        auto cmd_span = frozen
            ? std::span<const matrix::TetrisCommand>()
            : in.commands;
        const float matrix_dt = frozen ? 0.0f : ctx.dt;

        // Gravity cadence from progression level through pure config math.
        w.matrix_state.drop_interval = rules.gravity_for_level(w.score.level);

        // Boot queue first (L3 stamp / L4 special queue / L5 rain volleys).
        if (!boot_commands.empty()) {
            std::vector<matrix::TetrisCommand> merged;
            merged.reserve(boot_commands.size() + cmd_span.size());
            for (auto& c : boot_commands) merged.push_back(std::move(c));
            boot_commands.clear();
            for (const auto& c : cmd_span) merged.push_back(c);
            cmd_span = std::span<const matrix::TetrisCommand>(
                merged.data(), merged.size());
        }

        // Part 6 input feel: fold held soft-drop flag into intents.
        std::vector<matrix::TetrisCommand> folded(cmd_span.begin(), cmd_span.end());
        if (!frozen && in.soft_drop_held) folded.push_back(matrix::SoftDropIntent{});
        const auto folded_span = std::span<const matrix::TetrisCommand>(
            folded.data(), folded.size());

        // ---- PURE SIMULATION CORE ------------------------------------------
        matrix::MatrixStepResult mstep =
            matrix::reduce_matrix(w.matrix_state, folded_span, matrix_dt, mr);
        w.matrix_state = std::move(mstep.next_state);
        out.matrix_events = &mstep.events;

        // ---- EVENT-FED PROGRESSION ------------------------------------------
        if (restart_requested) {
            const int preserved_high = w.score.high_score;
            w.session_high = std::max(w.session_high, preserved_high);
            w.score = progression::ScoreState{};
            w.score.target_score = rules.target_score;
            w.score.mode_id      = rules.mode_id;
            w.score.time_left    = rules.time_limit;
            w.score.high_score   = preserved_high;
            hud = ui::HudState{};                       // no stale banners
            w.powerups_state = powerups::PowerupSnapshot{};
            w.env            = environment::EnvironmentSnapshot{};
        }
        progression::ProgressionStep prog = progression::reduce_progression(
            std::span<const matrix::MatrixEvent>(mstep.events.data(),
                                                 mstep.events.size()),
            w.score, rules, mr, ctx.dt,
            compute_stack_height(w.matrix_state));
        w.score = std::move(prog.next);
        out.prog_events = &prog.events;

        // ---- MOOD WIRES (planner consumes directly) -------------------------
        w.fx.mood_intensity =
            (rules.time_limit > 0.0f &&
             rules.mode_id != config::MODE_GARBAGE_CANYON)
                ? glm::clamp(
                      1.0f - w.score.time_left / rules.time_limit, 0.0f, 1.0f)
                : 0.0f;
        w.fx.env_dusk   =
            (rules.mode_id == config::MODE_GARBAGE_CANYON) ? 1.0f : 0.0f;
        w.fx.env_neon   =
            (rules.mode_id == config::MODE_CYBER_STORM) ? 1.0f : 0.0f;
        w.fx.env_finale =
            (rules.mode_id == config::MODE_ENCORE_FINALE) ? 1.0f : 0.0f;

        // ---- L5 ENCOUNTER OVERSEER (script or native) ------------------------
        if (w.fx.env_finale > 0.5f) {
            environment::CrowdPulse pulse{};
            for (const auto& ev : mstep.events) {
                if (ev.type == matrix::MatrixEventType::LINES_CLEARED) {
                    pulse = ctx.scripts && ctx.scripts->valid()
                        ? ctx.scripts->on_event(1, (int)ev.lines_cleared_count)
                        : environment::CrowdPulse{
                              true, 0.25f + 0.15f * ev.lines_cleared_count };
                }
            }
            for (const auto& pev : prog.events) {
                if (pev.type ==
                    progression::ProgressionEventType::OBJECTIVE_COMPLETED)
                    pulse = ctx.scripts && ctx.scripts->valid()
                        ? ctx.scripts->on_event(2, 1)
                        : environment::CrowdPulse{ true, 1.0f };
            }

            const int stack_h = compute_stack_height(w.matrix_state);
            const bool danger = stack_h >= matrix::VISIBLE_H - 4;
            const environment::OverseerRuling ruling =
                (ctx.scripts && ctx.scripts->valid())
                    ? ctx.scripts->decide_phase(
                          w.env.phase, w.env.phase_time,
                          w.score.lines_cleared, danger)
                    : environment::OverseerRuling{};   // no script => static CALM

            const environment::EnvironmentStepResult estep =
                environment::reduce_environment(w.env, ruling, pulse, ctx.dt,
                                                ctx.rain_every);
            w.env = estep.next;

            // Rain cadence: deterministic hole choice from frame parity.
            if (estep.rain_due) {
                matrix::AddGarbageRowsIntent rain;
                rain.rows = ctx.rain_every > 0.0f ? 1u : 1u;
                for (int r = 0; r < (int)rain.rows; ++r) {
                    rain.hole_x[r] = (uint8_t)((ctx.frame * 7 + r * 5)
                                               % matrix::GRID_W);
                }
                boot_commands.push_back(rain);
                hud.flash = std::max(hud.flash, 0.20f);
            }

            // Plain-value wires into FX.
            w.fx.mood_phase   = w.env.mood;
            w.fx.dim          = w.env.dim;
            w.fx.ghost_hidden = w.env.dim > 0.5f;
            w.fx.crowd_pulse  = w.env.crowd_pulse;
            w.fx.finale_phase = w.env.phase;

            // Phase-transition set pieces (audio via sink).
            if (estep.phase_changed) {
                w.fx.screen_flash = std::max(w.fx.screen_flash, 0.85f);
                switch (w.env.phase) {
                case environment::PHASE_RAIN:
                    hud.spawn_floater("GARBAGE RAIN",
                        shs::render::Color{ 255, 160, 60, 255 }, 2.2f);
                    if (ctx.audio) ctx.audio->play(11 /*SND_THUD*/);
                    break;
                case environment::PHASE_BLACKOUT:
                    hud.spawn_floater("BLACKOUT",
                        shs::render::Color{ 140, 150, 220, 255 }, 2.2f);
                    break;
                case environment::PHASE_CRESCENDO:
                    hud.spawn_floater("FINALE",
                        shs::render::Color{ 255, 210, 60, 255 }, 2.6f);
                    if (ctx.audio) ctx.audio->play(5 /*SND_TETRIS_FOUR*/);
                    break;
                default: break;
                }
            }
        }

        // ---- L4 POWERUP SCHEDULER (event-fed; script rulings as values) -----
        {
            std::pmr::vector<powerups::ApplyRulingIntent> frame_rulings(mr);
            if (ctx.scripts && ctx.scripts->valid()
                && ctx.scripts->has_special_lock()) {
                for (const auto& ev : mstep.events) {
                    if (ev.type != matrix::MatrixEventType::SPECIAL_LOCKED) continue;
                    auto ruling = ctx.scripts->on_special_lock(
                        ev.special_type, ev.lock_x, ev.lock_y,
                        w.matrix_state.grid);
                    if (!ruling.valid) continue;
                    frame_rulings.push_back(powerups::ApplyRulingIntent{
                        ruling, static_cast<int16_t>(ev.lock_x),
                        static_cast<int16_t>(ev.lock_y) });
                    if (ruling.fx_id == 1 || ruling.fx_id == 2) {
                        hud.flash = std::max(hud.flash,
                            ruling.fx_id == 1 ? 0.55f : 0.35f);
                    }
                }
            }
            powerups::PowerupStep pstep = powerups::reduce_powerups(
                w.powerups_state,
                std::span<const matrix::MatrixEvent>(mstep.events.data(),
                                                     mstep.events.size()),
                std::span<const powerups::ApplyRulingIntent>(
                    frame_rulings.data(), frame_rulings.size()),
                rules.special_every_n, rules.freeze_seconds, mr);
            w.powerups_state = std::move(pstep.next);
            for (auto& mut : pstep.mutations)
                boot_commands.push_back(std::move(mut));

            // Fulfill latched special request once via decide_spawn().
            bool spawn_requested_now = false;
            for (const auto& uev : pstep.events) {
                if (uev.type ==
                    powerups::PowerupEventType::SPAWN_SPECIAL_REQUESTED)
                    spawn_requested_now = true;
            }
            if (spawn_requested_now && ctx.scripts && ctx.scripts->valid()
                && ctx.scripts->has_decide_spawn()) {
                const auto dec = ctx.scripts->decide_spawn(
                    (int)pstep.next.pieces_since_special,
                    (int)pstep.next.armed_next);
                if (dec.valid) {
                    matrix::QueueSpecialIntent q;
                    q.special_type = dec.special_type;
                    boot_commands.push_back(q);
                }
            }

            // Powerup audio mapping.
            if (ctx.audio) {
                for (const auto& uev : pstep.events) {
                    switch (uev.type) {
                    case powerups::PowerupEventType::POWERUP_TRIGGERED:
                        ctx.audio->play(
                            uev.powerup == 1 ? 12 /*SND_BLAST*/
                          : uev.powerup == 2 ? 13 /*SND_ZAP*/
                                             : 14 /*SND_FROST*/);
                        break;
                    default: break;
                    }
                }
            }
        }

        // ---- RUN-END LATCH ---------------------------------------------------
        if (w.score.victory || w.score.time_up || w.matrix_state.game_over) {
            w.session.run_victory     = w.score.victory;
            w.session.run_time_up     = w.score.time_up;
            w.session.final_score     = w.score.score;
            w.session.final_lines     = w.score.lines_cleared;
            w.session.final_max_combo = w.score.max_combo;
            w.session.final_seconds   = w.matrix_state.game_time;
            if (w.score.victory) {
                w.session.unlocked_stages = std::min(
                    w.session.stage_count,
                    std::max(w.session.unlocked_stages,
                             w.session.current_stage + 2));
            }
            w.session.cursor = 0;
            w.session.screen = session::Screen::RESULTS;
        }

        // ---- FX STEP ---------------------------------------------------------
        spatial_fx::step_fx(w.fx,
            std::span<const matrix::MatrixEvent>(mstep.events.data(),
                                                 mstep.events.size()),
            std::span<const progression::ProgressionEvent>(
                prog.events.data(), prog.events.size()),
            ctx.dt);

        // ---- HUD STEP --------------------------------------------------------
        ui::step_hud(hud,
            std::span<const progression::ProgressionEvent>(
                prog.events.data(), prog.events.size()),
            ctx.dt,
            std::span<const matrix::MatrixEvent>(mstep.events.data(),
                                                 mstep.events.size()));
    }

} // namespace tetris::game
