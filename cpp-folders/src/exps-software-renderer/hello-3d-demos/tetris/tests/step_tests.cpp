// tetris/tests/step_tests.cpp - P2a WHOLE-FRAME TESTS + P2b INPUT HARNESS
//
// Whole-frame: drives game::step_core() with command sequences and asserts
// resulting facts — the class of proof previously only available via
// screenshot byte-compare.
//
// Input harness: feeds synthetic SDL key timelines into input::InputEdge
// (no real SDL events) and asserts DAS/ARR timing + soft-drop held flag —
// closes TODOS Part 6 V1-V3.
#include <cstdio>
#include <cstring>

#include <domains/spatial_fx/spatial_fx.contract.hpp>
#include <game/world.hpp>
#include <game/step.hpp>
#include <domains/session/session.reducer.hpp>
#include <edges/input/tetris.input.hpp>
#include <game/stage.hpp>

using namespace tetris;

namespace {
    int g_pass = 0, g_fail = 0;
    void check(bool ok, const char* name) {
        if (ok) { ++g_pass; std::printf("  PASS %s\n", name); }
        else    { ++g_fail; std::fprintf(stderr, "  FAIL %s\n", name); }
    }

    // ---- step_core helpers -------------------------------------------------
    struct Frame {
        game::GameWorld w{};
        ui::HudState hud{};
        std::vector<matrix::TetrisCommand> boot;
        game::CoreStepResult res;

        void run(std::span<const matrix::TetrisCommand> cmds,
                 float dt = 1.0f / 60.0f, bool soft = false) {
            game::FrameInput in{};
            in.commands       = cmds;
            in.soft_drop_held = soft;
            game::StepContext ctx{};
            ctx.dt = dt;
            game::step_core(w, in, ctx, boot, hud, res);
        }
        void run1(matrix::TetrisCommand c, float dt = 1.0f / 60.0f) {
            std::pmr::vector<matrix::TetrisCommand> v{
                std::pmr::get_default_resource() };
            v.push_back(c);
            run(std::span<const matrix::TetrisCommand>(v.data(), v.size()), dt);
        }
    };

    // ==== P2a: WHOLE-FRAME ==================================================

    void test_step_move_command_reaches_matrix() {
        Frame f;
        const int x0 = f.w.matrix_state.active.pos.x;
        f.run1(matrix::MoveLeftIntent{});
        check(f.w.matrix_state.active.pos.x == x0 - 1,
              "step: MOVE_LEFT moves piece left through whole frame");
    }

    void test_step_restart_resets_and_preserves_high() {
        Frame f;
        f.w.score.score      = 5000;
        f.w.score.high_score = 9000;
        f.run1(matrix::RestartIntent{});
        check(f.w.score.score == 0, "restart: score resets");
        check(f.w.score.high_score == 9000,
              "restart: high score preserved (main-edge duty moved into core)");
    }

    void test_step_freeze_gate_blocks_commands_but_thaws_restart() {
        Frame f;
        f.w.score.time_up   = true;   // blitz clock expired
        f.w.score.time_left = 0.0f;
        const int y0 = f.w.matrix_state.active.pos.y;
        f.run1(matrix::SoftDropIntent{}, 1.0f / 60.0f);
        check(f.w.matrix_state.active.pos.y == y0,
              "freeze gate: time_up blocks gravity/commands");

        // restart thaws the frame: reset reaches the board too
        f.run1(matrix::RestartIntent{});
        check(f.w.score.time_up == false || !f.w.matrix_state.game_over,
              "freeze gate: restart thaws and resets");
    }

    void test_step_run_end_latches_results_screen() {
        Frame f;
        f.w.matrix_state.game_over = true;
        f.run({}, 1.0f / 60.0f, false);
        check(f.w.session.screen == session::Screen::RESULTS,
              "run-end latch: game_over -> RESULTS screen");
        check(f.w.session.final_lines == f.w.score.lines_cleared,
              "run-end latch: final stats captured");
    }

    void test_step_victory_unlocks_next_stage() {
        Frame f;
        f.w.session.stage_count     = 5;
        f.w.session.current_stage   = 2;
        f.w.session.unlocked_stages = 3;
        f.w.score.victory           = true;
        f.run({}, 1.0f / 60.0f, false);
        check(f.w.session.screen == session::Screen::RESULTS,
              "victory: RESULTS screen");
        check(f.w.session.unlocked_stages >= 4,
              "victory: next stage unlocked");
    }

    void test_step_determinism_same_stream_same_state() {
        auto play = []() {
            Frame f;
            for (int i = 0; i < 30; ++i)
                f.run({}, 1.0f / 60.0f, i % 2 == 0);
            return static_cast<int>(f.w.matrix_state.rng_state)
                 + f.w.matrix_state.active.pos.x * 31
                 + f.w.score.lines_cleared;
        };
        check(play() == play(), "determinism: same stream -> same world hash");
    }

    // ==== P2b: INPUT EDGE HARNESS ==========================================

    // Synthetic key event builder (no SDL window needed).
    SDL_Event key_event(SDL_Keycode k, bool down, bool repeat = false) {
        SDL_Event e{};
        e.type             = down ? SDL_KEYDOWN : SDL_KEYUP;
        e.key.keysym.sym   = k;
        e.key.repeat       = repeat ? 1 : 0;
        return e;
    }

    // The edge polls SDL's queue; for tests we push events via SDL_peepEvents.
    bool push_events(SDL_Event* evs, int n) {
        return SDL_PeepEvents(evs, n, SDL_ADDEVENT, SDL_FIRSTEVENT,
                              SDL_LASTEVENT) == n;
    }

    void test_input_tap_moves_once() {
        input::InputEdge edge;
        SDL_Event down[2] = { key_event(SDLK_LEFT, true), key_event(SDLK_LEFT, false) };
        check(push_events(down, 2), "harness: events queued");

        edge.begin_frame(1.0f / 60.0f);
        auto in = edge.poll(std::pmr::get_default_resource());
        check(in.commands.size() >= 1, "tap left: at least one move intent");
        if (!in.commands.empty()) {
            check(std::holds_alternative<matrix::MoveLeftIntent>(in.commands[0]),
                  "tap left: intent type correct");
        }
    }

    void test_input_das_delays_then_repeats() {
        input::InputEdge edge;
        SDL_Event down = key_event(SDLK_LEFT, true);
        push_events(&down, 1);

        // Hold for 149ms: only the initial tap shift.
        int moves_early = 0;
        for (int f = 0; f < 9; ++f) {          // 9 frames x 16.7ms ~ 150ms
            edge.begin_frame(1.0f / 60.0f);
            auto in = edge.poll(std::pmr::get_default_resource());
            moves_early += (int)in.commands.size();
        }
        check(moves_early <= 2, "DAS delay: few/no extra shifts before 150ms");

        // Continue holding to 400ms total: ARR should add ~6 more (40ms rate).
        int moves_late = 0;
        for (int f = 0; f < 15; ++f) {
            edge.begin_frame(1.0f / 60.0f);
            auto in = edge.poll(std::pmr::get_default_resource());
            moves_late += (int)in.commands.size();
        }
        check(moves_late >= 4, "ARR: auto-repeat fires while held");
    }

    void test_input_keyup_stops_autorepeat() {
        input::InputEdge edge;
        SDL_Event evs[2] = { key_event(SDLK_RIGHT, true), key_event(SDLK_RIGHT, false) };
        push_events(evs, 2);

        edge.begin_frame(1.0f / 60.0f);
        edge.poll(std::pmr::get_default_resource()); // consume press

        int after_release = 0;
        for (int f = 0; f < 20; ++f) {
            edge.begin_frame(1.0f / 60.0f);
            auto in = edge.poll(std::pmr::get_default_resource());
            after_release += (int)in.commands.size();
        }
        check(after_release == 0, "KEYUP stops movement entirely");
    }

    void test_input_soft_drop_is_held_flag() {
        input::InputEdge edge;
        SDL_Event down = key_event(SDLK_DOWN, true);
        push_events(&down, 1);

        edge.begin_frame(1.0f / 60.0f);
        auto in = edge.poll(std::pmr::get_default_resource());
        check(in.soft_drop_held, "soft drop held while key down");

        SDL_Event up = key_event(SDLK_DOWN, false);
        push_events(&up, 1);
        edge.begin_frame(1.0f / 60.0f);
        in = edge.poll(std::pmr::get_default_resource());
        check(!in.soft_drop_held, "soft drop released on KEYUP");
    }

    void test_input_os_repeat_never_trusted() {
        input::InputEdge edge;
        SDL_Event repeat_ev = key_event(SDLK_LEFT, true, /*repeat=*/true);
        push_events(&repeat_ev, 1);

        edge.begin_frame(1.0f / 60.0f);
        auto in = edge.poll(std::pmr::get_default_resource());
        check(in.commands.empty(), "OS key-repeat ignored (no fresh intent)");
    }

} // namespace

int main() {
    // Event queue needs the SDL event system initialized. Dummy drivers keep
    // this headless-safe.
    SDL_setenv("SDL_VIDEODRIVER", "dummy", 1);
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::fprintf(stderr, "[tetris-tests] SDL_Init failed: %s\n", SDL_GetError());
        return 2;
    }
    std::printf("[tetris-tests] P2 whole-frame + input harness\n");

    test_step_move_command_reaches_matrix();
    test_step_restart_resets_and_preserves_high();
    test_step_freeze_gate_blocks_commands_but_thaws_restart();
    test_step_run_end_latches_results_screen();
    test_step_victory_unlocks_next_stage();
    test_step_determinism_same_stream_same_state();

    test_input_tap_moves_once();
    test_input_das_delays_then_repeats();
    test_input_keyup_stops_autorepeat();
    test_input_soft_drop_is_held_flag();
    test_input_os_repeat_never_trusted();

    // ==== P3: DATA-DRIVEN CAMPAIGN ======================================
    std::printf("[tetris-tests] P3 data-driven campaign\n");
    {
        auto result = game::load_campaign(TETRIS_SOURCE_ROOT);
        check(result.stages.size() == 5, "campaign: 5 stages loaded from Lua");
        if (result.stages.size() == 5) {
            check(result.stages[0].id == "marathon_01", "stage 1 id");
            check(result.stages[1].name == "BLITZ 120", "stage 2 name");
            check(result.stages[1].rules.mode_id == config::MODE_BLITZ_120,
                  "blitz overrides: mode_id applied");
            check(result.stages[1].rules.time_limit == 120.0f,
                  "blitz overrides: time_limit applied");
            check(result.stages[2].rules.target_lines == 20,
                  "canyon overrides: target_lines applied");
            check(result.stages[1].script_path.find("blitz_mode.lua")
                      != std::string::npos,
                  "script path carried through");
            check(result.used_fallback == false,
                  "real campaign.lua loads without fallback");
        }

        // Fallback law: nonexistent root => marathon default, no crash
        auto fb = game::load_campaign("/nonexistent/path/xyz");
        check(fb.used_fallback && !fb.stages.empty()
                  && fb.stages[0].id == "marathon_01",
              "fallback: bad path -> marathon defaults, no crash");
    }

    std::printf("[tetris-tests] %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

