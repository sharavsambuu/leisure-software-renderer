// SDL on Windows redefines `main` to `SDL_main`; SDL_MAIN_HANDLED opts out
// so the plain main() below links (matches snake/tetris/plane demos).
#define SDL_MAIN_HANDLED

// ============================================================================
// Hello3DTetris — MAIN ENTRY EDGE
// Owns: SDL lifecycle (window/audio), per-frame PMR arena, loop wiring,
// presentation, and the event→sound map. All simulation/render/HUD logic
// lives in domain pods and execution edges.
//
// Campaign + scripting (L1/L2/L3):
//   --stage=N                 select campaign stage (1 = MARATHON, 2 = BLITZ 120,
//                             3 = GARBAGE CANYON). Windowed mode with --stage>1
//                             jumps straight into that stage's run; without it,
//                             windowed boots to the TITLE menu (pick stages in
//                             the level-select carousel).
//   --script=<file>           override the stage's Lua rule script
//   --seed=N                  L3 canyon board-generator seed (default 20260822;
//                             same seed ⇒ byte-identical pre-ruined board)
//   --expect-target-score=N   smoke gate: assert the wired script overrode the
//                             target score, print SMOKE_TARGET_SCORE=PASS/FAIL
//   --expect-target-lines=N   L3 smoke gate: assert target_lines override,
//                             print SMOKE_TARGET_LINES=PASS/FAIL
//
// Session layer (M1): TITLE / LEVEL_SELECT / PLAYING / PAUSED / RESULTS state
// machine in domains/session/. Windowed keys: W/S/A/D navigate, ENTER/SPACE
// confirm, ESC pauses (in run) or goes back (menus), P pause toggle,
// M sound on/off. Quit via the ГАРАХ menu row or closing the window.
//
// Headless verification hooks (deterministic, display-less):
//   --screenshot <path.bmp>   render N frames, save BMP, exit (no window)
//   --frame=N                 frame count for the above (default 60)
//   --autodrive-harddrop      inject ONE synthetic HardDropIntent at frame 30
//
// Campaign flow (M2): a finished run lands on the RESULTS screen; its first
// row is ДАРААГИЙН ҮЕ after a victory (next stage) or ДАХИН ЭХЛҮҮЛЭХ otherwise.
// Every stage load goes through load_stage(): FULL state reset (board/score/
// HUD/FX/fresh script sandbox/window title) — no stale GUI across levels.
// ============================================================================

#include <SDL2/SDL.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <memory_resource>
#include <span>
#include <string>
#include <thread>

#include "shs_renderer.hpp"

#ifndef TETRIS_SOURCE_ROOT
#define TETRIS_SOURCE_ROOT "."
#endif

#include <config/rules.hpp>
#include <domains/powerups/powerups.reducer.hpp>
#include <config/campaign/main_campaign.hpp>

#include <domains/matrix/matrix.contract.hpp>
#include <domains/matrix/matrix.action.hpp>
#include <domains/matrix/matrix.reducer.hpp>
#include <domains/progression/progression.reducer.hpp>
#include <domains/spatial_fx/spatial_fx.reducer.hpp>
#include <domains/spatial_fx/spatial_fx.plan.hpp>
#include <domains/environment/environment.reducer.hpp>
#include <domains/session/session.reducer.hpp>

#include <edges/input/tetris.input.hpp>
#include <edges/audio/tetris.audio.hpp>
#include <edges/rasterizer/tetris.rasterizer.hpp>
#include <edges/ui/tetris.hud.hpp>
#include <edges/lua/lua.edge.hpp>

namespace {

    using namespace tetris;

    // Audio synth instance lives at file scope: the SDL callback thread
    // dereferences it for the lifetime of the audio device.
    audio::TetrisAudioSynth g_audio;

    constexpr int CANVAS_WIDTH  = 1280;
    constexpr int CANVAS_HEIGHT = 720;
    constexpr int TILE_SIZE_X   = 80;
    constexpr int TILE_SIZE_Y   = 80;

    unsigned thread_count() {
        const unsigned hw = std::thread::hardware_concurrency();
        return hw > 2 ? hw - 2 : std::max(2u, hw);
    }

    // Per-frame linear PMR arena (O(1) reset).
    class FrameMemoryResource final : public std::pmr::memory_resource {
    public:
        FrameMemoryResource() : buffer_(std::make_unique<std::byte[]>(kCapacity)) {}

        void   reset() noexcept { offset_ = 0; }
        std::pmr::memory_resource* get() noexcept { return this; }

    protected:
        void* do_allocate(size_t bytes, size_t alignment) override {
            auto aligned = [](size_t v, size_t a) { return (v + a - 1) & ~(a - 1); };
            const size_t base = aligned(offset_, alignment);
            if (base + bytes > kCapacity) throw std::bad_alloc();
            offset_ = base + bytes;
            return buffer_.get() + base;
        }
        void do_deallocate(void*, size_t, size_t) noexcept override {}
        bool do_is_equal(const memory_resource& other) const noexcept override {
            return this == &other;
        }

    private:
        static constexpr size_t kCapacity = 16ull * 1024ull * 1024ull;
        std::unique_ptr<std::byte[]> buffer_;
        size_t offset_ = 0;
    };

#ifdef TETRIS_LUA_ENABLED
    // --- Lua bridges: pure value-in/value-out function pointers (no captures).
    // Domains never see lua_State*; these adapt the evaluator edge to
    // progression::ScriptHooks. Lifetime: lua_eval outlives the main loop.
    lua_edge::StatelessLuaEvaluator* g_lua_eval = nullptr;

    progression::LineClearRuling bridge_line_clear_score(int level, int lines, int combo, bool is_tspin) {
        const lua_edge::ScoreRuleResult r = g_lua_eval->call_calculate_score(level, lines, combo, is_tspin);
        return progression::LineClearRuling{ r.score_added, r.level_up, r.danger_alert, r.time_bonus };
    }

    void bridge_clock_rule(float time_left, int stack_height, bool* danger_alert, bool* hurry) {
        const lua_edge::ClockRuleResult r = g_lua_eval->call_evaluate_clock(time_left, stack_height);
        *danger_alert = r.danger_alert;
        *hurry        = r.hurry;
    }
#endif

    // Highest occupied row (+1) — read-only fact wired into progression's
    // clock rule (same privilege model as drop_interval).
    int compute_stack_height(const matrix::MatrixSnapshot& w) {
        for (int y = matrix::GRID_H - 1; y >= 0; --y) {
            for (int x = 0; x < matrix::GRID_W; ++x) {
                if (w.grid[y][x] != 0) return y + 1;
            }
        }
        return 0;
    }

} // namespace

int main(int argc, char* argv[]) {
    // --- CLI parsing ----------------------------------------------------------
    std::string screenshot_path;
    int         screenshot_frame = -1;
    bool        autodrive_drop   = false;
    int         stage_number     = 1;
    std::string script_override;
    long long   expect_target    = -1;
    long long   expect_lines     = -1;
    long long   expect_special_n = -1;
    long long   seed_value       = 20260822;   // daily canyon variant identity
    int         expect_phase     = -1;         // L5 smoke gate: scripted phase advance
    float       encounter_gate_rain = -1.0f;
    bool        has_encounter_gate  = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--screenshot" && i + 1 < argc) {
            screenshot_path = argv[++i];
            screenshot_frame = 60;
        } else if (arg.rfind("--frame=", 0) == 0) {
            screenshot_frame = std::atoi(arg.c_str() + 8);
        } else if (arg == "--autodrive-harddrop") {
            autodrive_drop = true;
        } else if (arg.rfind("--stage=", 0) == 0) {
            stage_number = std::atoi(arg.c_str() + 8);
        } else if (arg.rfind("--script=", 0) == 0) {
            script_override = arg.substr(9);
        } else if (arg.rfind("--seed=", 0) == 0) {
            seed_value = std::atoll(arg.c_str() + 7);
        } else if (arg.rfind("--expect-target-score=", 0) == 0) {
            expect_target = std::atoll(arg.c_str() + 22);
        } else if (arg.rfind("--expect-target-lines=", 0) == 0) {
            expect_lines = std::atoll(arg.c_str() + 22);
        } else if (arg.rfind("--expect-special-every-n=", 0) == 0) {
            expect_special_n = std::atoll(arg.c_str() + 25);
        } else if (arg.rfind("--expect-encounter-config=", 0) == 0) {
            // L5 gate: rain_every must match the script value (e.g. 8.0 → "8")
            encounter_gate_rain = (float)std::atof(arg.c_str() + 26);
            has_encounter_gate  = true;
        }
    }
    const bool headless = (screenshot_frame >= 0);

    // --- Campaign stage selection (M2 manifest) ---------------------------------
    const config::campaign::Stage* stage = config::campaign::find_stage(stage_number);
    if (!stage) {
        std::cerr << "Unknown campaign stage: " << stage_number << std::endl;
        return 1;
    }
    config::Rules rules = stage->make_rules();

    // --- Lua rule-script boot (value-in/value-out; native C++ fallback) ----------
    // Re-runnable per stage: advancing the campaign loads the next stage's
    // script into a FRESH sandbox so no globals leak between stages.
    progression::ScriptHooks script_hooks{};
    // L3 boot queue: commands applied on the first playing frame after a stage
    // load (the initial-board stamp reaches the reducer like any other intent).
    std::vector<matrix::TetrisCommand> boot_commands;
    int canyon_seed_tag = 0;   // HUD seed-tag projection input
    // L4 special-piece scheduler state (cadence counter + armed cycle). Script
    // decisions become raw matrix commands that ride the boot queue like the
    // L3 stamp — the grid is only ever touched through plain intents.
    powerups::PowerupSnapshot powerup_state{};
    // L5 encounter state (pod 5). The overseer script decides phases/moods;
    // its rulings cross as plain values; garbage volleys ride boot_commands.
    environment::EnvironmentSnapshot env_state{};
    environment::EncounterConfig     env_cfg{};
#ifdef TETRIS_LUA_ENABLED
    std::unique_ptr<lua_edge::StatelessLuaEvaluator> lua_eval;   // fresh per load
#endif
    auto apply_stage_script = [&](const config::campaign::Stage& st, config::Rules& r) {
        script_hooks = progression::ScriptHooks{};   // null hooks ⇒ native rules
        g_lua_eval   = nullptr;
        boot_commands.clear();
        canyon_seed_tag = 0;
        powerup_state = powerups::PowerupSnapshot{};
        env_state = environment::EnvironmentSnapshot{};
        env_cfg   = environment::EncounterConfig{};
#ifdef TETRIS_LUA_ENABLED
        lua_eval.reset();
#endif
        std::string script_path;
        if (!script_override.empty()) {
            script_path = script_override;
        } else if (st.script_path[0] != '\0') {
            script_path = std::string(TETRIS_SOURCE_ROOT) + "/" + st.script_path;
        }
        if (script_path.empty()) return;
#ifdef TETRIS_LUA_ENABLED
        lua_eval = std::make_unique<lua_edge::StatelessLuaEvaluator>();
        if (lua_eval->valid() && lua_eval->load_script_file(script_path.c_str())) {
            g_lua_eval = lua_eval.get();
            if (lua_eval->has_function("BlitzRules", "calculate_score")) {
                script_hooks.line_clear_score = &bridge_line_clear_score;
            }
            if (lua_eval->has_function("BlitzRules", "evaluate_clock")) {
                script_hooks.clock_rule = &bridge_clock_rule;
            }
            lua_eval->apply_config_overrides(r);          // economy overrides

            // L3 board generator: CanyonGen.generate(difficulty, seed) stamps a
            // pre-ruined board at boot. Plain-value rows cross the boundary;
            // main maps them into the matrix stamp payload (raw facts only).
            if (lua_eval->has_table("CanyonGen")) {
                lua_eval->apply_config_overrides(r, "CanyonGen");   // objective numbers
                const lua_edge::GenerationResult gen =
                    lua_eval->call_generate("CanyonGen", 3, seed_value);
                if (gen.valid) {
                    matrix::StampInitialBoardIntent stamp;
                    stamp.cells = {};
                    const int rows = std::min(gen.row_count, matrix::GRID_H);
                    for (int ry = 0; ry < rows; ++ry) {
                        for (int cx = 0; cx < matrix::GRID_W && gen.rows[ry][cx]; ++cx) {
                            stamp.cells[ry][cx] = (gen.rows[ry][cx] == 'X')
                                ? static_cast<uint8_t>(matrix::PieceType::Garbage) : 0;
                        }
                    }
                    boot_commands.push_back(std::move(stamp));
                    canyon_seed_tag = gen.seed_tag;
                    if (gen.target_lines > 0) r.target_lines = gen.target_lines;
                    if (gen.time_limit   > 0.0f) r.time_limit   = gen.time_limit;
                    std::cout << "[lua] canyon board generated: seed=" << seed_value
                              << " rows=" << rows
                              << " target_lines=" << r.target_lines << std::endl;
                }
            }

            // L4 mechanics table: cadence/freeze numbers patch Rules; spawn
            // decisions and lock rulings are evaluated per-frame below.
            if (lua_eval->has_table("CyberRules")) {
                lua_eval->apply_config_overrides(r, "CyberRules");
                std::cout << "[lua] cyber rules active: special_every_n="
                          << r.special_every_n
                          << " freeze_seconds=" << r.freeze_seconds << std::endl;
            }

            // L5 encounter table: phase cadence numbers patch nothing in Rules
            // (the show is script-owned); we just snapshot them for main.
            if (lua_eval->has_table("Encounter")) {
                env_cfg = lua_eval->call_encounter_config("Encounter");
                if (env_cfg.valid) {
                    std::cout << "[lua] encounter active: phases="
                              << env_cfg.phase_count
                              << " rain_every=" << env_cfg.rain_every << "s" << std::endl;
                }
            }

            std::cout << "[lua] rule script active: " << script_path << std::endl;
        } else {
            std::cerr << "[lua] script unavailable (" << script_path
                      << ") — native C++ rules active" << std::endl;
        }
#else
        std::cerr << "[lua] built without Lua — native C++ rules active" << std::endl;
#endif
    };
    apply_stage_script(*stage, rules);

    // --- Smoke gate: wired script must have overridden the default target --------
    if (expect_target >= 0) {
        const bool pass = (rules.target_score == static_cast<int>(expect_target));
        std::cout << "SMOKE_TARGET_SCORE=" << (pass ? "PASS" : "FAIL")
                  << " (expected=" << expect_target
                  << ", actual=" << rules.target_score << ")" << std::endl;
        return pass ? 0 : 2;
    }
    // L3 gate: generator/config must agree on the excavation objective.
    if (expect_lines >= 0) {
        const bool pass = (rules.target_lines == static_cast<int>(expect_lines));
        std::cout << "SMOKE_TARGET_LINES=" << (pass ? "PASS" : "FAIL")
                  << " (expected=" << expect_lines
                  << ", actual=" << rules.target_lines << ")" << std::endl;
        return pass ? 0 : 2;
    }
    // L4 gate: the cyber mechanics table must have patched the cadence.
    if (expect_special_n >= 0) {
        const bool pass = (rules.special_every_n == static_cast<int>(expect_special_n));
        std::cout << "SMOKE_SPECIAL_EVERY_N=" << (pass ? "PASS" : "FAIL")
                  << " (expected=" << expect_special_n
                  << ", actual=" << rules.special_every_n << ")" << std::endl;
        return pass ? 0 : 2;
    }

    // L5 gate: encounter config must be live with the scripted rain cadence.
    if (has_encounter_gate) {
        const bool pass = env_cfg.valid
            && std::fabs(env_cfg.rain_every - encounter_gate_rain) < 0.001f;
        std::cout << "SMOKE_ENCOUNTER_CONFIG=" << (pass ? "PASS" : "FAIL")
                  << " (expected rain_every=" << encounter_gate_rain
                  << ", actual=" << (env_cfg.valid ? env_cfg.rain_every : -1.0f)
                  << ")" << std::endl;
        return pass ? 0 : 2;
    }
    // Native-fallback gate: no Lua ⇒ overseer absent, phase stays CALM.
    if (expect_phase >= 0) {
        std::cout << "SMOKE_ENCOUNTER_PHASE=" << (env_state.phase >= expect_phase ? "PASS" : "FAIL")
                  << " (expected>=" << expect_phase << ", actual=" << env_state.phase << ")" << std::endl;
        return env_state.phase >= expect_phase ? 0 : 2;
    }

    // --- SDL lifecycle ------------------------------------------------------------
    Uint32 sdl_flags = SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_AUDIO;
    if (SDL_Init(sdl_flags) < 0) {
        if (headless) {
            sdl_flags &= ~static_cast<Uint32>(SDL_INIT_AUDIO);
            if (SDL_Init(sdl_flags) < 0) {
                std::cerr << "SDL_Init error: " << SDL_GetError() << std::endl;
                return 1;
            }
        } else {
            std::cerr << "SDL_Init error: " << SDL_GetError() << std::endl;
            return 1;
        }
    }

    SDL_Window*       window         = nullptr;
    SDL_Renderer*     sdl_renderer   = nullptr;
    SDL_Texture*      screen_texture = nullptr;
    SDL_Surface*      screen_surface = nullptr;
    SDL_AudioDeviceID audio_dev      = 0;

    if (!headless) {
        window         = SDL_CreateWindow(stage->display_name,
                                          SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                          CANVAS_WIDTH, CANVAS_HEIGHT, SDL_WINDOW_SHOWN);
        sdl_renderer   = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
        screen_texture = SDL_CreateTexture(sdl_renderer, SDL_PIXELFORMAT_RGBA32,
                                           SDL_TEXTUREACCESS_STREAMING, CANVAS_WIDTH, CANVAS_HEIGHT);
        screen_surface = SDL_CreateRGBSurfaceWithFormat(0, CANVAS_WIDTH, CANVAS_HEIGHT, 32,
                                                        SDL_PIXELFORMAT_RGBA32);

        SDL_AudioSpec want{}, have{};
        want.freq     = 44100;
        want.format   = AUDIO_F32SYS;
        want.channels = 2;
        want.samples  = 2048;
        want.callback = audio::audio_callback;
        want.userdata = &g_audio;

        audio_dev = SDL_OpenAudioDevice(nullptr, 0, &want, &have, 0);
        if (audio_dev) SDL_PauseAudioDevice(audio_dev, 0);
    }

    // --- Renderer state -----------------------------------------------------------
    shs::Canvas  canvas(CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{ 14, 16, 22, 255 });
    shs::ZBuffer z_buffer(CANVAS_WIDTH, CANVAS_HEIGHT, -1.0f, 1.0f);

    shs::Job::ThreadedPriorityJobSystem job_system(static_cast<int>(thread_count()));
    shs::Job::WaitGroup                 wg_render;

    FrameMemoryResource frame_memory;

    // --- Persistent pod states ------------------------------------------------------
    matrix::MatrixSnapshot world;
    world.active.type = matrix::pull_next_piece(world.rng_state, world.next_queue);
    world.active.pos  = { 4, 19 };

    progression::ScoreState score_state;
    score_state.target_score = rules.target_score;
    score_state.mode_id      = rules.mode_id;
    score_state.time_left    = rules.time_limit;

    // NOTE: default resource, NOT the frame arena — particles/rings outlive frames.
    spatial_fx::FxState fx(std::pmr::get_default_resource());

    ui::HudState hud;

    // --- Session layer (M1): meta game-state machine ---------------------------------
    session::SessionSnapshot session;
    session.stage_count = config::campaign::STAGE_COUNT;
    if (headless) {
        // Verification runs skip menus entirely: straight into PLAYING.
        session.screen         = session::Screen::PLAYING;
        session.current_stage  = stage_number - 1;
        session.unlocked_stages = config::campaign::STAGE_COUNT;
    } else if (stage_number > 1) {
        // Windowed --stage=N: jump straight into that stage's run (rules and
        // script were already applied above). Without --stage, boot to TITLE.
        session.screen          = session::Screen::PLAYING;
        session.current_stage   = stage_number - 1;
        session.stage_cursor    = stage_number - 1;
        session.unlocked_stages = config::campaign::STAGE_COUNT;
    }

    int session_high = 0;   // best score across stages this session

    // Full stage (re)load: fresh board/score/HUD/FX + stage rules + fresh
    // script sandbox. This is what guarantees NO stale level-finished GUI
    // (modal, banners, floaters, particles) leaks into the next level —
    // every presentation surface is reset here.
    auto load_stage = [&](const config::campaign::Stage* st) {
        stage = st;
        rules = st->make_rules();
        apply_stage_script(*st, rules);
        world = matrix::MatrixSnapshot{};
        world.active.type = matrix::pull_next_piece(world.rng_state, world.next_queue);
        world.active.pos  = { 4, 19 };
        session_high = std::max(session_high, score_state.high_score);
        score_state = progression::ScoreState{};
        score_state.target_score = rules.target_score;
        score_state.mode_id      = rules.mode_id;
        score_state.time_left    = rules.time_limit;
        score_state.high_score   = session_high;
        hud = ui::HudState{};
        fx  = spatial_fx::FxState(std::pmr::get_default_resource());
        if (window) SDL_SetWindowTitle(window, st->display_name);
    };

    // Manifest metadata for the level-select carousel (ui-edge projection input).
    const char* stage_names[config::campaign::STAGE_COUNT];
    const char* stage_tiers[config::campaign::STAGE_COUNT];
    for (int i = 0; i < config::campaign::STAGE_COUNT; ++i) {
        stage_names[i] = config::campaign::STAGES[i].display_name;
        stage_tiers[i] = config::campaign::STAGES[i].script_path[0] != '\0'
                             ? ui::TAG_SCRIPTED : ui::TAG_PURE;
    }

    bool   quit  = false;
    int    frame = 0;
    Uint32 last_tick = SDL_GetTicks();

    // --- Main loop ---------------------------------------------------------------------
    while (!quit) {
        float dt;
        if (headless) {
            dt = 1.0f / 60.0f; // deterministic stepping for screenshots
        } else {
            const Uint32 cur_tick = SDL_GetTicks();
            dt = (cur_tick - last_tick) / 1000.0f;
            last_tick = cur_tick;
            if (dt > 0.05f) dt = 0.05f;
        }

        frame_memory.reset();
        std::pmr::memory_resource* arena = frame_memory.get();

        // 1. INPUT EDGE
        input::InputState in = input::poll_input(arena);
        quit = quit || in.quit;
        if (autodrive_drop && frame == 30) in.commands.push_back(matrix::HardDropIntent{});

        // 1b. SESSION EDGE (meta state machine: title/select/pause/results)
        auto sstep = session::reduce_session(session,
            std::span<const session::SessionCommand>(
                in.session_commands.data(), in.session_commands.size()),
            dt, arena);
        session = std::move(sstep.next);
        for (const auto& sev : sstep.events) {
            switch (sev.type) {
            case session::SessionEventType::STAGE_SELECTED:
                load_stage(config::campaign::find_stage(sev.stage + 1));
                break;
            case session::SessionEventType::RUN_RESTART_REQUESTED:
                load_stage(config::campaign::find_stage(session.current_stage + 1));
                break;
            case session::SessionEventType::QUIT_REQUESTED:
                quit = true;
                break;
            case session::SessionEventType::SOUND_TOGGLED:
                g_audio.set_enabled(sev.enabled);
                break;
            case session::SessionEventType::NAV_MOVED:
                if (!headless) g_audio.play(audio::SND_MENU_MOVE);
                break;
            case session::SessionEventType::CONFIRMED:
                if (!headless) g_audio.play(audio::SND_MENU_CONFIRM);
                break;
            default: break;
            }
        }
        const bool playing = (session.screen == session::Screen::PLAYING);

        // Gameplay pods only step during a live run; menus freeze everything.
        if (playing) {
            // Restart preservation: high score survives a manual reset (main-edge duty)
            bool restart_requested = false;
            for (const auto& cmd : in.commands) {
                if (std::holds_alternative<matrix::RestartIntent>(cmd)) restart_requested = true;
            }

            // Blitz time-up freezes the run: no commands reach the matrix, no gravity.
            // A restart press thaws that frame so the reset reaches the board too.
            const bool frozen = score_state.time_up && !restart_requested;
            auto cmd_span = frozen
                ? std::span<const matrix::TetrisCommand>()
                : std::span<const matrix::TetrisCommand>(in.commands.data(), in.commands.size());
            const float matrix_dt = frozen ? 0.0f : dt;

            // Gravity cadence wired from progression level through pure config math
            world.drop_interval = rules.gravity_for_level(score_state.level);

            // Boot queue first: the L3 initial-board stamp rides in as an
            // ordinary command on the frame right after a stage load.
            if (!boot_commands.empty()) {
                std::vector<matrix::TetrisCommand> merged;
                merged.reserve(boot_commands.size() + in.commands.size());
                for (auto& c : boot_commands) merged.push_back(std::move(c));
                boot_commands.clear();
                for (const auto& c : cmd_span) merged.push_back(c);
                cmd_span = std::span<const matrix::TetrisCommand>(merged.data(), merged.size());
            }

            // 2. PURE SIMULATION CORE
            matrix::MatrixStepResult step = matrix::reduce_matrix(world, cmd_span, matrix_dt, arena);
            world = std::move(step.next_state);

            // 3. EVENT-FED PROGRESSION (+ blitz clock via injected rule hooks)
            if (restart_requested) {
                const int preserved_high = score_state.high_score;
                session_high = std::max(session_high, preserved_high);
                score_state = progression::ScoreState{};
                score_state.target_score = rules.target_score;
                score_state.mode_id      = rules.mode_id;
                score_state.time_left    = rules.time_limit;
                score_state.high_score   = preserved_high;
                hud = ui::HudState{};   // no stale banners/floaters across resets
                powerup_state = powerups::PowerupSnapshot{};
                env_state = environment::EnvironmentSnapshot{};   // L5 fresh show
            }
            progression::ProgressionStep prog = progression::reduce_progression(
                std::span<const matrix::MatrixEvent>(step.events.data(), step.events.size()),
                score_state, rules, arena,
                dt, compute_stack_height(world), script_hooks
            );
            score_state = std::move(prog.next);

            // Environment mood wire (pod-5 embryo): amber intensity rises as the
            // blitz clock drains; untimed modes stay neutral. The canyon stage
            // pins the dusk environment on instead (mesas/torches/sandstone).
            fx.mood_intensity = (rules.time_limit > 0.0f && rules.mode_id != config::MODE_GARBAGE_CANYON)
                ? glm::clamp(1.0f - score_state.time_left / rules.time_limit, 0.0f, 1.0f)
                : 0.0f;
            fx.env_dusk = (rules.mode_id == config::MODE_GARBAGE_CANYON) ? 1.0f : 0.0f;
            fx.env_neon = (rules.mode_id == config::MODE_CYBER_STORM) ? 1.0f : 0.0f;
            fx.env_finale = (rules.mode_id == config::MODE_ENCORE_FINALE) ? 1.0f : 0.0f;

            // 3c. L5 ENCOUNTER OVERSEER (event-fed; rulings cross as plain values)
            if (fx.env_finale > 0.5f) {
                // Crowd pulses from discrete events (value-in/value-out).
                environment::CrowdPulse pulse{};
                for (const auto& ev : step.events) {
                    if (ev.type == matrix::MatrixEventType::LINES_CLEARED) {
                        pulse = lua_eval && lua_eval->valid()
                            ? lua_eval->call_on_event("Encounter", 1,
                                  (int)ev.lines_cleared_count)
                            : environment::CrowdPulse{ true, 0.25f + 0.15f * ev.lines_cleared_count };
                    }
                }
                for (const auto& pev : prog.events) {
                    if (pev.type == progression::ProgressionEventType::OBJECTIVE_COMPLETED) {
                        pulse = lua_eval && lua_eval->valid()
                            ? lua_eval->call_on_event("Encounter", 2, 1)
                            : environment::CrowdPulse{ true, 1.0f };
                    }
                }

                // Danger = stack near the ceiling (same projection as the HUD).
                const int stack_h = compute_stack_height(world);
                const bool danger = stack_h >= matrix::VISIBLE_H - 4;

                const environment::OverseerRuling ruling =
                    (lua_eval && lua_eval->valid())
                        ? lua_eval->call_decide_phase("Encounter",
                              env_state.phase, env_state.phase_time,
                              score_state.lines_cleared, danger)
                        : environment::OverseerRuling{};   // no script ⇒ static CALM

                const environment::EnvironmentStepResult estep =
                    environment::reduce_environment(env_state, ruling, pulse, dt,
                                                    env_cfg.valid ? env_cfg.rain_every : 0.0f);
                env_state = estep.next;

                // Rain cadence: one volley per elapsed interval, holes decided
                // deterministically from frame parity + volley index (raw facts
                // only; the script owns WHEN/HOW MUCH, not the grid layout).
                if (estep.rain_due) {
                    matrix::AddGarbageRowsIntent rain;
                    rain.rows = env_cfg.valid ? (uint8_t)std::min(env_cfg.rain_rows, 4) : 1;
                    const int rows = (int)rain.rows;
                    for (int r = 0; r < rows; ++r) {
                        rain.hole_x[r] = (uint8_t)((frame * 7 + r * 5) % matrix::GRID_W);
                    }
                    boot_commands.push_back(rain);
                    // Warning beat: brief pre-volley flash + HUD banner.
                    hud.flash = std::max(hud.flash, 0.20f);
                }

                // Phase-transition set pieces: white-out wipe + banner text.
                if (estep.phase_changed) {
                    fx.screen_flash = std::max(fx.screen_flash, 0.85f);   // white-out wipe
                    switch (env_state.phase) {
                    case environment::PHASE_RAIN:
                        hud.spawn_floater("GARBAGE RAIN", shs::Color{ 255, 160, 60, 255 }, 2.2f);
                        if (!headless) g_audio.play(audio::SND_THUD);
                        break;
                    case environment::PHASE_BLACKOUT:
                        hud.spawn_floater("BLACKOUT", shs::Color{ 140, 150, 220, 255 }, 2.2f);
                        break;
                    case environment::PHASE_CRESCENDO:
                        hud.spawn_floater("FINALE", shs::Color{ 255, 210, 60, 255 }, 2.6f);
                        if (!headless) g_audio.play(audio::SND_TETRIS_FOUR);
                        break;
                    default: break;
                    }
                }

                // Plain-value wires into FX (planner consumes these directly).
                fx.mood_phase   = env_state.mood;
                fx.dim          = env_state.dim;
                fx.ghost_hidden = env_state.dim > 0.5f;   // ghost hides past halfway
                fx.crowd_pulse  = env_state.crowd_pulse;
                fx.finale_phase = env_state.phase;   // plain-value phase mirror
            }

            // 3b. L4 POWERUP SCHEDULER (event-fed; rulings cross as plain values)
            // Cadence counts spawns; when a special locks, its lock ruling from
            // the stage script becomes raw ClearCells / FreezeGravity commands.
            std::pmr::vector<powerups::ApplyRulingIntent> frame_rulings(arena);
            if (lua_eval && lua_eval->valid()
                && lua_eval->has_function("CyberRules", "on_special_lock")) {
                for (const auto& ev : step.events) {
                    if (ev.type != matrix::MatrixEventType::SPECIAL_LOCKED) continue;
                    const auto ruling = lua_eval->call_on_special_lock(
                        "CyberRules", ev.special_type, ev.lock_x, ev.lock_y, world.grid);
                    if (!ruling.valid) continue;
                    frame_rulings.push_back(powerups::ApplyRulingIntent{
                        ruling, static_cast<int16_t>(ev.lock_x),
                        static_cast<int16_t>(ev.lock_y) });
                    if (ruling.fx_id == 1 || ruling.fx_id == 2) {   // bomb/laser flash
                        hud.flash = std::max(hud.flash,
                            ruling.fx_id == 1 ? 0.55f : 0.35f);
                    }
                }
            }
            powerups::PowerupStep pstep = powerups::reduce_powerups(
                powerup_state,
                std::span<const matrix::MatrixEvent>(step.events.data(), step.events.size()),
                std::span<const powerups::ApplyRulingIntent>(
                    frame_rulings.data(), frame_rulings.size()),
                rules.special_every_n, rules.freeze_seconds, arena);
            powerup_state = std::move(pstep.next);
            for (auto& mut : pstep.mutations) boot_commands.push_back(std::move(mut));

            // Fulfill a latched special request ONCE (edge-triggered by this
            // frame's SPAWN_SPECIAL_REQUESTED event) via the script's pure
            // decide_spawn(); the queued piece enters the bag next frame.
            bool spawn_requested_now = false;
            for (const auto& uev : pstep.events) {
                if (uev.type == powerups::PowerupEventType::SPAWN_SPECIAL_REQUESTED) {
                    spawn_requested_now = true;
                }
            }
            if (spawn_requested_now && lua_eval && lua_eval->valid()
                && lua_eval->has_function("CyberRules", "decide_spawn")) {
                const auto dec = lua_eval->call_decide_spawn("CyberRules",
                    pstep.next.pieces_since_special,
                    static_cast<int>(pstep.next.armed_next));
                if (dec.valid) {
                    matrix::QueueSpecialIntent q;
                    q.special_type = dec.special_type;
                    boot_commands.push_back(q);
                }
            }

            // Run-end latch: hand the finished run to the RESULTS screen.
            if (score_state.victory || score_state.time_up || world.game_over) {
                session.run_victory     = score_state.victory;
                session.run_time_up     = score_state.time_up;
                session.final_score     = score_state.score;
                session.final_lines     = score_state.lines_cleared;
                session.final_max_combo = score_state.max_combo;
                session.final_seconds   = world.game_time;
                if (score_state.victory) {
                    session.unlocked_stages = std::min(session.stage_count,
                        std::max(session.unlocked_stages, session.current_stage + 2));
                }
                session.cursor = 0;
                session.screen = session::Screen::RESULTS;
            }

            // 4. FX STEP (particles + rings + camera spring/pulse, deterministic xorshift)
            spatial_fx::step_fx(fx,
                std::span<const matrix::MatrixEvent>(step.events.data(), step.events.size()),
                std::span<const progression::ProgressionEvent>(prog.events.data(), prog.events.size()),
                dt);

            // HUD transient presentation state (banners/floaters + dust trigger)
            ui::step_hud(hud,
                std::span<const progression::ProgressionEvent>(prog.events.data(), prog.events.size()),
                dt,
                std::span<const matrix::MatrixEvent>(step.events.data(), step.events.size()));

            // Audio edge mapping (windowed mode only)
            if (!headless) {
                for (const auto& ev : step.events) {
                    switch (ev.type) {
                    case matrix::MatrixEventType::PIECE_MOVED:       g_audio.play(audio::SND_MOVE);        break;
                    case matrix::MatrixEventType::PIECE_ROTATED:     g_audio.play(audio::SND_ROTATE);      break;
                    case matrix::MatrixEventType::PIECE_LOCK_IMPACT: g_audio.play(audio::SND_DROP_SLAM);   break;
                    case matrix::MatrixEventType::HOLD_SWAPPED:      g_audio.play(audio::SND_HOLD);        break;
                    case matrix::MatrixEventType::GAME_OVER:         g_audio.play(audio::SND_GAME_OVER);   break;
                    case matrix::MatrixEventType::HARD_DROP_SLAM:    g_audio.play(audio::SND_DROP_SLAM);   break;
                    case matrix::MatrixEventType::LINES_CLEARED:
                        g_audio.play(ev.lines_cleared_count >= 4 ? audio::SND_TETRIS_FOUR
                                                                 : audio::SND_LINE_CLEAR);
                        // Heavy garbage collapse: layered deep thud under the chime.
                        if (ev.garbage_cells >= 12) g_audio.play(audio::SND_THUD);
                        break;
                    default: break;
                    }
                }
                for (const auto& pev : prog.events) {
                    switch (pev.type) {
                    case progression::ProgressionEventType::CLOCK_TICK:          g_audio.play(audio::SND_TICK);        break;
                    case progression::ProgressionEventType::TIME_UP:             g_audio.play(audio::SND_GAME_OVER);   break;
                    case progression::ProgressionEventType::OBJECTIVE_COMPLETED: g_audio.play(audio::SND_TETRIS_FOUR); break;
                    default: break;
                    }
                }
                for (const auto& uev : pstep.events) {
                    switch (uev.type) {
                    case powerups::PowerupEventType::POWERUP_TRIGGERED:
                        g_audio.play(uev.powerup == 1 ? audio::SND_BLAST
                                   : uev.powerup == 2 ? audio::SND_ZAP
                                                      : audio::SND_FROST);
                        break;
                    default: break;
                    }
                }
            }
        }

        // 5. PURE SCENE PLANNER (per-level camera preset rides in via Rules)
        spatial_fx::PipelineExecutionPlan plan = spatial_fx::plan_tetris_scene(
            world, fx, CANVAS_WIDTH, CANVAS_HEIGHT, arena, rules.camera
        );

        // HUD wiring bundle (plain values): canyon projections active only on
        // the excavation stage.
        const ui::CanyonHudInfo canyon_info{
            rules.mode_id == config::MODE_GARBAGE_CANYON, canyon_seed_tag };
        // L5 encore projections: phase meter + pre-volley warning window.
        const ui::EncoreHudInfo encore_info{
            rules.mode_id == config::MODE_ENCORE_FINALE,
            env_state.phase,
            env_state.phase_time,
            [&] {
                // Per-phase intensity curve: CALM 0-40s, RAIN 0-50s,
                // BLACKOUT 0-22s, CRESCENDO ramps to full over 10s.
                switch (env_state.phase) {
                case environment::PHASE_CALM:      return glm::clamp(env_state.phase_time / 40.0f, 0.0f, 1.0f);
                case environment::PHASE_RAIN:      return glm::clamp(env_state.phase_time / 50.0f, 0.0f, 1.0f);
                case environment::PHASE_BLACKOUT:  return glm::clamp(env_state.phase_time / 22.0f, 0.0f, 1.0f);
                default:                           return glm::clamp(env_state.phase_time / 10.0f, 0.0f, 1.0f);
                }
            }(),
            (env_state.phase == environment::PHASE_RAIN)
                ? glm::clamp(env_cfg.valid ? env_cfg.rain_every - env_state.rain_timer : 0.0f,
                             0.0f, 2.0f)
                : 0.0f,
            env_state.dim };
        const ui::CyberHudInfo cyber_info{
            rules.mode_id == config::MODE_CYBER_STORM,
            rules.special_every_n > 0
                ? glm::clamp((float)powerup_state.pieces_since_special
                                 / (float)rules.special_every_n, 0.0f, 1.0f)
                : 0.0f,
            static_cast<int>(powerup_state.armed_next),
            world.gravity_freeze,
            [&] {
                const int nq0 = static_cast<int>(world.next_queue[0]);
                return nq0 >= 9 && nq0 <= 11;
            }(),
            static_cast<int>(world.next_queue[0]) };

        // 6. TILED PARALLEL RASTERIZATION
        canvas.buffer().clear(shs::Color{ 14, 16, 22, 255 });
        z_buffer.clear();

        const int W    = canvas.get_width();
        const int H    = canvas.get_height();
        const int cols = (W + TILE_SIZE_X - 1) / TILE_SIZE_X;
        const int rows = (H + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

        wg_render.reset();
        for (int ty = 0; ty < rows; ++ty) {
            for (int tx = 0; tx < cols; ++tx) {
                wg_render.add(1);
                job_system.submit({ [&, tx, ty, W, H]() {
                    glm::ivec2 tmin(tx * TILE_SIZE_X, ty * TILE_SIZE_Y);
                    glm::ivec2 tmax(std::min((tx + 1) * TILE_SIZE_X, W) - 1,
                                    std::min((ty + 1) * TILE_SIZE_Y, H) - 1);

                    for (const auto& tri : plan.triangles) {
                        const shs::Raster::FrustumClipPolygon poly =
                            shs::Raster::clip_triangle_to_frustum(tri.c0, tri.c1, tri.c2);
                        if (poly.count < 3) continue;

                        glm::vec4 s0 = vop::clip_to_screen_vec4(poly.vertices[0], W, H);
                        for (int i = 1; i + 1 < poly.count; ++i) {
                            glm::vec4 s1 = vop::clip_to_screen_vec4(poly.vertices[i], W, H);
                            glm::vec4 s2 = vop::clip_to_screen_vec4(poly.vertices[i + 1], W, H);
                            vop::rasterize_triangle_tile(canvas, z_buffer, s0, s1, s2,
                                                         tri.lit_color, tri.depth_bias, tmin,
                                                         tmax, tri.alpha);
                        }
                    }
                    wg_render.done();
                }, shs::Job::PRIORITY_HIGH });
            }
        }
        wg_render.wait();

        // 7. UI EDGE — per-session-screen projection
        switch (session.screen) {
        case session::Screen::TITLE:
            ui::draw_title_screen(canvas, session);
            break;
        case session::Screen::LEVEL_SELECT:
            ui::draw_level_select(canvas, session, stage_names, stage_tiers,
                                  config::campaign::STAGE_COUNT);
            break;
        case session::Screen::PAUSED:
            ui::draw_hud(canvas, world, score_state, hud, false, canyon_info, cyber_info);
            ui::draw_pause_overlay(canvas, session);
            break;
        case session::Screen::RESULTS:
            ui::draw_results_screen(canvas, session);
            break;
        default: // PLAYING
            ui::draw_hud(canvas, world, score_state, hud,
                         stage->index < config::campaign::STAGE_COUNT, canyon_info,
                         cyber_info, encore_info);
            ui::draw_encore_hud(canvas, encore_info, hud);
            break;
        }

        ++frame;

        // Headless exit: save BMP and stop
        if (headless) {
            if (frame >= screenshot_frame) {
                SDL_Surface* shot = SDL_CreateRGBSurfaceWithFormat(0, CANVAS_WIDTH, CANVAS_HEIGHT, 32,
                                                                   SDL_PIXELFORMAT_RGBA32);
                shs::Canvas::copy_to_SDLSurface(shot, &canvas);
                SDL_SaveBMP(shot, screenshot_path.c_str());
                SDL_FreeSurface(shot);
                std::cout << "Screenshot saved: " << screenshot_path
                          << " (frame " << frame << ")" << std::endl;
                break;
            }
            continue;
        }

        // 8. SWAPCHAIN PRESENTATION
        shs::Canvas::copy_to_SDLSurface(screen_surface, &canvas);
        SDL_UpdateTexture(screen_texture, NULL, screen_surface->pixels, screen_surface->pitch);
        SDL_RenderClear(sdl_renderer);
        SDL_RenderCopy(sdl_renderer, screen_texture, NULL, NULL);
        SDL_RenderPresent(sdl_renderer);
    }

    // --- Cleanup -----------------------------------------------------------------------
    if (audio_dev)      SDL_CloseAudioDevice(audio_dev);
    if (screen_surface) SDL_FreeSurface(screen_surface);
    if (screen_texture) SDL_DestroyTexture(screen_texture);
    if (sdl_renderer)   SDL_DestroyRenderer(sdl_renderer);
    if (window)         SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}