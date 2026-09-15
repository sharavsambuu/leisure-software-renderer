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
#include "shs/memory/frame_memory_resource.hpp"   // P1.5: shared frame arena (was demo-private)

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
#include <game/step.hpp>
#include <game/stage.hpp>
#include <game/world.hpp>

#ifdef TETRIS_LUA_ENABLED
// P1b: adapts the Lua evaluator edge to game::IScriptHost. Lives in main
// because it touches edges/lua directly (purity: game/ never does).
class LuaScriptHost final : public tetris::game::IScriptHost {
public:
    explicit LuaScriptHost(tetris::lua_edge::StatelessLuaEvaluator* eval) : ev_(eval) {}
    bool valid() const override { return ev_ && ev_->valid(); }

    bool has_special_lock() const override {
        return ev_ && ev_->has_function("CyberRules", "on_special_lock");
    }
    tetris::powerups::SpecialRuling on_special_lock(
        int special_type, int lock_x, int lock_y,
        const tetris::matrix::CellGrid& grid) override {
        return ev_->call_on_special_lock("CyberRules", special_type, lock_x,
                                         lock_y, grid);
    }
    bool has_decide_spawn() const override {
        return ev_ && ev_->has_function("CyberRules", "decide_spawn");
    }
    tetris::powerups::SpawnDecision decide_spawn(int pieces_since,
                                                 int armed_next) override {
        return ev_->call_decide_spawn("CyberRules", pieces_since, armed_next);
    }
    bool has_goal_test(const char* goal_table) const override {
        return ev_ && ev_->has_function(goal_table, "test");
    }
    bool evaluate_goal(
        const char* goal_table,
        const std::vector<tetris::mission::MissionEventView>& events,
        const tetris::mission::MissionSnapshot& snap) override {
        if (!ev_ || !ev_->valid()) return false;
        lua_State* L = ev_->raw();
        lua_getglobal(L, goal_table);
        if (!lua_istable(L, -1)) { lua_pop(L, 1); return false; }
        lua_getfield(L, -1, "test");
        if (!lua_isfunction(L, -1)) { lua_pop(L, 2); return false; }

        // events: array of { type="...", a=N, b=N }
        lua_createtable(L, (int)events.size(), 0);
        for (size_t i = 0; i < events.size(); ++i) {
            lua_createtable(L, 0, 3);
            lua_pushstring(L, events[i].type);
            lua_setfield(L, -2, "type");
            lua_pushinteger(L, events[i].a);
            lua_setfield(L, -2, "a");
            lua_pushinteger(L, events[i].b);
            lua_setfield(L, -2, "b");
            lua_rawseti(L, -2, (int)(i + 1));
        }

        // snapshot: plain table
        lua_createtable(L, 0, 5);
        lua_pushinteger(L, snap.score);       lua_setfield(L, -2, "score");
        lua_pushinteger(L, snap.lines);       lua_setfield(L, -2, "lines");
        lua_pushinteger(L, snap.level);       lua_setfield(L, -2, "level");
        lua_pushboolean(L, snap.overdrive ? 1 : 0);
        lua_setfield(L, -2, "overdrive");
        lua_pushnumber(L, snap.stack_ratio);
        lua_setfield(L, -2, "stack_ratio");

        if (lua_pcall(L, 2, 1, 0) != LUA_OK) {   // (events, snapshot) -> result
            lua_pop(L, 1);                        // error message
            return false;
        }
        const bool ok = lua_toboolean(L, -1) != 0;
        lua_pop(L, 2);                            // result + Goals table
        return ok;
    }
    tetris::environment::CrowdPulse on_event(int kind, int value) override {
        return ev_->call_on_event("Encounter", kind, value);
    }
    tetris::environment::OverseerRuling decide_phase(
        int phase, float phase_time, int lines_cleared, bool danger) override {
        return ev_->call_decide_phase("Encounter", phase, phase_time,
                                      lines_cleared, danger);
    }
private:
    tetris::lua_edge::StatelessLuaEvaluator* ev_;
};
#endif

namespace {

    using namespace tetris;

    // Audio synth instance lives at file scope: the SDL callback thread
    // dereferences it for the lifetime of the audio device.
    audio::TetrisAudioSynth g_audio;

    // P1b: IAudioSink adapter so step_core can request sounds without
    // touching the audio edge directly.
    struct MainAudioSink final : public game::IAudioSink {
        bool enabled = true;
        void play(int sound_id) override {
            if (enabled) g_audio.play(static_cast<audio::SoundType>(sound_id));
        }
    };
    MainAudioSink g_audio_sink;

    constexpr int CANVAS_WIDTH  = 1280;
    constexpr int CANVAS_HEIGHT = 720;
    constexpr int TILE_SIZE_X   = 80;
    constexpr int TILE_SIZE_Y   = 80;

    unsigned thread_count() {
        const unsigned hw = std::thread::hardware_concurrency();
        return hw > 2 ? hw - 2 : std::max(2u, hw);
    }

    // Per-frame linear PMR arena (O(1) reset). P1.5: promoted to the shared
    // lib (shs/memory/frame_memory_resource.hpp) per §7.2 rule 6 — the
    // demo-private copy is gone. Tetris keeps its historical 16 MB capacity
    // (see frame_memory instantiation below); overflow is strict bad_alloc.
    using FrameMemoryResource = shs::memory::FrameMemoryResource;

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

    // --- Campaign stage selection (P3: data-driven from campaign.lua) -----------
    auto campaign_load = tetris::game::load_campaign(TETRIS_SOURCE_ROOT);
    if (campaign_load.used_fallback)
        std::cerr << "[campaign] campaign.lua unavailable - fallback stages" << std::endl;
    const auto& stages = campaign_load.stages;

    const tetris::game::StageDef* stage = nullptr;
    for (const auto& st : stages)
        if (st.rules.mode_id == static_cast<int>(stage_number)
            || (&st == &stages[stage_number - 1])) { stage = &st; break; }
    if (!stage) {
        std::cerr << "Unknown campaign stage: " << stage_number << std::endl;
        return 1;
    }
    config::Rules rules = stage->rules;

    // P3.5: per-level level.lua refines the campaign entry (name + overrides).
    std::string level_name = stage->name;
    tetris::game::load_level(TETRIS_SOURCE_ROOT, stage->id, level_name, rules);
#ifdef TETRIS_LUA_ENABLED
    // level.lua may also carry a script reference; campaign script wins only
    // if the level does not override it.
#endif

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
    // (P1a: powerup/env snapshots now live inside game::GameWorld)
    environment::EncounterConfig     env_cfg{};
    // P1a: single world aggregate, declared early so lambdas/helpers see it.
    game::GameWorld g_world{};
    // Main-edge aliases into the world (single source of truth).
    matrix::MatrixSnapshot&                world         = g_world.matrix_state;
    progression::ScoreState&               score_state   = g_world.score;
    powerups::PowerupSnapshot&             powerup_state = g_world.powerups_state;
    environment::EnvironmentSnapshot&      env_state     = g_world.env;
    spatial_fx::FxState&                   fx            = g_world.fx;
    session::SessionSnapshot&              session       = g_world.session;
#ifdef TETRIS_LUA_ENABLED
    std::unique_ptr<lua_edge::StatelessLuaEvaluator> lua_eval;   // fresh per load
    lua_edge::StatelessLuaEvaluator* g_lua_eval = nullptr;        // global hook ptr
#endif
        auto apply_stage_script = [&](const tetris::game::StageDef& st, config::Rules& r) {
        (void)st;
        script_hooks = progression::ScriptHooks{};   // null hooks ⇒ native rules
#ifdef TETRIS_LUA_ENABLED
        g_lua_eval   = nullptr;
#endif
        boot_commands.clear();
        canyon_seed_tag = 0;
        g_world.powerups_state = powerups::PowerupSnapshot{};
        g_world.env            = environment::EnvironmentSnapshot{};
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
                    if (std::getenv("TETRIS_DEBUG")) {
                        for (int ry = 0; ry < gen.row_count; ++ry)
                            std::fprintf(stderr, "[gen] %02d %s\n", ry,
                                         gen.rows[ry]);
                        std::fflush(stderr);
                    }
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
        window         = SDL_CreateWindow(stage->name.c_str(),
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

    // Determinism probe: allow forcing single-threaded rasterization.
    const char* jobs_env = std::getenv("TETRIS_JOBS");
    const int job_threads = (jobs_env && std::atoi(jobs_env) > 0)
        ? std::atoi(jobs_env) : static_cast<int>(thread_count());
    shs::Job::ThreadedPriorityJobSystem job_system(job_threads);
    shs::Job::WaitGroup                 wg_render;

    FrameMemoryResource frame_memory{16ull * 1024ull * 1024ull};   // historical 16 MB arena

    // --- Persistent pod states (P1a: GameWorld declared above) ----------------------
    // First piece spawns lazily in reduce_matrix (frame 1) so the RNG stream
    // starts from the fixed seed regardless of boot ordering.
    world.active.type = matrix::PieceType::None;
    world.active.pos  = { 4, 19 };
    g_world.rules = rules;

    score_state.target_score = rules.target_score;
    score_state.mode_id      = rules.mode_id;
    score_state.time_left    = rules.time_limit;

    ui::HudState hud;

    // --- Session layer (M1): meta game-state machine ---------------------------------
    session.stage_count = (int)stages.size();
    if (headless) {
        // Verification runs skip menus entirely: straight into PLAYING.
        session.screen         = session::Screen::PLAYING;
        session.current_stage  = stage_number - 1;
        session.unlocked_stages = (int)stages.size();
    } else if (stage_number > 1) {
        // Windowed --stage=N: jump straight into that stage's run (rules and
        // script were already applied above). Without --stage, boot to TITLE.
        session.screen          = session::Screen::PLAYING;
        session.current_stage   = stage_number - 1;
        session.stage_cursor    = stage_number - 1;
        session.unlocked_stages = (int)stages.size();
    }

    int session_high = 0;   // best score across stages this session

    // Full stage (re)load: fresh board/score/HUD/FX + stage rules + fresh
    // script sandbox. This is what guarantees NO stale level-finished GUI
    // (modal, banners, floaters, particles) leaks into the next level —
    // every presentation surface is reset here.
        auto load_stage = [&](const tetris::game::StageDef* st) {
        stage = st;
        rules = st->rules;
        g_world.rules = st->rules;
        apply_stage_script(*st, rules);
        const uint32_t saved_rng = world.rng_state;   // keep the boot RNG stream
        world = matrix::MatrixSnapshot{};
        world.rng_state     = saved_rng;
        world.active.type   = matrix::PieceType::None; // reducer spawns frame 1
        world.active.pos    = { 4, 19 };
        session_high = std::max(session_high, score_state.high_score);
        score_state = progression::ScoreState{};
        score_state.target_score = rules.target_score;
        score_state.mode_id      = rules.mode_id;
        score_state.time_left    = rules.time_limit;
        score_state.high_score   = session_high;
        hud = ui::HudState{};
        fx  = spatial_fx::FxState(std::pmr::get_default_resource());
        if (window) SDL_SetWindowTitle(window, st->name.c_str());
    };

    // Manifest metadata for the level-select carousel (ui-edge projection input).
    static std::vector<std::string> stage_name_strs;
    static std::vector<const char*> stage_names;
    static std::vector<const char*> stage_tiers;
    for (const auto& st : stages) {
        stage_name_strs.push_back(st.name);
        stage_names.push_back(stage_name_strs.back().c_str());
        stage_tiers.push_back(st.script_path.empty() ? ui::TAG_PURE
                                                     : ui::TAG_SCRIPTED);
    }

    bool   quit  = false;
    int    frame = 0;
    // Part 6 input feel: stateful edge owns held-state + DAS/ARR scheduling.
    input::InputEdge input_edge;
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

        // 1. INPUT EDGE (stateful: DAS/ARR scheduler advances with frame dt)
        input_edge.begin_frame(dt);
        input::InputState in = input_edge.poll(arena);
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
                if (sev.stage >= 0 && sev.stage < (int)stages.size()) load_stage(&stages[sev.stage]);
                break;
            case session::SessionEventType::RUN_RESTART_REQUESTED:
                if (session.current_stage + 1 < (int)stages.size()) load_stage(&stages[session.current_stage + 1]);
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
            // P1b: the FULL tick lives in game/step.hpp now — scripting via
            // IScriptHost, sounds via IAudioSink, both platform-wired here.
#ifdef TETRIS_LUA_ENABLED
            LuaScriptHost script_host(lua_eval.get());
#else
            game::IScriptHost* script_host = nullptr;
#endif
            game::StepContext s_ctx{};
            s_ctx.dt          = dt;
            s_ctx.frame       = frame;
            s_ctx.headless    = headless;
            s_ctx.audio       = &g_audio_sink;
            g_audio_sink.enabled = !headless;
#ifdef TETRIS_LUA_ENABLED
            s_ctx.scripts     = lua_eval && lua_eval->valid()
                                ? static_cast<game::IScriptHost*>(&script_host)
                                : nullptr;
#else
            s_ctx.scripts     = nullptr;
#endif
            s_ctx.rain_every  = env_cfg.valid ? env_cfg.rain_every : 0.0f;

            game::FrameInput fin{};
            fin.commands       = std::span<const matrix::TetrisCommand>(
                in.commands.data(), in.commands.size());
            fin.soft_drop_held = in.soft_drop_held;

            game::CoreStepResult cres;
            game::step_core(g_world, fin, s_ctx, boot_commands, hud, cres);

            // Aliases for remaining main-side blocks (rendering etc).
            auto& world        = g_world.matrix_state;
            auto& score_state  = g_world.score;
            auto& env_state    = g_world.env;   // encore HUD projection reads this
        }  // playing

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
            ui::draw_level_select(canvas, session, stage_names.data(),
                                  stage_tiers.data(), (int)stages.size());
            break;
        case session::Screen::PAUSED:
            ui::draw_hud(canvas, world, score_state, hud, false, canyon_info, cyber_info);
            ui::draw_pause_overlay(canvas, session);
            break;
        case session::Screen::RESULTS:
            ui::draw_results_screen(canvas, session);
            break;
        default: // PLAYING
            const bool more_stages =
                (stage - stages.data()) < (int)stages.size() - 1;
            ui::draw_hud(canvas, world, score_state, hud,
                         more_stages, canyon_info,
                         cyber_info, encore_info);
            ui::draw_encore_hud(canvas, encore_info, hud);
            break;
        }

        ++frame;

        // TEMP DEBUG: state checksum probe (TETRIS_DEBUG=1)
        if (headless && std::getenv("TETRIS_DEBUG")) {
            unsigned long long sum = 0;
            for (int y = 0; y < matrix::GRID_H; ++y)
                for (int x = 0; x < matrix::GRID_W; ++x)
                    sum = sum * 31 + g_world.matrix_state.grid[y][x];
            // Per-triangle dump on mismatch: run twice, diff the dumps.
            static unsigned long long last_trisum = 0;
            unsigned long long fsum = 0;
            const auto& tris = plan.triangles;
            for (size_t ti = 0; ti < tris.size(); ++ti) {
                const auto& t = tris[ti];
                auto h1 = [&](const glm::vec4& v){
                    unsigned long long x = *(unsigned*)&v.x;
                    unsigned long long y = *(unsigned*)&v.y;
                    unsigned long long z = *(unsigned*)&v.z;
                    return (x*73856093ULL) ^ (y*19349663ULL) ^ (z*83492791ULL);
                };
                fsum = fsum*31 + h1(t.c0)+h1(t.c1)+h1(t.c2)
                     + t.lit_color.r + t.lit_color.g + t.lit_color.b;
            }
            if (last_trisum != 0 && fsum != last_trisum) {
                std::fprintf(stderr, "[dbg] PLAN MISMATCH at frame %d\n", frame);
                for (size_t ti = 0; ti < tris.size(); ++ti) {
                    const auto& t = tris[ti];
                    std::fprintf(stderr,
                        "[tri] %zu v=(%.6f,%.6f,%.6f|%.6f,%.6f,%.6f|%.6f,%.6f,%.6f)"
                        " col=(%u,%u,%u,%u) src=(%u,%u,%u,%u) bias=%.5f a=%u\n",
                        ti,
                        t.c0.x,t.c0.y,t.c0.z, t.c1.x,t.c1.y,t.c1.z,
                        t.c2.x,t.c2.y,t.c2.z,
                        t.lit_color.r,t.lit_color.g,t.lit_color.b,t.lit_color.a,
                        t.src_color.r,t.src_color.g,t.src_color.b,t.src_color.a,
                        t.depth_bias, (unsigned)t.alpha);
                }
            }
            last_trisum = fsum;
            if (frame <= 4) {
                std::fprintf(stderr, "[spawntrace] active=%d pos=%d,%d\n",
                    (int)g_world.matrix_state.active.type,
                    g_world.matrix_state.active.pos.x,
                    g_world.matrix_state.active.pos.y);
                std::fprintf(stderr, "[nq] full=%d,%d,%d,%d,%d rng=%u "
                             "fx.time=%.6f dusk=%.4f neon=%.4f finale=%.4f "
                             "mood=%.4f pulse=%.4f shake=%.4f\n",
                    (int)g_world.matrix_state.next_queue[0],
                    (int)g_world.matrix_state.next_queue[1],
                    (int)g_world.matrix_state.next_queue[2],
                    (int)g_world.matrix_state.next_queue[3],
                    (int)g_world.matrix_state.next_queue[4],
                    g_world.matrix_state.rng_state,
                    g_world.fx.time, g_world.fx.env_dusk, g_world.fx.env_neon,
                    g_world.fx.env_finale, g_world.fx.mood_intensity,
                    g_world.fx.camera_pulse, g_world.fx.camera_shake);
                for (int y = matrix::GRID_H - 1; y >= 0; --y) {
                    std::fprintf(stderr, "[grid] %02d ", y);
                    for (int x = 0; x < matrix::GRID_W; ++x)
                        std::fprintf(stderr, "%X",
                                     g_world.matrix_state.grid[y][x] & 0xF);
                    std::fprintf(stderr, "\n");
                }
                std::fflush(stderr);
            }
            std::printf("[dbg] frame=%d gridsum=%llu active=%d pos=%d,%d "
                        "nq=%d,%d,%d rng=%u tris=%zu trisum=%llu\n",
                        frame, sum, (int)g_world.matrix_state.active.type,
                        g_world.matrix_state.active.pos.x,
                        g_world.matrix_state.active.pos.y,
                        (int)g_world.matrix_state.next_queue[0],
                        (int)g_world.matrix_state.next_queue[1],
                        (int)g_world.matrix_state.next_queue[2],
                        g_world.matrix_state.rng_state,
                        tris.size(), fsum);
            std::fflush(stdout);
        }

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