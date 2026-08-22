// SDL on Windows redefines `main` to `SDL_main`; SDL_MAIN_HANDLED opts out
// so the plain main() below links (matches snake/tetris/plane demos).
#define SDL_MAIN_HANDLED

// ============================================================================
// Hello3DTetris — MAIN ENTRY EDGE
// Owns: SDL lifecycle (window/audio), per-frame PMR arena, loop wiring,
// presentation, and the event→sound map. All simulation/render/HUD logic
// lives in domain pods and execution edges.
//
// Campaign + scripting (L1/L2):
//   --stage=N                 select campaign stage (1 = MARATHON, 2 = BLITZ 120)
//   --script=<file>           override the stage's Lua rule script
//   --expect-target-score=N   smoke gate: assert the wired script overrode the
//                             target score, print SMOKE_TARGET_SCORE=PASS/FAIL
//
// Headless verification hooks (deterministic, display-less):
//   --screenshot <path.bmp>   render N frames, save BMP, exit (no window)
//   --frame=N                 frame count for the above (default 60)
//   --autodrive-harddrop      inject ONE synthetic HardDropIntent at frame 30
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
#include <config/campaign/main_campaign.hpp>

#include <domains/matrix/matrix.contract.hpp>
#include <domains/matrix/matrix.action.hpp>
#include <domains/matrix/matrix.reducer.hpp>
#include <domains/progression/progression.reducer.hpp>
#include <domains/spatial_fx/spatial_fx.reducer.hpp>
#include <domains/spatial_fx/spatial_fx.plan.hpp>

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
        } else if (arg.rfind("--expect-target-score=", 0) == 0) {
            expect_target = std::atoll(arg.c_str() + 22);
        }
    }
    const bool headless = (screenshot_frame >= 0);

    // --- Campaign stage selection (M2 manifest) ---------------------------------
    const config::campaign::Stage* stage = config::campaign::find_stage(stage_number);
    if (!stage) {
        std::cerr << "Unknown campaign stage: " << stage_number << std::endl;
        return 1;
    }
    config::Rules rules_cfg = stage->make_rules();

    // --- Lua rule-script boot (value-in/value-out; native C++ fallback) ----------
    progression::ScriptHooks script_hooks{};
#ifdef TETRIS_LUA_ENABLED
    lua_edge::StatelessLuaEvaluator lua_eval;
#endif
    {
        std::string script_path;
        if (!script_override.empty()) {
            script_path = script_override;
        } else if (stage->script_path[0] != '\0') {
            script_path = std::string(TETRIS_SOURCE_ROOT) + "/" + stage->script_path;
        }

        if (!script_path.empty()) {
#ifdef TETRIS_LUA_ENABLED
            if (lua_eval.valid() && lua_eval.load_script_file(script_path.c_str())) {
                g_lua_eval = &lua_eval;
                if (lua_eval.has_function("BlitzRules", "calculate_score")) {
                    script_hooks.line_clear_score = &bridge_line_clear_score;
                }
                if (lua_eval.has_function("BlitzRules", "evaluate_clock")) {
                    script_hooks.clock_rule = &bridge_clock_rule;
                }
                lua_eval.apply_config_overrides(rules_cfg);   // economy overrides
                std::cout << "[lua] rule script active: " << script_path << std::endl;
            } else {
                std::cerr << "[lua] script unavailable (" << script_path
                          << ") — native C++ rules active" << std::endl;
            }
#else
            std::cerr << "[lua] built without Lua — native C++ rules active" << std::endl;
#endif
        }
    }

    // --- Smoke gate: wired script must have overridden the default target --------
    if (expect_target >= 0) {
        const bool pass = (rules_cfg.target_score == static_cast<int>(expect_target));
        std::cout << "SMOKE_TARGET_SCORE=" << (pass ? "PASS" : "FAIL")
                  << " (expected=" << expect_target
                  << ", actual=" << rules_cfg.target_score << ")" << std::endl;
        return pass ? 0 : 2;
    }

    const config::Rules rules = rules_cfg;

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

        // Restart preservation: high score survives a manual reset (main-edge duty)
        bool restart_requested = false;
        for (const auto& cmd : in.commands) {
            if (std::holds_alternative<matrix::RestartIntent>(cmd)) restart_requested = true;
        }

        // Blitz time-up freezes the run: no commands reach the matrix, no gravity.
        const bool frozen = score_state.time_up;
        const auto cmd_span = frozen
            ? std::span<const matrix::TetrisCommand>()
            : std::span<const matrix::TetrisCommand>(in.commands.data(), in.commands.size());
        const float matrix_dt = frozen ? 0.0f : dt;

        // Gravity cadence wired from progression level through pure config math
        world.drop_interval = rules.gravity_for_level(score_state.level);

        // 2. PURE SIMULATION CORE
        matrix::MatrixStepResult step = matrix::reduce_matrix(world, cmd_span, matrix_dt, arena);
        world = std::move(step.next_state);

        // 3. EVENT-FED PROGRESSION (+ blitz clock via injected rule hooks)
        if (restart_requested) {
            const int preserved_high = score_state.high_score;
            score_state = progression::ScoreState{};
            score_state.target_score = rules.target_score;
            score_state.mode_id      = rules.mode_id;
            score_state.time_left    = rules.time_limit;
            score_state.high_score   = preserved_high;
        }
        progression::ProgressionStep prog = progression::reduce_progression(
            std::span<const matrix::MatrixEvent>(step.events.data(), step.events.size()),
            score_state, rules, arena,
            dt, compute_stack_height(world), script_hooks
        );
        score_state = std::move(prog.next);

        // Environment mood wire (pod-5 embryo): amber intensity rises as the
        // blitz clock drains; untimed modes stay neutral.
        fx.mood_intensity = (rules.time_limit > 0.0f)
            ? glm::clamp(1.0f - score_state.time_left / rules.time_limit, 0.0f, 1.0f)
            : 0.0f;

        // 4. FX STEP (particles + rings + camera spring/pulse, deterministic xorshift)
        spatial_fx::step_fx(fx,
            std::span<const matrix::MatrixEvent>(step.events.data(), step.events.size()),
            std::span<const progression::ProgressionEvent>(prog.events.data(), prog.events.size()),
            dt);

        // HUD transient presentation state (banners/floaters)
        ui::step_hud(hud,
            std::span<const progression::ProgressionEvent>(prog.events.data(), prog.events.size()),
            dt);

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
        }

        // 5. PURE SCENE PLANNER
        spatial_fx::PipelineExecutionPlan plan = spatial_fx::plan_tetris_scene(
            world, fx, CANVAS_WIDTH, CANVAS_HEIGHT, arena
        );

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
                                                         tri.lit_color, tri.depth_bias, tmin, tmax);
                        }
                    }
                    wg_render.done();
                }, shs::Job::PRIORITY_HIGH });
            }
        }
        wg_render.wait();

        // 7. UI EDGE
        ui::draw_hud(canvas, world, score_state, hud);

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