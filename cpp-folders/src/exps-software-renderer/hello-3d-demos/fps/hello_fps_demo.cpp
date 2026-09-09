// SDL on Windows redefines `main` to `SDL_main`; SDL_MAIN_HANDLED opts out so
// the plain main() below links (matches snake/tetris/plane demos).
#define SDL_MAIN_HANDLED

// ============================================================================
// HelloFPSDemo — MAIN ENTRY EDGE
// Owns: SDL lifecycle (window/audio), per-frame PMR arena, the game loop
// wiring, and presentation. All simulation/render/HUD logic lives in pods.
//
// Headless verification hooks (deterministic, display-less):
//   --screenshot <path.bmp>   render N frames, save BMP, exit (no window)
//   --frame=N                 frame count for the above (default 60)
//   --autodrive-fire          inject ONE synthetic FireIntent at frame 30
// ============================================================================

#include <SDL2/SDL.h>

#include <algorithm>
#include <iostream>
#include <memory_resource>
#include <span>
#include <sstream>
#include <string>
#include <thread>

#include "shs_renderer.hpp"

#include <config/difficulty.hpp>
#include <config/levels/fps_level_01.hpp>

#include <domains/matrix/fps.reducer.hpp>
#include <domains/progression/fps.reducer.hpp>
#include <domains/spatial_fx/fps.plan.hpp>

#include <edges/input/fps.input.hpp>
#include <edges/audio/fps.audio.hpp>
#include <edges/rasterizer/fps.rasterizer.hpp>
#include <edges/ui/fps.hud.hpp>

namespace {

    // All pods live under fps:: — pull them in for concise main-edge wiring.
    using namespace fps;

    // Audio synth instance lives at file scope: the SDL callback thread
    // dereferences it for the lifetime of the audio device.
    audio::FpsSoundSynth g_synth;

    constexpr int WINDOW_WIDTH  = 1280;
    constexpr int WINDOW_HEIGHT = 720;
    constexpr int CANVAS_WIDTH  = 960;
    constexpr int CANVAS_HEIGHT = 540;
    constexpr int TILE_SIZE_X   = 64;
    constexpr int TILE_SIZE_Y   = 64;
    constexpr float Z_NEAR      = 0.1f;
    constexpr float Z_FAR       = 200.0f;
    constexpr float FOV_DEGREES = 75.0f;

    unsigned thread_count() {
        const unsigned hw = std::thread::hardware_concurrency();
        return std::max(1u, hw > 1 ? hw - 1 : hw);
    }

    // Per-frame linear PMR arena (O(1) reset). Kept per-demo until hoisted
    // into the shared renderer library (see docs/STATUS.md remaining work).
    class FrameMemoryResource final : public std::pmr::memory_resource {
    public:
        FrameMemoryResource() : buffer_(std::make_unique<std::byte[]>(kCapacity)) {
        }

        void reset() { offset_ = 0; }

        std::pmr::memory_resource* get() { return this; }

    protected:
        void* do_allocate(size_t bytes, size_t alignment) override {
            auto aligned = [](size_t v, size_t a) { return (v + a - 1) & ~(a - 1); };
            const size_t base = aligned(offset_, alignment);
            if (base + bytes > kCapacity) {
                throw std::bad_alloc();
            }
            offset_ = base + bytes;
            return buffer_.get() + base;
        }

        void do_deallocate(void*, size_t, size_t) override {}

        bool do_is_equal(const memory_resource& other) const noexcept override {
            return this == &other;
        }

    private:
        static constexpr size_t kCapacity = 8ull * 1024ull * 1024ull;
        std::unique_ptr<std::byte[]> buffer_;
        size_t offset_ = 0;
    };

} // namespace

int main(int argc, char* argv[]) {
    // --- CLI parsing ----------------------------------------------------------
    std::string screenshot_path;
    int         screenshot_frame = -1;
    bool        autodrive_fire   = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--screenshot" && i + 1 < argc) {
            screenshot_path = argv[++i];
            screenshot_frame = 60;
        } else if (arg.rfind("--frame=", 0) == 0) {
            screenshot_frame = std::atoi(arg.c_str() + 8);
        } else if (arg == "--autodrive-fire") {
            autodrive_fire = true;
        }
    }
    const bool headless = (screenshot_frame >= 0);

    // --- Config -----------------------------------------------------------------
    const config::Difficulty  diff{};
    const config::FpsLevel01  level{};

    // --- SDL lifecycle ------------------------------------------------------------
    SDL_setenv("PULSE_LATENCY_MSEC", "60", 1);
    SDL_SetHintWithPriority(SDL_HINT_MOUSE_RELATIVE_MODE_WARP, "1", SDL_HINT_OVERRIDE);
    SDL_SetHint(SDL_HINT_GRAB_KEYBOARD, "1");

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
        window = SDL_CreateWindow("SHS Renderer - VOP Low-Poly FPS Combat Arena",
                                  SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                  WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
        sdl_renderer   = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
        screen_texture = SDL_CreateTexture(sdl_renderer, SDL_PIXELFORMAT_RGBA32,
                                           SDL_TEXTUREACCESS_STREAMING, CANVAS_WIDTH, CANVAS_HEIGHT);
        screen_surface = SDL_CreateRGBSurfaceWithFormat(0, CANVAS_WIDTH, CANVAS_HEIGHT, 32,
                                                        SDL_PIXELFORMAT_RGBA32);

        SDL_AudioSpec want{}, have{};
        want.freq     = 44100;
        want.format   = AUDIO_F32SYS;
        want.channels = 2;
        want.samples  = 4096;
        want.callback = audio::fps_audio_callback;
        want.userdata = &g_synth;

        audio_dev = SDL_OpenAudioDevice(nullptr, 0, &want, &have, 0);
        if (audio_dev) SDL_PauseAudioDevice(audio_dev, 0);

        SDL_SetRelativeMouseMode(SDL_TRUE);
    }

    // --- Renderer state ---------------------------------------------------------
    shs::Canvas  canvas(CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{ 20, 25, 35, 255 });
    shs::ZBuffer z_buffer(CANVAS_WIDTH, CANVAS_HEIGHT, Z_NEAR, Z_FAR);

    shs::Job::ThreadedPriorityJobSystem job_system(static_cast<int>(thread_count()));
    shs::Job::WaitGroup                 wg_render;

    std::cout << "Building 3D models..." << std::endl;
    const std::vector<spatial_fx::LowPolyTriangle> arena_mesh      = spatial_fx::build_arena_mesh(level);
    const std::vector<spatial_fx::LowPolyTriangle> bot_mesh_normal = spatial_fx::build_bot_mesh(false);
    const std::vector<spatial_fx::LowPolyTriangle> bot_mesh_flash  = spatial_fx::build_bot_mesh(true);
    const std::vector<spatial_fx::LowPolyTriangle> gun_mesh        = spatial_fx::build_gun_mesh();
    const std::vector<spatial_fx::LowPolyTriangle> flash_mesh      = spatial_fx::build_muzzle_flash();
    const std::vector<spatial_fx::LowPolyTriangle> bolt_mesh       = spatial_fx::build_projectile_mesh();

    const spatial_fx::SceneMeshes scene_meshes{
        .arena        = &arena_mesh,
        .bot_normal   = &bot_mesh_normal,
        .bot_flash    = &bot_mesh_flash,
        .gun          = &gun_mesh,
        .muzzle_flash = &flash_mesh,
        .bolt         = &bolt_mesh,
    };

    FrameMemoryResource frame_memory;

    // --- Initial world state -------------------------------------------------------
    matrix::WorldSnapshot current_world(std::pmr::get_default_resource());
    for (const auto& spawn : level.bot_spawns) {
        current_world.bots.add_bot(spawn.position, spawn.waypoint);
    }
    current_world.rng_state = diff.rng_seed;

    progression::ScoreState score_state = progression::ScoreState::fresh();

    float hitmarker_timer = 0.0f;
    bool  quit            = false;
    int   frame           = 0;

    Uint32 last_tick  = SDL_GetTicks();
    int    fps_frames = 0;
    float  fps_timer  = 0.0f;

    // --- Main loop -------------------------------------------------------------------
    while (!quit) {
        float dt;
        if (headless) {
            dt = 1.0f / 60.0f; // deterministic stepping for screenshots
        } else {
            const Uint32 current_tick = SDL_GetTicks();
            dt = static_cast<float>(current_tick - last_tick) / 1000.0f;
            last_tick = current_tick;
            if (dt > 0.1f) dt = 0.1f;
        }

        frame_memory.reset();
        std::pmr::memory_resource* arena = frame_memory.get();

        // Input edge
        input::InputState in = input::poll_input();
        quit = quit || in.quit_requested;
        if (autodrive_fire && frame == 30) in.fire_pressed = true;

        const std::pmr::vector<matrix::UserCommand> commands =
            input::reduce_input(in, diff, dt, arena);

        // Pure simulation core
        matrix::WorldStepResult step = matrix::reduce_world(
            current_world,
            std::span<const matrix::UserCommand>(commands.data(), commands.size()),
            diff, level, dt, arena
        );
        current_world = std::move(step.next_world);

        // Progression pod (event-driven scoring)
        score_state = progression::reduce_progression(step.events, score_state);

        // Hitmarker UI timer: derived from events here (single owner)
        bool got_hit = false;
        for (const auto& ev : step.events) {
            if (ev.type == matrix::EventType::BOT_HIT) got_hit = true;
        }
        if (got_hit) hitmarker_timer = diff.hitmarker_time;
        else if (hitmarker_timer > 0.0f) hitmarker_timer -= dt;

        // Audio edge mapping (windowed mode only)
        if (!headless) {
            for (const auto& ev : step.events) {
                switch (ev.type) {
                case matrix::EventType::PLAYER_FIRED:   g_synth.play(audio::SoundType::PLAYER_SHOOT);  break;
                case matrix::EventType::BOT_FIRED:      g_synth.play(audio::SoundType::ENEMY_SHOOT);   break;
                case matrix::EventType::BOT_HIT:        g_synth.play(audio::SoundType::HITMARKER);     break;
                case matrix::EventType::BOT_KILLED:     g_synth.play(audio::SoundType::ENEMY_EXPLODE); break;
                case matrix::EventType::PLAYER_DAMAGED: g_synth.play(audio::SoundType::PLAYER_HURT);   break;
                case matrix::EventType::PLAYER_JUMPED:  g_synth.play(audio::SoundType::PLAYER_JUMP);   break;
                }
            }
        }

        // Render planner (pure)
        spatial_fx::PipelineExecutionPlan plan = spatial_fx::plan_fps_scene(
            current_world, scene_meshes, level.sun_dir_world,
            FOV_DEGREES, Z_NEAR, Z_FAR,
            CANVAS_WIDTH, CANVAS_HEIGHT, arena
        );

        // Tiled multithreaded rasterization
        canvas.buffer().clear(shs::Color{ 22, 28, 38, 255 });
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
                    raster::TileRasterContract contract{
                        .active_triangles = std::span<const spatial_fx::ProcessedTriangle>(
                            plan.triangles.data(), plan.triangles.size()),
                        .tile_min = glm::ivec2(tx * TILE_SIZE_X, ty * TILE_SIZE_Y),
                        .tile_max = glm::ivec2(std::min((tx + 1) * TILE_SIZE_X, W) - 1,
                                               std::min((ty + 1) * TILE_SIZE_Y, H) - 1),
                        .canvas_w = W,
                        .canvas_h = H
                    };
                    raster::execute_tile_raster_job(canvas, z_buffer, contract);
                    wg_render.done();
                }, shs::Job::PRIORITY_HIGH });
            }
        }
        wg_render.wait();

        // UI edge overlays
        ui::draw_tracers(canvas, plan.vp_matrix, current_world.tracers);
        ui::draw_enemy_health_bars(canvas, plan.vp_matrix, current_world.bots);
        ui::draw_fps_hud(canvas, current_world.player, hitmarker_timer, score_state.score);

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

        // Present
        shs::Canvas::copy_to_SDLSurface(screen_surface, &canvas);
        SDL_UpdateTexture(screen_texture, nullptr, screen_surface->pixels, screen_surface->pitch);
        SDL_RenderClear(sdl_renderer);
        SDL_RenderCopy(sdl_renderer, screen_texture, nullptr, nullptr);
        SDL_RenderPresent(sdl_renderer);

        // Window-title stats
        ++fps_frames;
        fps_timer += dt;
        if (fps_timer >= 0.5f) {
            int alive_count = 0;
            for (size_t i = 0; i < current_world.bots.size(); ++i) {
                if (current_world.bots.state[i] != matrix::BotState::DEAD) alive_count++;
            }

            std::ostringstream ss;
            ss << "VOP FPS Arena | FPS: " << static_cast<int>(static_cast<float>(fps_frames) / fps_timer)
               << " | Blood: " << current_world.player.hp
               << " | Score: " << score_state.score
               << " | Deleted: " << score_state.kills
               << " | Enemies Alive: " << alive_count << "/" << config::FpsLevel01::BOT_COUNT
               << " | [WASD: move, Space: Jump, F/LMB: Shoot, Mouse: view]";
            SDL_SetWindowTitle(window, ss.str().c_str());
            fps_frames = 0;
            fps_timer  = 0.0f;
        }
    }

    // --- Cleanup ---------------------------------------------------------------------
    if (audio_dev) SDL_CloseAudioDevice(audio_dev);
    if (screen_surface) SDL_FreeSurface(screen_surface);
    if (screen_texture) SDL_DestroyTexture(screen_texture);
    if (sdl_renderer)   SDL_DestroyRenderer(sdl_renderer);
    if (window)         SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}