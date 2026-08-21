#define SDL_MAIN_HANDLED

// hello-3d-snake — semi-3D low-poly snake demo. Mirrors tetris's proven renderer API: world-space boxes →
// vp transform → pre-shaded triangles (PipelineExecutionPlan) → per-tile rasterization with frustum clip +
// barycentric depth test. No Camera3D / draw_triangle_flat_shading; the tiled path is the canonical one.
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <sstream>
#include <atomic>
#include <thread>
#include <span>
#include <memory_resource>

#include <SDL2/SDL.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "shs_renderer.hpp"
#include "domains/spatial_fx/snake.plan.hpp"   // plan_snake_scene, ShatterParticleSoA
#include "domains/matrix/snake.contract.hpp"    // SnakeSnapshot, SnakeCommandType, etc.
#include "domains/progression/snake.contract.hpp"  // ScoreState
#include "edges/input/snake.input.hpp"          // reduce_input

namespace vop {
    class FrameMemoryResource : public std::pmr::memory_resource {
    public:
        static constexpr size_t CAPACITY = 8 * 1024 * 1024;   // 8 MB
        FrameMemoryResource() : buffer_(std::make_unique<uint8_t[]>(CAPACITY)), offset_(0) {}
        inline void reset() noexcept { offset_ = 0; }
        inline std::pmr::memory_resource* get() noexcept { return this; }
    protected:
        void* do_allocate(size_t bytes, size_t alignment) override {
            uintptr_t base = reinterpret_cast<uintptr_t>(buffer_.get());
            uintptr_t current_addr = base + offset_;
            uintptr_t aligned_addr = (current_addr + (alignment - 1)) & ~(alignment - 1);
            size_t new_offset = (aligned_addr - base) + bytes;
            if (new_offset > CAPACITY) return std::pmr::get_default_resource()->allocate(bytes, alignment);
            offset_ = new_offset;
            return reinterpret_cast<void*>(aligned_addr);
        }
        void do_deallocate(void* p, size_t bytes, size_t alignment) noexcept override {
            uintptr_t base = reinterpret_cast<uintptr_t>(buffer_.get());
            uintptr_t ptr  = reinterpret_cast<uintptr_t>(p);
            if (ptr < base || ptr >= base + CAPACITY) std::pmr::get_default_resource()->deallocate(p, bytes, alignment);
        }
        bool do_is_equal(const std::pmr::memory_resource& other) const noexcept { return this == &other; }
    private:
        std::unique_ptr<uint8_t[]> buffer_;
        size_t offset_ = 0;
    };
}

// ============================================================================
// TILED RASTERIZER HELPERS (mirror shs_renderer rasterization path)
// ============================================================================
static inline glm::vec4 clip_to_screen_vec4(const glm::vec4& clip, int W, int H) {
    float inv_w = 1.0f / clip.w;
    glm::vec3 ndc = glm::vec3(clip) * inv_w;
    return glm::vec4(
        (ndc.x + 1.0f) * 0.5f * (float)(W - 1),
        (1.0f - ndc.y) * 0.5f * (float)(H - 1),
        ndc.z,
        inv_w
    );
}

static void rasterize_triangle_tile(
    shs::Canvas& canvas, shs::ZBuffer& z_buffer,
    const glm::vec4& sc0, const glm::vec4& sc1, const glm::vec4& sc2,
    shs::Color lit_color, float depth_bias,
    glm::ivec2 tile_min, glm::ivec2 tile_max)
{
    glm::vec2 v0(sc0.x, sc0.y), v1(sc1.x, sc1.y), v2(sc2.x, sc2.y);
    float area = (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);
    if (!shs::Raster::is_front_facing_screen(area, shs::Raster::FrontFace::CounterClockwise)) return;

    glm::vec2 bboxmin = glm::max(glm::vec2(tile_min), glm::min(v0, glm::min(v1, v2)));
    glm::vec2 bboxmax = glm::min(glm::vec2(tile_max), glm::max(v0, glm::max(v1, v2)));
    if (bboxmin.x > bboxmax.x || bboxmin.y > bboxmax.y) return;

    std::vector<glm::vec2> v2d = { v0, v1, v2 };
    int min_x = (int)bboxmin.x, max_x = (int)bboxmax.x;
    int min_y = (int)bboxmin.y, max_y = (int)bboxmax.y;

    for (int py = min_y; py <= max_y; ++py) {
        for (int px = min_x; px <= max_x; ++px) {
            glm::vec3 bc = shs::Canvas::barycentric_coordinate(glm::vec2((float)px + 0.5f, (float)py + 0.5f), v2d);
            if (bc.x < 0.0f || bc.y < 0.0f || bc.z < 0.0f) continue;

            float interp_z = shs::Raster::interpolate_ndc_depth(bc, sc0.z, sc1.z, sc2.z);
            float final_z  = interp_z + depth_bias;
            if (final_z < -1.0f || final_z > 1.0f) continue;

            if (z_buffer.test_and_set_depth_screen_space(px, py, final_z)) {
                canvas.draw_pixel_screen_space(px, py, lit_color);
            }
        }
    }
}

// ============================================================================
// MAIN ENTRY EDGE
// ============================================================================
int main(int argc, char* argv[]) {
    (void)argc; (void)argv;

    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_TIMER) < 0) {
        std::cerr << "SDL_Init failed: " << SDL_GetError() << std::endl;
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow(
        "SHS Renderer - Semi-3D Low-Poly Snake",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        CANVAS_WIDTH, CANVAS_HEIGHT, SDL_WINDOW_SHOWN
    );
    if (!window) { std::cerr << "SDL window failed: " << SDL_GetError() << "\n"; return 1; }

    SDL_Renderer* sdl_renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture* screen_texture = SDL_CreateTexture(
        sdl_renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING,
        CANVAS_WIDTH, CANVAS_HEIGHT
    );
    SDL_Surface* screen_surface = SDL_CreateRGBSurfaceWithFormat(0, CANVAS_WIDTH, CANVAS_HEIGHT, 32, SDL_PIXELFORMAT_RGBA32);

    shs::Canvas   canvas(CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{ 14, 16, 22, 255 });
    shs::ZBuffer  z_buffer(CANVAS_WIDTH, CANVAS_HEIGHT, -1.0f, 1.0f);

    snake::config::Difficulty difficulty;   // default: soft walls (bounce)
    snake::matrix::SnakeSnapshot snap = snake::matrix::SnakeSnapshot{};   // initial spawn from level data
    const snake::SnakeLevel01     level{};   // arena bounds + food table (level 01)
    snake::progression::ScoreState score_state = snake::progression::ScoreState::fresh();

    ShatterParticleSoA particles(std::pmr::get_default_resource());

    static const int THREAD_COUNT = std::max(2u, std::thread::hardware_concurrency() > 2 ? std::thread::hardware_concurrency() - 2 : 2u);
    shs::Job::ThreadedPriorityJobSystem job_system(THREAD_COUNT);
    shs::Job::WaitGroup                 wg_render;

    vop::FrameMemoryResource frame_memory;

    float time_sec = 0.0f;       // orbiting camera timer

    bool quit = false;
    SDL_Event e;
    Uint32 last_tick = SDL_GetTicks();

    while (!quit) {
        Uint32 cur_tick = SDL_GetTicks();
        float dt = (cur_tick - last_tick) / 1000.0f;
        last_tick = cur_tick;
        if (dt > 0.05f) dt = 0.05f;

        time_sec += dt;   // advance orbiting camera timer

        frame_memory.reset();
        auto* arena = frame_memory.get();

        // 1. INPUT POLLING → matrix commands
        snake::input::InputState input_state{};
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) quit = true;
            if (e.type == SDL_KEYDOWN) {
                if (e.key.keysym.sym == SDLK_ESCAPE) quit = true;
                if (e.key.keysym.sym == SDLK_LEFT)  input_state.turn_left  = true;
                if (e.key.keysym.sym == SDLK_RIGHT) input_state.turn_right = true;
                if (e.key.keysym.sym == SDLK_UP)    input_state.strafe_up  = true;
                if (e.key.keysym.sym == SDLK_DOWN)  input_state.strafe_down = true;
            }
        }

        // 2. PURE REDUCTION → next state + events (input edge returns the command vector)
        auto commands = snake::input::reduce_input(input_state, arena);   // pmr::vector<SnakeCommand> from raw input
        auto step = snake::matrix::reduce_snake(snap, commands, difficulty, level);
        snap = step.next_state;

        for (const auto& ev : step.events) {
            switch (ev.type) {
                case snake::matrix::SnakeEventType::FOOD_EATEN:   score_state.score += 10; break;
                case snake::matrix::SnakeEventType::SELF_COLLISION:
                    if (!step.alive) { score_state.high_score = std::max(score_state.high_score, score_state.score); }
                    break;
                default: break;
            }
        }

        // 3. UPDATE PARTICLES (gravity + life decay)
        for (size_t i = 0; i < particles.position.size();) {
            particles.position[i] += particles.velocity[i] * dt;
            particles.velocity[i].y -= 18.0f * dt;   // gravity
            particles.life[i] -= dt;
            if (particles.life[i] <= 0.0f) {
                particles.position.erase(particles.position.begin() + i);
                particles.velocity.erase(particles.velocity.begin() + i);
                particles.color.erase(particles.color.begin() + i);
                particles.life.erase(particles.life.begin() + i);
            } else {
                ++i;
            }
        }

        // 4. PURE 3D BATCH PLANNER → pre-shaded triangles (clip-space corners + Lambert colors)
        auto plan = snake::spatial_fx::plan_snake_scene(snap, commands, difficulty, level, particles);

        // 5. TILED PARALLEL RASTERIZATION (frustum clip + barycentric depth test per tile)
        canvas.buffer().clear(shs::Color{ 14, 16, 22, 255 });
        z_buffer.clear();

        int W    = canvas.get_width();
        int H    = canvas.get_height();
        int cols = (W + TILE_SIZE_X - 1) / TILE_SIZE_X;
        int rows = (H + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

        wg_render.reset();
        for (int ty = 0; ty < rows; ++ty) {
            for (int tx = 0; tx < cols; ++tx) {
                wg_render.add(1);
                job_system.submit({ [&, tx, ty, W, H]() {
                    glm::ivec2 tmin(tx * TILE_SIZE_X, ty * TILE_SIZE_Y);
                    glm::ivec2 tmax(std::min((tx + 1) * TILE_SIZE_X, W) - 1, std::min((ty + 1) * TILE_SIZE_Y, H) - 1);

                    for (const auto& tri : plan.triangles) {
                        const shs::Raster::FrustumClipPolygon poly =
                            shs::Raster::clip_triangle_to_frustum(tri.c0, tri.c1, tri.c2);
                        if (poly.count < 3) continue;

                        glm::vec4 s0 = clip_to_screen_vec4(poly.vertices[0], W, H);
                        for (int i = 1; i + 1 < poly.count; ++i) {
                            glm::vec4 s1 = clip_to_screen_vec4(poly.vertices[i], W, H);
                            glm::vec4 s2 = clip_to_screen_vec4(poly.vertices[i + 1], W, H);
                            rasterize_triangle_tile(canvas, z_buffer, s0, s1, s2, tri.lit_color, tri.depth_bias, tmin, tmax);
                        }
                    }
                    wg_render.done();
                }, shs::Job::PRIORITY_HIGH });
            }
        }
        wg_render.wait();

        // 6. SWAPCHAIN PRESENTATION
        shs::Canvas::copy_to_SDLSurface(screen_surface, &canvas);
        SDL_UpdateTexture(screen_texture, NULL, screen_surface->pixels, screen_surface->pitch);
        SDL_RenderClear(sdl_renderer);
        SDL_RenderCopy(sdl_renderer, screen_texture, NULL, NULL);
        SDL_RenderPresent(sdl_renderer);
    }

    if (sdl_renderer) SDL_DestroyRenderer(sdl_renderer);
    if (screen_texture) SDL_DestroyTexture(screen_texture);
    if (screen_surface) SDL_FreeSurface(screen_surface);
    if (window) SDL_DestroyWindow(window);
    if (SDL_WasInit(SDL_INIT_AUDIO)) SDL_QuitAudio();
    SDL_Quit();

    return 0;
}
