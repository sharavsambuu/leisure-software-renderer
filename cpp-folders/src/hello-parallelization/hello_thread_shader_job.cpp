/*
    PIXEL-BY-PIXEL SHADING + FAST DISPATCH (HEADER-ONLY)
    we submit jobs as contiguous pixel ranges (chunks), not 86,400 jobs per frame.
*/

#include <SDL2/SDL.h>
#include <glm/glm.hpp>
#include <glm/gtc/noise.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>

#include "shs_renderer.hpp"

#define FRAMES_PER_SECOND  60
#define WINDOW_WIDTH       640
#define WINDOW_HEIGHT      520
#define CANVAS_WIDTH       360
#define CANVAS_HEIGHT      240
#define CONCURRENCY_COUNT  20
#define NUM_OCTAVES        5

#define PIXELS_PER_JOB     4096

static inline glm::vec4 rescale_vec4_1_255(const glm::vec4 &input_vec) {
    return glm::clamp(input_vec, 0.0f, 1.0f) * 255.0f;
}

static float fbm(const glm::vec2& st) {
    glm::vec2 _st = st;
    float v = 0.0f;
    float a = 0.5f;
    glm::vec2 shift(100.0f);

    glm::mat2 rot(cos(0.5f), sin(0.5f),
                  -sin(0.5f), cos(0.5f));

    for (int i = 0; i < NUM_OCTAVES; ++i) {
        v += a * glm::simplex(_st);
        _st = rot * _st * 2.0f + shift;
        a *= 0.5f;
    }
    return v;
}

static glm::vec4 fragment_shader(glm::vec2 uniform_uv, float uniform_time)
{
    glm::vec2 st = (uniform_uv / glm::vec2(CANVAS_WIDTH, CANVAS_HEIGHT)) * 3.0f;
    st += float(glm::abs(glm::sin(uniform_time * 0.1f) * 3.0f)) * st;

    glm::vec2 q(0.0f);
    q.x = fbm(st + 0.00f * uniform_time);
    q.y = fbm(st + glm::vec2(1.0f));

    glm::vec2 r(0.0f);
    r.x = fbm(st + 1.0f * q + glm::vec2(1.7f, 9.2f) + 0.15f * uniform_time);
    r.y = fbm(st + 1.0f * q + glm::vec2(8.3f, 2.8f) + 0.126f * uniform_time);

    float f = fbm(st + r);

    glm::vec3 color = glm::mix(glm::vec3(0.101961f, 0.619608f, 0.666667f),
                               glm::vec3(0.666667f, 0.666667f, 0.498039f),
                               glm::clamp((f * f) * 4.0f, 0.0f, 1.0f));

    color = glm::mix(color, glm::vec3(0.0f, 0.0f, 0.164706f), glm::clamp(glm::length(q), 0.0f, 1.0f));
    color = glm::mix(color, glm::vec3(0.666667f, 1.0f, 1.0f), glm::clamp(glm::length(r.x), 0.0f, 1.0f));

    glm::vec4 out = glm::vec4(color * float(f*f*f + 0.6f*f*f + 0.5f*f), 1.0f);
    return rescale_vec4_1_255(out);
}

int main(int argc, char* argv[])
{
    (void)argc; (void)argv;

    auto *job_system = new shs::Job::ThreadedPriorityJobSystem(CONCURRENCY_COUNT);
    shs::Job::WaitGroup wg;

    SDL_Window   *window   = nullptr;
    SDL_Renderer *renderer = nullptr;

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL init failed: " << SDL_GetError() << "\n";
        return 1;
    }

    SDL_CreateWindowAndRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, 0, &window, &renderer);

    shs::Canvas *main_canvas = new shs::Canvas(CANVAS_WIDTH, CANVAS_HEIGHT);
    SDL_Surface *main_surface = main_canvas->create_sdl_surface();
    SDL_Texture *screen_texture = SDL_CreateTextureFromSurface(renderer, main_surface);

    bool exit_loop = false;
    SDL_Event ev;

    int frame_delay = 1000 / FRAMES_PER_SECOND;
    float time_accumulator = 0.0f;

    int frame_counter = 0;
    float fps_timer = 0.0f;

    while (!exit_loop)
    {
        Uint32 frame_start = SDL_GetTicks();

        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) exit_loop = true;
            if (ev.type == SDL_KEYDOWN && ev.key.keysym.sym == SDLK_ESCAPE) exit_loop = true;
        }
        if (exit_loop) break;

        shs::Canvas::fill_pixel(*main_canvas, 0, 0, CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{0, 0, 0, 255});

        const int W = CANVAS_WIDTH;
        const int H = CANVAS_HEIGHT;
        const int total_pixels = W * H;

        wg.reset();

        for (int base = 0; base < total_pixels; base += PIXELS_PER_JOB)
        {
            int count = std::min(PIXELS_PER_JOB, total_pixels - base);

            wg.add(1);

            job_system->submit({[base, count, W, time_accumulator, main_canvas, &wg]() {

                for (int k = 0; k < count; ++k) {
                    int idx = base + k;
                    int x = idx % W;
                    int y = idx / W;

                    glm::vec2 uv = { (float)x, (float)y };
                    glm::vec4 s  = fragment_shader(uv, time_accumulator);

                    main_canvas->draw_pixel(x, y, shs::Color{
                        (uint8_t)s[0], (uint8_t)s[1], (uint8_t)s[2], (uint8_t)s[3]
                    });
                }

                wg.done();

            }, shs::Job::PRIORITY_NORMAL});
        }

        wg.wait();

        shs::Canvas::copy_to_SDLSurface(main_surface, main_canvas);
        SDL_UpdateTexture(screen_texture, NULL, main_surface->pixels, main_surface->pitch);

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        SDL_Rect dst{0, 0, WINDOW_WIDTH, WINDOW_HEIGHT};
        SDL_RenderCopy(renderer, screen_texture, NULL, &dst);
        SDL_RenderPresent(renderer);

        Uint32 dt_ms = SDL_GetTicks() - frame_start;
        float dt_s = dt_ms / 1000.0f;
        time_accumulator += dt_s;

        frame_counter++;
        fps_timer += dt_s;
        if (fps_timer >= 1.0f) {
            SDL_SetWindowTitle(window, ("FPS: " + std::to_string(frame_counter)).c_str());
            frame_counter = 0;
            fps_timer = 0.0f;
        }

        if (dt_ms < (Uint32)frame_delay) SDL_Delay(frame_delay - dt_ms);
    }

    delete job_system;
    delete main_canvas;

    SDL_DestroyTexture(screen_texture);
    SDL_FreeSurface(main_surface);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    std::cout << "Clean exit.\n";
    return 0;
}
