#include <SDL2/SDL.h>
#include <string>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <random>
#include <vector>
#include "shs_renderer.hpp"

#define FRAMES_PER_SECOND 60
#define WINDOW_WIDTH      640
#define WINDOW_HEIGHT     480

#define CANVAS_WIDTH      320
#define CANVAS_HEIGHT     240

struct MovingCircle {
    float cx, cy;
    float vx, vy;
    int   r;
    int   segments;
    shs::Color color;
};

static shs::Color random_color(std::mt19937 &rng)
{
    std::uniform_int_distribution<int> dist(0, 255);
    return shs::Color((uint8_t)dist(rng), (uint8_t)dist(rng), (uint8_t)dist(rng), 255);
}

static float randf(std::mt19937 &rng, float a, float b)
{
    std::uniform_real_distribution<float> dist(a, b);
    return dist(rng);
}

static int randi(std::mt19937 &rng, int a, int b)
{
    std::uniform_int_distribution<int> dist(a, b);
    return dist(rng);
}

int main()
{
    SDL_Window   *window   = nullptr;
    SDL_Renderer *renderer = nullptr;

    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, 0, &window, &renderer);
    SDL_RenderSetScale(renderer, 1, 1);

    shs::Canvas *main_canvas     = new shs::Canvas(CANVAS_WIDTH, CANVAS_HEIGHT);
    SDL_Surface *main_sdlsurface = main_canvas->create_sdl_surface();
    SDL_Texture *screen_texture  = SDL_CreateTextureFromSurface(renderer, main_sdlsurface);

    bool exit = false;
    SDL_Event event_data;

    int   frame_delay            = 1000 / FRAMES_PER_SECOND;
    float frame_time_accumulator = 0.0f;
    int   frame_counter          = 0;

    std::mt19937 rng((uint32_t)std::time(nullptr));

    int circle_count = randi(rng, 3, 4);

    std::vector<MovingCircle> circles;
    circles.reserve((size_t)circle_count);

    for (int i = 0; i < circle_count; i++)
    {
        MovingCircle c;
        c.r        = randi(rng, 12, 50);
        c.segments = randi(rng, 36, 140);
        c.color    = random_color(rng);

        c.cx = randf(rng, (float)c.r, (float)(CANVAS_WIDTH  - 1 - c.r));
        c.cy = randf(rng, (float)c.r, (float)(CANVAS_HEIGHT - 1 - c.r));

        // px/sec хурд (canvas пиксел/секунд)
        c.vx = randf(rng, -120.0f, 120.0f);
        c.vy = randf(rng, -120.0f, 120.0f);

        // 0 хурдтай битгий үлдээгээрэй
        if (std::abs(c.vx) < 20.0f) c.vx = (c.vx < 0 ? -40.0f : 40.0f);
        if (std::abs(c.vy) < 20.0f) c.vy = (c.vy < 0 ? -40.0f : 40.0f);

        circles.push_back(c);
    }

    Uint32 last_ticks = SDL_GetTicks();

    while (!exit)
    {
        Uint32 frame_start_ticks = SDL_GetTicks();

        while (SDL_PollEvent(&event_data))
        {
            switch (event_data.type)
            {
            case SDL_QUIT:
                exit = true;
                break;
            case SDL_KEYDOWN:
                if (event_data.key.keysym.sym == SDLK_ESCAPE) exit = true;
                break;
            }
        }

        // dt секунд
        Uint32 now_ticks = SDL_GetTicks();
        float dt = (now_ticks - last_ticks) / 1000.0f;
        last_ticks = now_ticks;
        if (dt > 0.05f) dt = 0.05f; // tab-out үед хэт үсрэхээс хамгаалалт

        // --- physics: move + bounce ---
        for (auto &c : circles)
        {
            c.cx += c.vx * dt;
            c.cy += c.vy * dt;

            float minx = (float)c.r;
            float maxx = (float)(CANVAS_WIDTH - 1 - c.r);
            float miny = (float)c.r;
            float maxy = (float)(CANVAS_HEIGHT - 1 - c.r);

            if (c.cx < minx) { c.cx = minx; c.vx = -c.vx; }
            if (c.cx > maxx) { c.cx = maxx; c.vx = -c.vx; }
            if (c.cy < miny) { c.cy = miny; c.vy = -c.vy; }
            if (c.cy > maxy) { c.cy = maxy; c.vy = -c.vy; }
        }

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        shs::Canvas::fill_pixel(*main_canvas, 0, 0, CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color::black());

        for (const auto &c : circles)
        {
            shs::Canvas::draw_circle_poly(
                *main_canvas,
                (int)std::lround(c.cx),
                (int)std::lround(c.cy),
                c.r,
                c.segments,
                c.color
            );
        }

        shs::Canvas::copy_to_SDLSurface(main_sdlsurface, main_canvas);
        SDL_UpdateTexture(screen_texture, NULL, main_sdlsurface->pixels, main_sdlsurface->pitch);

        SDL_Rect destination_rect{0, 0, WINDOW_WIDTH, WINDOW_HEIGHT};
        SDL_RenderCopy(renderer, screen_texture, NULL, &destination_rect);
        SDL_RenderPresent(renderer);

        frame_counter++;
        Uint32 delta_frame_time = SDL_GetTicks() - frame_start_ticks;

        frame_time_accumulator += delta_frame_time / 1000.0f;
        if (delta_frame_time < (Uint32)frame_delay) {
            SDL_Delay((Uint32)frame_delay - delta_frame_time);
        }
        if (frame_time_accumulator >= 1.0f) {
            std::string window_title = "FPS : " + std::to_string(frame_counter);
            frame_time_accumulator   = 0.0f;
            frame_counter            = 0;
            SDL_SetWindowTitle(window, window_title.c_str());
        }
    }

    delete main_canvas;

    SDL_DestroyTexture(screen_texture);
    SDL_FreeSurface(main_sdlsurface);

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
