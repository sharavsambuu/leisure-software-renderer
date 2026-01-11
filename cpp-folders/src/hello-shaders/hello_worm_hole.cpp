/*
    HELLO WORMHOLE 
*/

#include <SDL2/SDL.h>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <algorithm>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>

#include "shs_renderer.hpp"

#define FRAMES_PER_SECOND  60
#define WINDOW_WIDTH       700
#define WINDOW_HEIGHT      500
#define CANVAS_WIDTH       380
#define CANVAS_HEIGHT      280

#define THREAD_COUNT       16
#define TILE_SIZE_X        40
#define TILE_SIZE_Y        40

static inline float clampf(float v, float lo, float hi) { return std::max(lo, std::min(hi, v)); }
static inline int   clampi(int v, int lo, int hi)       { return std::max(lo, std::min(hi, v)); }

static inline glm::vec2 rot2(glm::vec2 v, float t)
{
    float s = std::sin(t), c = std::cos(t);
    return glm::vec2(c*v.x - s*v.y, s*v.x + c*v.y);
}

static inline glm::vec3 aces_tonemap(glm::vec3 c)
{
    glm::mat3 m1(
        0.59719f, 0.07600f, 0.02840f,
        0.35458f, 0.90834f, 0.13383f,
        0.04823f, 0.01566f, 0.83777f
    );
    glm::mat3 m2(
        1.60475f, -0.10208f, -0.00327f,
        -0.53108f,  1.10813f, -0.07276f,
        -0.07367f, -0.00605f,  1.07602f
    );

    glm::vec3 v = m1 * c;
    glm::vec3 a = v * (v + 0.0245786f) - 0.000090537f;
    glm::vec3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
    return m2 * (a / b);
}

static inline float n_dotnoise(glm::vec3 p)
{
    const float PHI = 1.618033988f;

    glm::mat3 GOLD(
        -0.571464913f, +0.814921382f, +0.096597072f,
        -0.278044873f, -0.303026659f, +0.911518454f,
        +0.772087367f, +0.494042493f, +0.399753815f
    );

    glm::vec3 gp = GOLD * p;
    glm::vec3 c = glm::cos(gp);

    glm::vec3 p2 = (PHI * p);
    glm::vec3 sp = GOLD * p2;
    glm::vec3 s = glm::sin(sp);

    return glm::dot(c, s);
}

static inline glm::vec2 path(float z, float t)
{
    float A  = 2.6f;
    float B  = 1.9f;
    float k1 = 0.35f;
    float k2 = 0.22f;

    return glm::vec2(
        A * std::sin(k1*z + 0.6f*t),
        B * std::cos(k2*z + 0.4f*t + 1.2f)
    );
}

static inline glm::vec3 render_wormhole(glm::vec2 pixel_xy, float time_sec)
{
    float t = time_sec;
    float travel = t * 3.0f;

    glm::vec3 p(0.0f), l(0.0f), b(0.0f), d(0.0f);

    glm::vec3 ro(path(travel, t), travel);
    glm::vec3 la(path(travel + 3.0f, t), travel + 3.0f);

    glm::vec3 fwd = glm::normalize(la - ro);
    glm::vec3 rgt = glm::normalize(glm::cross(glm::vec3(0,1,0), fwd));
    glm::vec3 up  = glm::normalize(glm::cross(fwd, rgt));

    ro += rgt * 0.75f + up * 0.15f;

    glm::vec2 uv = (2.0f * pixel_xy - glm::vec2(CANVAS_WIDTH, CANVAS_HEIGHT)) / float(CANVAS_HEIGHT);
    uv = rot2(uv, t * 0.15f);

    d = glm::normalize(fwd + uv.x * rgt + uv.y * up);

    p = ro;

    for (int ii = 0; ii < 10; ++ii)
    {
        glm::vec2 c = path(p.z, t);

        b = p;
        b.x -= c.x;
        b.y -= c.y;

        glm::vec2 bw = rot2(glm::sin(glm::vec2(b.x, b.y)), t*1.5f + b.z*3.0f);
        b.x = bw.x;
        b.y = bw.y;

        float s = 0.001f + std::abs(n_dotnoise(b * 12.0f) / 12.0f - n_dotnoise(b)) * 0.4f;

        glm::vec2 pxy = glm::vec2(p.x, p.y) - c;
        s = std::max(s, 2.2f - glm::length(pxy));

        s += std::abs(b.y * 0.75f + std::sin(p.z + t*0.1f + b.x*1.5f)) * 0.2f;

        p += d * s * 0.9f;

        glm::vec3 phase = glm::vec3(3.0f, 1.5f, 1.0f);
        glm::vec3 add = (glm::vec3(1.0f) + glm::sin(float(ii) + glm::length(glm::vec2(b.x, b.y)*0.1f) + phase)) / s;
        l += add;
    }

    glm::vec3 col = aces_tonemap((l*l) / 300.0f);
    col = glm::clamp(col, glm::vec3(0.0f), glm::vec3(1.0f));
    return col;
}

static inline shs::Color fragment_shader(glm::vec2 pixel_xy, float time_sec)
{
    const int SAMPLES = 7;

    float span = 0.0045f + 0.0002f * glm::length(pixel_xy);
    glm::vec3 col(0.0f);

    for (int k = 0; k < SAMPLES; ++k)
    {
        float f = float(k) / float(SAMPLES - 1); // forward-only
        col += render_wormhole(pixel_xy, time_sec + f * span);
    }
    col *= (1.0f / float(SAMPLES));

    glm::vec3 rgb255 = glm::clamp(col, glm::vec3(0.0f), glm::vec3(1.0f)) * 255.0f;

    return shs::Color{
        (uint8_t)clampi((int)std::round(rgb255.r), 0, 255),
        (uint8_t)clampi((int)std::round(rgb255.g), 0, 255),
        (uint8_t)clampi((int)std::round(rgb255.b), 0, 255),
        255
    };
}

int main(int argc, char* argv[])
{
    (void)argc; (void)argv;

    if (SDL_Init(SDL_INIT_VIDEO) < 0) return 1;

    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_CreateWindowAndRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, 0, &window, &renderer);

    shs::Canvas* canvas = new shs::Canvas(CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{0,0,0,255});
    SDL_Surface* surface = canvas->create_sdl_surface();
    SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);

    auto* job_system = new shs::Job::ThreadedPriorityJobSystem(THREAD_COUNT);
    shs::Job::WaitGroup wg;

    bool exit = false;
    SDL_Event ev;

    Uint32 last_tick = SDL_GetTicks();
    float time_accum = 0.0f;

    while (!exit)
    {
        Uint32 tick = SDL_GetTicks();
        float dt = float(tick - last_tick) / 1000.0f;
        last_tick = tick;
        time_accum += dt;

        while (SDL_PollEvent(&ev))
        {
            if (ev.type == SDL_QUIT) exit = true;
            if (ev.type == SDL_KEYDOWN && ev.key.keysym.sym == SDLK_ESCAPE) exit = true;
        }

        int W = canvas->get_width();
        int H = canvas->get_height();

        int cols = (W + TILE_SIZE_X - 1) / TILE_SIZE_X;
        int rows = (H + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

        wg.reset();

        for (int ty = 0; ty < rows; ++ty)
        {
            for (int tx = 0; tx < cols; ++tx)
            {
                wg.add(1);

                job_system->submit({[&, tx, ty, W, H, time_accum]() {

                    int x0 = tx * TILE_SIZE_X;
                    int y0 = ty * TILE_SIZE_Y;
                    int x1 = std::min(x0 + TILE_SIZE_X, W);
                    int y1 = std::min(y0 + TILE_SIZE_Y, H);

                    for (int y = y0; y < y1; ++y)
                    {
                        for (int x = x0; x < x1; ++x)
                        {
                            shs::Color out = fragment_shader(glm::vec2(float(x), float(y)), time_accum);
                            canvas->draw_pixel(x, y, out);
                        }
                    }

                    wg.done();
                }, shs::Job::PRIORITY_HIGH});
            }
        }

        wg.wait();

        shs::Canvas::copy_to_SDLSurface(surface, canvas);
        SDL_UpdateTexture(texture, NULL, surface->pixels, surface->pitch);

        SDL_RenderClear(renderer);
        SDL_Rect dst{0, 0, WINDOW_WIDTH, WINDOW_HEIGHT};
        SDL_RenderCopy(renderer, texture, NULL, &dst);
        SDL_RenderPresent(renderer);

        Uint32 frame_ms = SDL_GetTicks() - tick;
        Uint32 target_ms = 1000 / FRAMES_PER_SECOND;
        if (frame_ms < target_ms) SDL_Delay(target_ms - frame_ms);

        std::string title = "Wormhole";
        SDL_SetWindowTitle(window, title.c_str());
    }

    delete job_system;

    SDL_DestroyTexture(texture);
    SDL_FreeSurface(surface);
    delete canvas;

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
