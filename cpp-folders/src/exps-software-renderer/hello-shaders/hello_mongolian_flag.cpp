/*
 *       FLAG OF MONGOLIA 
 *
 * @Author Sharavsambuu Gunchinish
 * https://github.com/sharavsambuu/leisure-software-renderer
 * 
 * References :
 * - https://en.wikipedia.org/wiki/Flag_of_Mongolia
 * - https://www.shadertoy.com/view/MsX3Wn
 *
 * Features :
 * - Fast Analytic & 72-point Continuous Spline SDF Soyombo (Correct Orientation)
 * - Anti-Aliased Vector Soyombo & Stripe Rendering
 * - Majestic Slow-motion Flag Wave (Speed = 1.65)
 * - Rich Fabric Noise, Wave Shading & Dave Hoskins Dithering
 * - Micro-weave Scanline Grid & Deep Lens Vignette
 * 
 */

#include <SDL2/SDL.h>
#include <glm/glm.hpp>
#include <glm/gtc/noise.hpp> 
#include <algorithm>
#include <string>
#include <vector>
#include <iostream>
#include <array>
#include <cstdlib>
#include <cmath>
#include <tuple>
#include <thread>
#include <mutex>
#include "shs_renderer.hpp"

#define FRAMES_PER_SECOND  60
#define WINDOW_WIDTH       860
#define WINDOW_HEIGHT      460
#define CANVAS_WIDTH       860
#define CANVAS_HEIGHT      460
#define CONCURRENCY_COUNT  8

constexpr float PI = 3.14159265358979323846f;
const glm::vec3 MONGOLIAN_RED   (0.855f, 0.125f, 0.192f);
const glm::vec3 MONGOLIAN_BLUE  (0.0f  , 0.400f, 0.698f);
const glm::vec3 MONGOLIAN_YELLOW(1.0f  , 0.851f, 0.0f  );

// Dave Hoskins Hash 
inline float hash(glm::vec2 p) {
    glm::uvec2 q = glm::uvec2(glm::ivec2(p)) * glm::uvec2(1597334673U, 3812015801U);
    uint32_t  n = (q.x ^ q.y) * 1597334673U;
    return float(n) / float(0xffffffffU);
}

// Procedural Fabric Grain / Noise
inline float noise(glm::vec2 t) {
    glm::vec2 i = glm::floor(t);
    glm::vec2 f = glm::fract(t);
    f = f * f * (3.0f - 2.0f * f);
    float a = hash(i);
    float b = hash(i + glm::vec2(1.0f, 0.0f));
    float c = hash(i + glm::vec2(0.0f, 1.0f));
    float d = hash(i + glm::vec2(1.0f, 1.0f));
    return glm::mix(glm::mix(a, b, f.x), glm::mix(c, d, f.x), f.y);
}

// Basic SDF Primitives
inline float sd_circle(glm::vec2 p, float r) { 
    return glm::length(p) - r; 
}

inline float sd_box(glm::vec2 p, glm::vec2 b) {
    glm::vec2 d = glm::abs(p) - b;
    return glm::length(glm::max(d, glm::vec2(0.0f))) + glm::min(glm::max(d.x, d.y), 0.0f);
}

inline float sd_triangle(glm::vec2 p, glm::vec2 p0, glm::vec2 p1, glm::vec2 p2) {
    glm::vec2 e0 = p1 - p0, e1 = p2 - p1, e2 = p0 - p2;
    glm::vec2 v0 = p  - p0, v1 = p  - p1, v2 = p  - p2;
    glm::vec2 pq0 = v0 - e0 * glm::clamp(glm::dot(v0, e0) / glm::dot(e0, e0), 0.0f, 1.0f);
    glm::vec2 pq1 = v1 - e1 * glm::clamp(glm::dot(v1, e1) / glm::dot(e1, e1), 0.0f, 1.0f);
    glm::vec2 pq2 = v2 - e2 * glm::clamp(glm::dot(v2, e2) / glm::dot(e2, e2), 0.0f, 1.0f);
    float s = glm::sign(e0.x * e2.y - e0.y * e2.x);
    glm::vec2 d = glm::min(glm::min(
        glm::vec2(glm::dot(pq0, pq0), s * (v0.x * e0.y - v0.y * e0.x)),
        glm::vec2(glm::dot(pq1, pq1), s * (v1.x * e1.y - v1.y * e1.x))),
        glm::vec2(glm::dot(pq2, pq2), s * (v2.x * e2.y - v2.y * e2.x)));
    return -glm::sqrt(d.x) * glm::sign(d.y);
}

// 72-Point Continuous Spline Flame (Гал)
inline float sd_flame(glm::vec2 p) {
    constexpr int N = 72;
    static const std::array<glm::vec2, N> v = {
        glm::vec2( 4.30f , 225.00f), glm::vec2( 2.80f , 222.50f), glm::vec2( 0.60f , 219.20f), glm::vec2(-1.80f , 216.00f),
        glm::vec2(-3.10f , 212.80f), glm::vec2(-3.40f , 208.50f), glm::vec2(-3.00f , 203.50f), glm::vec2(-3.80f , 198.00f),
        glm::vec2(-5.50f , 192.50f), glm::vec2(-7.80f , 187.80f), glm::vec2(-8.60f , 182.50f), glm::vec2(-7.20f , 178.00f),
        glm::vec2(-5.20f , 175.20f), glm::vec2(-6.80f , 171.50f), glm::vec2(-9.80f , 169.80f), glm::vec2(-12.80f, 170.60f),
        glm::vec2(-14.90f, 173.20f), glm::vec2(-15.20f, 176.50f), glm::vec2(-14.10f, 179.80f), glm::vec2(-12.40f, 182.50f),
        glm::vec2(-10.80f, 186.20f), glm::vec2(-10.40f, 191.00f), glm::vec2(-11.20f, 196.00f), glm::vec2(-10.80f, 199.50f),
        glm::vec2(-13.80f, 196.50f), glm::vec2(-16.50f, 191.50f), glm::vec2(-18.50f, 186.50f), glm::vec2(-19.80f, 181.50f),
        glm::vec2(-21.80f, 177.50f), glm::vec2(-24.20f, 172.50f), glm::vec2(-25.00f, 168.50f), glm::vec2(-24.50f, 164.00f),
        glm::vec2(-23.00f, 159.50f), glm::vec2(-20.50f, 155.50f), glm::vec2(-17.00f, 151.80f), glm::vec2(-12.50f, 148.80f),
        glm::vec2(-7.00f , 146.40f), glm::vec2( 0.00f , 145.00f), glm::vec2( 7.00f , 146.40f), glm::vec2( 12.50f, 148.80f),
        glm::vec2( 17.00f, 151.80f), glm::vec2( 20.50f, 155.50f), glm::vec2( 23.00f, 159.50f), glm::vec2( 24.50f, 164.00f),
        glm::vec2( 25.00f, 168.50f), glm::vec2( 24.20f, 172.50f), glm::vec2( 21.80f, 177.50f), glm::vec2( 19.80f, 181.50f),
        glm::vec2( 18.50f, 186.50f), glm::vec2( 16.50f, 191.50f), glm::vec2( 13.80f, 196.50f), glm::vec2( 10.80f, 199.50f),
        glm::vec2( 11.20f, 196.00f), glm::vec2( 10.40f, 191.00f), glm::vec2( 10.80f, 186.20f), glm::vec2( 12.40f, 182.50f),
        glm::vec2( 14.10f, 179.80f), glm::vec2( 15.20f, 176.50f), glm::vec2( 14.90f, 173.20f), glm::vec2( 12.80f, 170.60f),
        glm::vec2(  9.80f, 169.80f), glm::vec2(  6.80f, 171.50f), glm::vec2(  5.20f, 175.20f), glm::vec2(  7.20f, 178.00f),
        glm::vec2(  8.60f, 182.50f), glm::vec2(  8.80f, 188.00f), glm::vec2(  7.50f, 193.50f), glm::vec2(  5.20f, 199.50f),
        glm::vec2(  3.00f, 205.00f), glm::vec2(  1.50f, 211.00f), glm::vec2(  1.60f, 216.50f), glm::vec2(  3.20f, 221.50f)
    };

    float d = glm::dot(p - v[0], p - v[0]);
    float s = 1.0f;
    for (int i = 0, j = N - 1; i < N; j = i++) {
        glm::vec2 e = v[j] - v[i];
        glm::vec2 w = p - v[i];
        glm::vec2 b = w - e * glm::clamp(glm::dot(w, e) / glm::dot(e, e), 0.0f, 1.0f);
        d = std::min(d, glm::dot(b, b));
        bool c0 = (p.y >= v[i].y);
        bool c1 = (p.y < v[j].y);
        bool c2 = (e.x * w.y > e.y * w.x);
        if ((c0 && c1 && c2) || (!c0 && !c1 && !c2)) {
            s = -s;
        }
    }
    return s * std::sqrt(d);
}

// Map Soyombo Scene
inline float map_soyombo(glm::vec2 p) {
    float d_pillars = sd_box(glm::vec2(std::abs(p.x) - 85.0f, p.y + 105.0f), glm::vec2(25.0f, 120.0f));
    float d_bar_top = sd_box(p - glm::vec2(0.0f,  -35.0f), glm::vec2(50.0f, 10.0f));
    float d_bar_bot = sd_box(p - glm::vec2(0.0f, -175.0f), glm::vec2(50.0f, 10.0f));
    float d_tri_top = sd_triangle(p, glm::vec2(0.0f,  -15.0f), glm::vec2(50.0f,   15.0f), glm::vec2(-50.0f,   15.0f));
    float d_tri_bot = sd_triangle(p, glm::vec2(0.0f, -225.0f), glm::vec2(50.0f, -195.0f), glm::vec2(-50.0f, -195.0f));

    glm::vec2 tp    = p - glm::vec2(0.0f, -105.0f);
    float d_c       = sd_circle(tp, 50.0f);
    glm::vec2 p1    = tp - glm::vec2(0.0f,  26.5f);
    glm::vec2 p2    = tp - glm::vec2(0.0f, -26.5f);
    float d_arc1    = (p1.x >= 0.0f) ? std::abs(glm::length(p1) - 26.5f) : std::min(glm::length(p1 - glm::vec2(0.0f, 26.5f)), glm::length(p1 - glm::vec2(0.0f, -26.5f)));
    float d_arc2    = (p2.x <= 0.0f) ? std::abs(glm::length(p2) - 26.5f) : std::min(glm::length(p2 - glm::vec2(0.0f, 26.5f)), glm::length(p2 - glm::vec2(0.0f, -26.5f)));
    float d_fish    = std::max(d_c, -(std::min(d_arc1, d_arc2) - 3.2f));
    float d_taijitu = std::max(d_fish, -sd_circle(glm::vec2(tp.x, std::abs(tp.y) - 26.5f), 10.0f));

    float d_moon    = std::max(sd_circle(p - glm::vec2(0.0f,  80.0f), 55.0f), -sd_circle(p - glm::vec2(0.0f, 105.0f), 60.0f));
    float d_sun     = sd_circle(p - glm::vec2(0.0f,  95.0f), 40.0f);
    float d_fire    = sd_flame(p);

    float d_core    = std::min(d_pillars, std::min(d_bar_top, std::min(d_bar_bot, std::min(d_tri_top, std::min(d_tri_bot, d_taijitu)))));
    return std::min(d_core, std::min(d_moon, std::min(d_sun, d_fire)));
}

// 3-Stripe Background
inline glm::vec3 paint_background(glm::vec2 uv) {
    float part_width = 1.0f / 3.0f;
    if (uv.x < part_width || uv.x > part_width * 2.0f) {
        return MONGOLIAN_RED;
    }
    return MONGOLIAN_BLUE;
}

// Main Fragment Shader
shs::Color fragment_shader(glm::vec2 u_uv, float u_time)
{
    glm::vec2 resolution(float(CANVAS_WIDTH), float(CANVAS_HEIGHT));

    glm::vec2 uv = u_uv / resolution;
    glm::vec2 st = u_uv / resolution;

    // 1. Намуухан сүрлэг даллагдах хурд (Speed = 1.65)
    float w = std::sin((uv.x + uv.y - u_time * 1.65f + std::sin(15.5f * uv.x + 4.5f * uv.y) * PI * 0.1f) * PI * 0.6f);
    uv *= 1.0f + (0.026f - 0.026f * w);

    // 2. Арын дэвсгэр өнгө
    glm::vec3 out_color = paint_background(uv);

    // 3. Зүүн талын баганад Соёмбыг голлуулах (+Y дээшээ)
    glm::vec2 p;
    p.x = (uv.x - 1.0f / 6.0f) * (resolution.x / resolution.y);
    p.y = uv.y - 0.5f;

    float view_scale = 650.0f;
    glm::vec2 q = p * view_scale;

    // 4. Anti-Aliased Соёмбо зураглал
    float d = map_soyombo(q);
    float aa = 1.5f * view_scale / resolution.y;
    float mask = glm::smoothstep(aa, -aa, d);
    out_color = glm::mix(out_color, MONGOLIAN_YELLOW, mask);

    // 5. Анхны хүчтэй, тод долгионы гэрэл сүүдэр
    out_color += w * 0.225f;

    // 6. Хөвөөний гүн Vignette
    float v = 16.0f * st.x * (1.0f - st.x) * st.y * (1.0f - st.y);
    out_color *= 1.0f - 0.6f * std::exp2(-1.75f * v);

    // 7. Dave Hoskins Hash Dithering
    out_color = glm::clamp(out_color - hash(u_uv) * 0.004f, 0.0f, 1.0f);

    // 8. Анхны баялаг даавууны Noise бүтэц
    out_color -= noise(u_uv) * 0.045f;

    // 9. Анхны Scanline / Micro-weave торлог
    float scan_y = (std::fmod(u_uv.y, 0.8f) < 1.0f) ? 0.01f : 0.0f;
    float scan_x = (std::fmod(u_uv.x, 0.8f) < 1.0f) ? 0.01f : 0.0f;
    out_color -= glm::vec3(scan_y + scan_x);

    out_color = glm::clamp(out_color, 0.0f, 1.0f);

    return shs::rgb01_to_color(out_color);
}

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

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

    Uint32 frame_delay            = 1000 / FRAMES_PER_SECOND; // 60 FPS
    float  frame_time_accumulator = 0.0f;
    int    frame_counter          = 0;
    float  time_accumulator       = 0.0f;

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
                if (event_data.key.keysym.sym == SDLK_ESCAPE) {
                    exit = true;
                }
                break;
            }
        }

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        // Multithreaded Tile Rendering
        std::vector<std::thread> thread_pool;
        int region_width  = CANVAS_WIDTH  / CONCURRENCY_COUNT;
        int region_height = CANVAS_HEIGHT / CONCURRENCY_COUNT;

        for (int i = 0; i < CONCURRENCY_COUNT; i++) {
            int start_x = i       * region_width;
            int end_x   = (i + 1) * region_width;

            for (int j = 0; j < CONCURRENCY_COUNT; j++) {
                int start_y = j       * region_height;
                int end_y   = (j + 1) * region_height;

                thread_pool.emplace_back([start_x, end_x, start_y, end_y, time_accumulator, main_canvas]() {
                    for (int x = start_x; x < end_x; x++) {
                        for (int y = start_y; y < end_y; y++) {
                            glm::vec2 uv = {float(x), float(y)};
                            shs::Color shader_output = fragment_shader(uv, time_accumulator);
                            shs::Canvas::draw_pixel(*main_canvas, x, y, shader_output);
                        }
                    }
                });
            }
        }

        for (auto &thread : thread_pool) {
            thread.join();
        }

        // SDL Surface рүү хуулж дэлгэцэнд зурах
        shs::Canvas::copy_to_SDLSurface(main_sdlsurface, main_canvas);
        SDL_UpdateTexture(screen_texture, NULL, main_sdlsurface->pixels, main_sdlsurface->pitch);
        SDL_Rect destination_rect{0, 0, WINDOW_WIDTH, WINDOW_HEIGHT};
        SDL_RenderCopy(renderer, screen_texture, NULL, &destination_rect);
        SDL_RenderPresent(renderer);

        frame_counter++;
        Uint32 delta_frame_time  = SDL_GetTicks() - frame_start_ticks;
        frame_time_accumulator  += delta_frame_time / 1000.0f;
        time_accumulator        += delta_frame_time / 1000.0f;

        if (delta_frame_time < frame_delay) {
            SDL_Delay(frame_delay - delta_frame_time);
        }
        if (frame_time_accumulator >= 1.0f) {
            std::string window_title = "Flag of Mongolia - FPS : " + std::to_string(frame_counter);
            frame_time_accumulator   = 0.0f;
            frame_counter            = 0;
            SDL_SetWindowTitle(window, window_title.c_str());
        }
    }

    delete main_canvas;
    main_canvas = nullptr;
    SDL_DestroyTexture(screen_texture);
    SDL_FreeSurface(main_sdlsurface);

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}