/*
    THREADED SHADER RENDERER EXAMPLE
    
    Дэлгэцийг хэд хэдэн бүсэд (Region) хувааж, 
    бүс тус бүрийг тусдаа Thread-ээр зэрэгцүүлэн тооцоолох.

     shs::Job::WaitGroup instead of global atomic+cv.
*/

#include <SDL2/SDL.h>
#include <glm/glm.hpp>
#include <glm/gtc/noise.hpp> 
#include <algorithm>
#include <string>
#include <vector>
#include <iostream>
#include <cmath>

#include "shs_renderer.hpp" 

#define FRAMES_PER_SECOND  60
#define WINDOW_WIDTH       640
#define WINDOW_HEIGHT      520
#define CANVAS_WIDTH       360
#define CANVAS_HEIGHT      240
#define CONCURRENCY_COUNT  20
#define NUM_OCTAVES        5

glm::vec4 rescale_vec4_1_255(const glm::vec4 &input_vec) {
    return glm::clamp(input_vec, 0.0f, 1.0f) * 255.0f;
}

float fbm(const glm::vec2& st) {
    glm::vec2 _st = st;
    float v = 0.0f;
    float a = 0.5f;
    glm::vec2 shift(100.0f);
    glm::mat2 rot(cos(0.5f), sin(0.5f), -sin(0.5f), cos(0.5f));

    for (int i = 0; i < NUM_OCTAVES; ++i) {
        v += a * glm::simplex(_st); 
        _st = rot * _st * 2.0f + shift;
        a *= 0.5f;
    }
    return v;
}

glm::vec4 fragment_shader(glm::vec2 uniform_uv, float uniform_time)
{
    glm::vec2 st = (uniform_uv/glm::vec2(CANVAS_WIDTH, CANVAS_HEIGHT))*3.0f;
    st += float(glm::abs(glm::sin(uniform_time*0.1f)*3.0f))*st;
    
    glm::vec2 q(0.0);
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
    
    return rescale_vec4_1_255(glm::vec4(color*float(f*f*f+0.6f*f*f+0.5*f),1.0f));
};

int main(int argc, char* argv[])
{
    (void)argc; (void)argv;

    auto *job_system = new shs::Job::ThreadedPriorityJobSystem(CONCURRENCY_COUNT);
    shs::Job::WaitGroup wg;

    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *window;
    SDL_Renderer *renderer;
    SDL_CreateWindowAndRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, 0, &window, &renderer);
    SDL_RenderSetScale(renderer, 1, 1);

    shs::Canvas *main_canvas = new shs::Canvas(CANVAS_WIDTH, CANVAS_HEIGHT);
    SDL_Surface *main_sdlsurface = main_canvas->create_sdl_surface();
    SDL_Texture *screen_texture  = SDL_CreateTextureFromSurface(renderer, main_sdlsurface);

    bool exit_loop = false;
    SDL_Event event_data;

    int frame_delay = 1000 / FRAMES_PER_SECOND;
    float time_accumulator = 0.0f;
    int frame_counter = 0;
    float fps_timer = 0.0f;

    while (!exit_loop)
    {
        Uint32 frame_start_ticks = SDL_GetTicks();

        while (SDL_PollEvent(&event_data))
        {
            if (event_data.type == SDL_QUIT) exit_loop = true;
            else if (event_data.type == SDL_KEYDOWN) {
                if (event_data.key.keysym.sym == SDLK_ESCAPE) exit_loop = true;
            }
        }

        if (exit_loop) break;

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        int region_width  = (CANVAS_WIDTH  + CONCURRENCY_COUNT - 1) / CONCURRENCY_COUNT;
        int region_height = (CANVAS_HEIGHT + CONCURRENCY_COUNT - 1) / CONCURRENCY_COUNT;

        wg.reset();

        for (int i = 0; i < CONCURRENCY_COUNT; i++) {
            int start_x = i * region_width;
            int end_x   = std::min((i + 1) * region_width, CANVAS_WIDTH);

            for (int j = 0; j < CONCURRENCY_COUNT; j++) {
                int start_y = j * region_height;
                int end_y   = std::min((j + 1) * region_height, CANVAS_HEIGHT);

                wg.add(1);

                job_system->submit({[start_x, end_x, start_y, end_y, time_accumulator, main_canvas, &wg]() {

                    for (int x = start_x; x < end_x; x++) {
                        for (int y = start_y; y < end_y; y++) {

                            glm::vec2 uv = {float(x), float(y)};
                            glm::vec4 res = fragment_shader(uv, time_accumulator);

                            main_canvas->draw_pixel(x, y, shs::Color{
                                (uint8_t)res[0], (uint8_t)res[1], (uint8_t)res[2], 255
                            });
                        }
                    }

                    wg.done();

                }, shs::Job::PRIORITY_HIGH});
            }
        }

        wg.wait();

        shs::Canvas::fill_random_pixel(*main_canvas, 10, 10, 20, 20);

        shs::Canvas::copy_to_SDLSurface(main_sdlsurface, main_canvas);
        SDL_UpdateTexture(screen_texture, NULL, main_sdlsurface->pixels, main_sdlsurface->pitch);
        SDL_RenderCopy(renderer, screen_texture, NULL, NULL);
        SDL_RenderPresent(renderer);

        frame_counter++;
        Uint32 delta = SDL_GetTicks() - frame_start_ticks;
        float delta_s = delta / 1000.0f;
        time_accumulator += delta_s;
        fps_timer += delta_s;

        if (delta < (Uint32)frame_delay) SDL_Delay(frame_delay - delta);

        if (fps_timer >= 1.0f) {
            SDL_SetWindowTitle(window, ("FPS: " + std::to_string(frame_counter)).c_str());
            frame_counter = 0;
            fps_timer = 0.0f;
        }
    }

    delete job_system;
    delete main_canvas;

    SDL_DestroyTexture(screen_texture);
    SDL_FreeSurface(main_sdlsurface);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    std::cout << "Clean exit." << std::endl;
    return 0;
}
