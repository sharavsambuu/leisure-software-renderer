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
#define WINDOW_WIDTH       600
#define WINDOW_HEIGHT      600
#define CANVAS_WIDTH       320
#define CANVAS_HEIGHT      320
#define CONCURRENCY_COUNT  8
#define NUM_OCTAVES        5


/*
 * First things comes to my mind is like oh yes I have to use many thread as much as possible since
 * I have old i7 CPU with 8 cores. But in reality context switching on between threads is expensive thing
 * But in this example I'm using threads only for certain area of image region, and only 8 threads
 * are utilized, meaning fewer context switch on threads therefore more faster
 */

glm::vec4 rescale_vec4_1_255(const glm::vec4 &input_vec) {
    glm::vec4 clamped_value = glm::clamp(input_vec, 0.0f, 1.0f);
    glm::vec4 scaled_value  = clamped_value * 255.0f;
    return scaled_value;
}

// Function to generate a random value based on input vector using GLM
float random(const glm::vec2& _st) {
    return glm::fract(glm::sin(glm::dot(_st, glm::vec2(12.9898f, 78.233f))) * 43758.5453123f);
}

// Function to calculate noise using GLM
float noise(const glm::vec2& _st) {
    glm::vec2 i = glm::floor(_st);
    glm::vec2 f = glm::fract(_st);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + glm::vec2(1.0f, 0.0f));
    float c = random(i + glm::vec2(0.0f, 1.0f));
    float d = random(i + glm::vec2(1.0f, 1.0f));

    glm::vec2 u = f * f * (3.0f - 2.0f * f);

    return glm::mix(a, b, u.x) +
           (c - a) * u.y * (1.0f - u.x) +
           (d - b) * u.x * u.y;
}

float fbm(const glm::vec2& st) {
    glm::vec2 _st = st;
    float v = 0.0f;
    float a = 0.5f;
    glm::vec2 shift(100.0f);
    
    // Rotate to reduce axial bias
    glm::mat2 rot(cos(0.5f), sin(0.5f),
                  -sin(0.5f), cos(0.5f));

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
    glm::vec3 color(0.0);

    glm::vec2 q(0.0);

    q.x = fbm(st + 0.00f * uniform_time);
    q.y = fbm(st + glm::vec2(1.0f));

    glm::vec2 r(0.0f);
    r.x = fbm(st + 1.0f * q + glm::vec2(1.7f, 9.2f) + 0.15f * uniform_time);
    r.y = fbm(st + 1.0f * q + glm::vec2(8.3f, 2.8f) + 0.126f * uniform_time);

    float f = fbm(st + r);

    color = glm::mix(glm::vec3(0.101961f, 0.619608f, 0.666667f),
                     glm::vec3(0.666667f, 0.666667f, 0.498039f),
                     glm::clamp((f * f) * 4.0f, 0.0f, 1.0f));

    color = glm::mix(color,
                     glm::vec3(0.0f, 0.0f, 0.164706f),
                     glm::clamp(glm::length(q), 0.0f, 1.0f));

    color = glm::mix(color,
                     glm::vec3(0.666667f, 1.0f, 1.0f),
                     glm::clamp(glm::length(r.x), 0.0f, 1.0f));
    

    glm::vec4 output_arr = glm::vec4(color*float(f*f*f+0.6f*f*f+0.5*f),1.0f);
    return rescale_vec4_1_255(output_arr);
};


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

    int    frame_delay            = 1000 / FRAMES_PER_SECOND; // Delay for 60 FPS
    float  frame_time_accumulator = 0;
    int    frame_counter          = 0;
    int    fps                    = 0;
    float  time_accumulator       = 0.0;

    while (!exit)
    {

        Uint32 frame_start_ticks = SDL_GetTicks();

        // catching up input events happened on hardware
        while (SDL_PollEvent(&event_data))
        {
            switch (event_data.type)
            {
            case SDL_QUIT:
                exit = true;
                break;
            case SDL_KEYDOWN:
                switch(event_data.key.keysym.sym) {
                    case SDLK_ESCAPE: 
                        exit = true;
                        break;
                }
                break;
            }
        }


        // preparing to render on SDL2
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);


        // Run fragment shader with parallel threaded fashion
        std::mutex canvas_mutex;
        std::vector<std::thread> thread_pool;

        int region_width  = CANVAS_WIDTH  / CONCURRENCY_COUNT;
        int region_height = CANVAS_HEIGHT / CONCURRENCY_COUNT;

        for (int i = 0; i < CONCURRENCY_COUNT; i++) {
            int start_x = i       * region_width;
            int end_x   = (i + 1) * region_width;

            for (int j = 0; j < CONCURRENCY_COUNT; j++) {
                int start_y = j       * region_height;
                int end_y   = (j + 1) * region_height;

                std::thread task([start_x, end_x, start_y, end_y, time_accumulator, &main_canvas, &canvas_mutex]() {
                    for (int x = start_x; x < end_x; x++) {
                        for (int y = start_y; y < end_y; y++) {
                            glm::vec2 uv = {float(x), float(y)};
                            glm::vec4 shader_output = fragment_shader(uv, time_accumulator);
                            {
                                //std::lock_guard<std::mutex> lock(canvas_mutex);
                                shs::Canvas::draw_pixel(*main_canvas, x, y, shs::Color{u_int8_t(shader_output[0]), u_int8_t(shader_output[1]), u_int8_t(shader_output[2]), u_int8_t(shader_output[3])});
                            }
                        }
                    }
                });
                thread_pool.emplace_back(std::move(task));
            }
        }

        // waiting for other threads left in the pool finish its jobs
        for (auto &thread : thread_pool)
        {
            thread.join();
        }

        // debug draw for if it is rendering something
        shs::Canvas::fill_random_pixel(*main_canvas, 40, 30, 60, 80);


        // actually presenting canvas data on the hardware surface
        shs::Canvas::copy_to_SDLSurface(main_sdlsurface, main_canvas);
        SDL_UpdateTexture(screen_texture, NULL, main_sdlsurface->pixels, main_sdlsurface->pitch);
        SDL_Rect destination_rect{0, 0, WINDOW_WIDTH, WINDOW_HEIGHT};
        SDL_RenderCopy(renderer, screen_texture, NULL, &destination_rect);
        SDL_RenderPresent(renderer);

    
        frame_counter++;
        Uint32 delta_frame_time  = SDL_GetTicks() - frame_start_ticks;
        frame_time_accumulator  += delta_frame_time/1000.0;
        time_accumulator        += delta_frame_time/1000.0;
        if (delta_frame_time < frame_delay) {
            SDL_Delay(frame_delay - delta_frame_time);
        }
        if (frame_time_accumulator >= 1.0) {
            std::string window_title = "FPS : "+std::to_string(frame_counter);
            frame_time_accumulator   = 0.0;
            frame_counter            = 0;
            SDL_SetWindowTitle(window, window_title.c_str());
        }
    }


    // free the memory
    main_canvas  = nullptr; 
    delete main_canvas;
    SDL_DestroyTexture(screen_texture);
    SDL_FreeSurface(main_sdlsurface);

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}