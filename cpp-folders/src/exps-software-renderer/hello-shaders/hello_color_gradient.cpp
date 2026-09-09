#include <SDL2/SDL.h>
#include <algorithm>
#include <string>
#include <iostream>
#include <array>
#include <cstdlib>
#include <tuple>
#include <thread>
#include <mutex>
#include <vector>
#include "shs_renderer.hpp"

#define FRAMES_PER_SECOND  60
#define WINDOW_WIDTH       640
#define WINDOW_HEIGHT      520
#define CANVAS_WIDTH       360
#define CANVAS_HEIGHT      240
#define CONCURRENCY_COUNT  8


/*
* Source :
* - https://thebookofshaders.com/06/
*/

float plot(std::array<double, 2> st, double pct)
{
    return shs::Math::smoothstep(float(pct - 0.01), float(pct), float(st[1])) -
           shs::Math::smoothstep(float(pct), float(pct + 0.01), float(st[1]));
}

shs::Color fragment_shader(std::array<double, 2> uniform_uv, double uniform_time)
{
    glm::vec2 st      = {float(uniform_uv[0]/CANVAS_WIDTH), float(uniform_uv[1]/CANVAS_HEIGHT)};
    glm::vec3 color_a = {0.149, 0.141, 0.912};
    glm::vec3 color_b = {1.000, 0.833, 0.224};
    glm::vec3 pct     = {st.x, st.x, st.x};
    glm::vec3 color   = shs::Math::mix(color_a, color_b, pct.x);
    color = shs::Math::mix(color, glm::vec3{1.0,0.0,0.0}, plot({st.x, st.y}, pct[0]));
    color = shs::Math::mix(color, glm::vec3{0.0,1.0,0.0}, plot({st.x, st.y}, pct[1]));
    color = shs::Math::mix(color, glm::vec3{0.0,0.0,1.0}, plot({st.x, st.y}, pct[2]));
    return shs::rgb01_to_color(color);
};


int main(int argc, char* argv[])
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
    double frame_time_accumulator = 0;
    int    frame_counter          = 0;
    int    fps                    = 0;
    double time_accumulator       = 0.0;

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
                            std::array<double, 2> uv = {float(x), float(y)};
                            shs::Color shader_output = fragment_shader(uv, time_accumulator);
                            {
                                //std::lock_guard<std::mutex> lock(canvas_mutex);
                                shs::Canvas::draw_pixel(*main_canvas, x, y, shader_output);
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

        // debug draw for if it rendering something
        shs::Canvas::fill_random_pixel(*main_canvas, 40, 30, 60, 80);


        // actually prensenting canvas data on hardware surface
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