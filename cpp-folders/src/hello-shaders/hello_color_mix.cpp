#include <SDL2/SDL.h>
#include <algorithm>
#include <string>
#include <iostream>
#include <array>
#include <cstdlib>
#include <tuple>
#include "shs_renderer.hpp"

#define FRAMES_PER_SECOND 60
#define WINDOW_WIDTH      640
#define WINDOW_HEIGHT     360

#define CANVAS_WIDTH      340
#define CANVAS_HEIGHT     260


/*
* Source :
* - https://thebookofshaders.com/06/
*/

std::array<double, 4> rescale_vec4_1_255(const std::array<double, 4> &input_arr)
{
    std::array<double, 4> output_arr;
    for (size_t i = 0; i < input_arr.size(); ++i)
    {
        double clamped_value = std::max(0.0, std::min(1.0, input_arr[i]));
        double scaled_value = clamped_value * 255.0;
        output_arr[i] = scaled_value;
    }
    return output_arr;
}

std::array<double, 3> mix_vec3(const std::array<double, 3> &array1, const std::array<double, 3> &array2, double factor)
{
    std::array<double, 3> result;
    for (size_t i = 0; i < result.size(); ++i)
    {
        result[i] = (1.0 - factor) * array1[i] + factor * array2[i];
    }
    return result;
}

std::array<double, 4> fragment_shader(std::array<double, 2> uniform_uv, double uniform_time)
{
    std::array<double, 3> color_a = {0.149, 0.141, 0.912};
    std::array<double, 3> color_b = {1.000, 0.833, 0.224};
    double pct = std::abs(std::sin(uniform_time));
    std::array<double, 3> mixed_color = mix_vec3(color_a, color_b, pct);
    std::array<double, 4> output_arr = {mixed_color[0], mixed_color[1], mixed_color[2], 1.0};
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

        // software rendering or drawing stuffs goes around here
        shs::Canvas::fill_pixel(*main_canvas, 0, 0, CANVAS_WIDTH, CANVAS_HEIGHT, shs::Pixel::blue_pixel());

        // Run fragment shader
        for (int x=0; x<CANVAS_WIDTH; x++)
        {
            for (int y=0; y<CANVAS_HEIGHT; y++)
            {
                // preparing shader input
                std::array<double, 2> uv = {float(x), float(y)};
                std::array<double, 4> shader_output = fragment_shader(uv, time_accumulator);
                shs::Canvas::draw_pixel(*main_canvas, x, y, shs::Color{shader_output[0], shader_output[1], shader_output[2], shader_output[3]});
            }
        }

        // debug draw for if it rendering something
        shs::Canvas::fill_random_pixel(*main_canvas, 40, 30, 60, 80);


        // actually prensenting canvas data on hardware surface
        shs::Canvas::flip_horizontally(*main_canvas); // origin at the left bottom corner of the canvas
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