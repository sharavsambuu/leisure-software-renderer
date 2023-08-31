#include <SDL2/SDL.h>
#include <algorithm>
#include <string>
#include <cstdlib>
#include <tuple>
#include "shs_renderer.hpp"

#define FRAMES_PER_SECOND 60
#define WINDOW_WIDTH      640
#define WINDOW_HEIGHT     360

#define CANVAS_WIDTH      340
#define CANVAS_HEIGHT     260


/*
* Converting some shaders from shadertoy.com
* - https://www.shadertoy.com/view/DtXfDr
*/

float shader_smooth_step(float edge0, float edge1, float x) {
    x = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x);
}

std::array<float, 4> shader_line(std::array<float, 2> uv, float speed, float height, std::array<float, 3> col, float i_time)
{
    float uv_x = uv[0];
    float uv_y = uv[1];
    uv_y += shader_smooth_step(1.0, 0.0, std::abs(uv_x))*std::sin(i_time*speed+uv_x*height)*0.2;
    std::array<float, 4> output_arr = {0.0, 0.0, 0.0, 1.0};
    output_arr[0] = (0.06*shader_smooth_step(0.2, 0.9, std::abs(uv_x)), 0.0, std::abs(uv_y)-0.004)*col[0] * shader_smooth_step(1.0, 0.3, std::abs(uv_x));
    output_arr[1] = (0.06*shader_smooth_step(0.2, 0.9, std::abs(uv_x)), 0.0, std::abs(uv_y)-0.004)*col[1] * shader_smooth_step(1.0, 0.3, std::abs(uv_x));
    output_arr[2] = (0.06*shader_smooth_step(0.2, 0.9, std::abs(uv_x)), 0.0, std::abs(uv_y)-0.004)*col[2] * shader_smooth_step(1.0, 0.3, std::abs(uv_x));
    return output_arr;
}

std::array<float, 4> fragment_shader(std::array<float, 2> i_uv, float i_time)
{
    std::array<float, 2> uv = {
        float((i_uv[0] - 5.0*CANVAS_WIDTH)/CANVAS_HEIGHT),
        float((i_uv[1] - 5.0*CANVAS_WIDTH)/CANVAS_HEIGHT)
    };
    std::array<float, 4> output_arr = {0.0, 0.0, 0.0, 1.0};
    for (float i = 0.0; i <= 5.0; i += 1.0)
    {
        float t = i/5.0;
        std::array<float, 3> col = {0.2+t*0.7, 0.2+t*0.4, 0.3};
        std::array<float, 4> line_color = shader_line(uv, 1.0+t, 4.0+t, col, i_time);
        output_arr[0] += line_color[0];
        output_arr[1] += line_color[1];
        output_arr[2] += line_color[2];
    }
    return output_arr;
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

    int   frame_delay            = 1000 / FRAMES_PER_SECOND; // Delay for 60 FPS
    float frame_time_accumulator = 0;
    int   frame_counter          = 0;
    int   fps                    = 0;
    float time_accumulator       = 0.0;

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
                std::array<float, 2> uv = {float(x), float(y)};
                std::array<float, 4> shader_output = fragment_shader(uv, time_accumulator);
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