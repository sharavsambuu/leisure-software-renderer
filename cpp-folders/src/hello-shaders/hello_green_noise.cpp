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
#define WINDOW_WIDTH       900
#define WINDOW_HEIGHT      600
#define CANVAS_WIDTH       900
#define CANVAS_HEIGHT      600


/*
*
* Modified from :
*   Matrix by patriciogv - https://www.shadertoy.com/view/MlfXzN
*
*/

glm::vec4 rescale_vec4_1_255(const glm::vec4 &input_vec) {
    glm::vec4 clamped_value = glm::clamp(input_vec, 0.0f, 1.0f);
    glm::vec4 scaled_value  = clamped_value * 255.0f;
    return scaled_value;
}

glm::vec4 fragment_shader(glm::vec2 u_uv, float u_time)
{
    glm::vec2 i = u_uv;
    i *= 1.1f;
    glm::vec2 j = glm::fract(i);
    glm::vec2 p = glm::vec2(140.0f, static_cast<int>(u_time* (10.0f + 18.0f * glm::sin(i.x - j.x)))) + i;

    glm::vec4 o(0.0f); 

    o.g = glm::fract(40.0f * glm::sin(glm::dot(p, p)));

    p *= j;
    o *= (glm::fract(40.0f * glm::sin(glm::dot(p, p))) > 0.25f && j.x < 0.75f && j.y < 0.75f) ? 1.0f : 0.0f;

    return rescale_vec4_1_255(o);
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


        for (int x = 0; x < CANVAS_WIDTH; x++)
        {
            for (int y = 0; y < CANVAS_HEIGHT; y++)
            {
                glm::vec2 uv = {float(x), float(y)};
                glm::vec4 shader_output = fragment_shader(uv, time_accumulator);
                shs::Canvas::draw_pixel(*main_canvas, x, y, shs::Color{u_int8_t(shader_output[0]), u_int8_t(shader_output[1]), u_int8_t(shader_output[2]), u_int8_t(shader_output[3])});
            }
        }

        // debug draw for if it rendering something
        shs::Canvas::fill_random_pixel(*main_canvas, 40, 30, 60, 80);


        // actually prensenting canvas data on hardware surface
        shs::Canvas::flip_vertically(*main_canvas);
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