#include <SDL2/SDL.h>
#include "shs_renderer.hpp"

#define WINDOW_WIDTH  640
#define WINDOW_HEIGHT 480
#define CANVAS_WIDTH  256
#define CANVAS_HEIGHT 256



int main()
{

    SDL_Window   *window   = nullptr;
    SDL_Renderer *renderer = nullptr;

    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, 0, &window, &renderer);
    SDL_RenderSetScale(renderer, CANVAS_WIDTH/WINDOW_WIDTH, CANVAS_HEIGHT/WINDOW_HEIGHT);

    shs::Canvas *main_canvas     = new shs::Canvas(CANVAS_WIDTH+1, CANVAS_HEIGHT+1);
    SDL_Surface *main_sdlsurface = main_canvas->create_sdl_surface();
    SDL_Texture* screen_texture  = SDL_CreateTextureFromSurface(renderer, main_sdlsurface);


    bool exit = false;
    SDL_Event eventData;
    while (!exit)
    {
        while (SDL_PollEvent(&eventData))
        {
            switch (eventData.type)
            {
            case SDL_QUIT:
                exit = true;
                break;
            }
        }


        // preparing to render on SDL2
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        // SHS rendering
        //shs::Canvas::fill_pixel(*main_canvas, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, shs::Pixel::blue_pixel());
        shs::Canvas::fill_pixel(*main_canvas, 10, 10, 20, 30, shs::Pixel::white_pixel());
        shs::Canvas::fill_random_pixel(*main_canvas, 40, 30, 60, 80);
        shs::Canvas::copy_to_SDLSurface(main_sdlsurface, main_canvas);

        // actually prensenting on hardware surface
        SDL_Rect destination_rect{0, 0, WINDOW_WIDTH, WINDOW_HEIGHT};
        SDL_RenderCopy(renderer, screen_texture, NULL, &destination_rect);
        SDL_RenderPresent(renderer);
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