#include <SDL2/SDL.h>

int main() {

    int width  = 480;
    int height = 360;

    SDL_Window*   window   = nullptr;
    SDL_Renderer* renderer = nullptr;

    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer(width*2, height*2, 0, &window, &renderer);
    SDL_RenderSetScale(renderer, 2, 2);

    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);

    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_RenderDrawPoint(renderer, width/2, height/2);

    SDL_RenderPresent(renderer);

    SDL_Delay(10000);


    return 0;
}