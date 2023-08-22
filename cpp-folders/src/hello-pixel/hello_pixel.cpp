#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <algorithm>

namespace shs
{

    struct Pixel
    {
        std::uint8_t r;
        std::uint8_t g;
        std::uint8_t b;
        std::uint8_t a;
    };

    void canvas_generate_random(std::vector<std::vector<Pixel>> &canvas, int width, int height)
    {
        srand(static_cast<unsigned>(time(nullptr))); // seeding
        canvas.resize(width);
        for (int x = 0; x < width; ++x)
        {
            canvas[x].resize(height);
            for (int y = 0; y < height; ++y)
            {
                canvas[x][y].r = rand() % 256;
                canvas[x][y].g = rand() % 256;
                canvas[x][y].b = rand() % 256;
                canvas[x][y].a = 255;
            }
        }
    };

    void canvas_fill_color(std::vector<std::vector<Pixel>> &canvas, int width, int height, Pixel &pixel)
    {
        canvas.resize(width);
        for (int x = 0; x < width; ++x)
        {
            canvas[x].resize(height);
            for (int y = 0; y < height; ++y)
            {
                canvas[x][y].r = pixel.r;
                canvas[x][y].g = pixel.g;
                canvas[x][y].b = pixel.b;
                canvas[x][y].a = pixel.a;
            }
        }
    };

    void canvas_set_color(std::vector<std::vector<Pixel>> &canvas, int x, int y, Pixel &pixel)
    {
        canvas[x][y].r = pixel.r;
        canvas[x][y].g = pixel.g;
        canvas[x][y].b = pixel.b;
        canvas[x][y].a = pixel.a;
    };

    void canvas_flip_vertically(std::vector<std::vector<Pixel>> &canvas)
    {
        int height = canvas.size();
        int half_height = height / 2;

        for (int row = 0; row < half_height; ++row)
        {
            std::swap(canvas[row], canvas[height - 1 - row]);
        }
    }

    void canvas_flip_horizontally(std::vector<std::vector<Pixel>> &canvas)
    {
        int width = canvas.size();
        int height = canvas[0].size();

        for (int row = 0; row < height; ++row)
        {
            std::reverse(canvas[row].begin(), canvas[row].end());
        }
    }

    void SDL_SetPixel(SDL_Surface *surface, int x, int y, Uint32 color)
    {
        Uint32 *pixel = (Uint32 *)((Uint8 *)surface->pixels + y * surface->pitch + x * sizeof(Uint32));
        *pixel = color;
    }

    void canvas_save_png(const std::string &filename, std::vector<std::vector<Pixel>> &canvas, int width, int height)
    {
        if (SDL_Init(SDL_INIT_VIDEO) < 0)
        {
            std::cerr << "SDL_Init Error: " << SDL_GetError() << std::endl;
            return;
        }
        int imgFlags = IMG_INIT_PNG;
        if ((IMG_Init(imgFlags) & imgFlags) != imgFlags)
        {
            std::cerr << "IMG_Init Error: " << IMG_GetError() << std::endl;
            SDL_Quit();
            return;
        }

        SDL_Surface *surface = SDL_CreateRGBSurface(0, width, height, 32, 0, 0, 0, 0);

        if (!surface)
        {
            std::cerr << "SDL_CreateRGBSurface Error: " << SDL_GetError() << std::endl;
            IMG_Quit();
            SDL_Quit();
            return;
        }

        for (int x = 0; x < width; ++x)
        {
            for (int y = 0; y < height; ++y)
            {
                Pixel pixel = canvas[x][y];
                Uint32 color = SDL_MapRGBA(surface->format, pixel.r, pixel.g, pixel.b, pixel.a);
                SDL_SetPixel(surface, x, y, color);
            }
        }

        if (IMG_SavePNG(surface, filename.c_str()) < 0)
        {
            std::cerr << "IMG_SavePNG Error: " << IMG_GetError() << std::endl;
        }
        else
        {
            std::cout << "Image saved successfully." << std::endl;
        }

        SDL_FreeSurface(surface);
        IMG_Quit();
        SDL_Quit();
    }

}


int main() {

    std::cout<<"Hello Pixel"<<std::endl;

    int canvas_width  = 100;
    int canvas_height = 100;

    shs::Pixel color_white = {255, 255, 255, 255};
    shs::Pixel color_red   = {255,   0,   0, 255};
    shs::Pixel color_black = {  0,   0,   0, 255};

    std::vector<std::vector<shs::Pixel>> random_canvas;
    std::vector<std::vector<shs::Pixel>> white_canvas;
    std::vector<std::vector<shs::Pixel>> red_canvas;
    std::vector<std::vector<shs::Pixel>> canvas_canvas;

    shs::canvas_generate_random(random_canvas, canvas_width, canvas_height);
    shs::canvas_fill_color(white_canvas , canvas_width, canvas_height, color_white);
    shs::canvas_fill_color(red_canvas   , canvas_width, canvas_height, color_red  );
    shs::canvas_fill_color(canvas_canvas, canvas_width, canvas_height, color_black);

    shs::canvas_set_color(canvas_canvas, 10, 10, color_red);
    shs::canvas_set_color(canvas_canvas, 20, 20, color_red);
    shs::canvas_set_color(canvas_canvas, 30, 30, color_red);
    shs::canvas_set_color(canvas_canvas,  5, 60, color_white);
    shs::canvas_flip_horizontally(canvas_canvas); // origin at the left bottom corner of the canvas

    shs::canvas_save_png("random_canvas.png", random_canvas, canvas_width, canvas_height);
    shs::canvas_save_png("white_canvas.png" , white_canvas , canvas_width, canvas_height);
    shs::canvas_save_png("red_canvas.png"   , red_canvas   , canvas_width, canvas_height);
    shs::canvas_save_png("canvas_canvas.png", canvas_canvas, canvas_width, canvas_height);


    return 0;
}