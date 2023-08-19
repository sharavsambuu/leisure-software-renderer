#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cstdlib>


struct Pixel {
    std::uint8_t r;
    std::uint8_t g;
    std::uint8_t b;
    std::uint8_t a;
};


void generate_random_image(std::vector<std::vector<Pixel>>& image, int width, int height) {
    srand(static_cast<unsigned>(time(nullptr))); // seeding
    image.resize(width);
    for (int x = 0; x < width; ++x) {
        image[x].resize(height);
        for (int y = 0; y < height; ++y) {
            image[x][y].r = rand() % 256;
            image[x][y].g = rand() % 256;
            image[x][y].b = rand() % 256;
            image[x][y].a = 255;
        }
    }
};

void fill_image_with_color(std::vector<std::vector<Pixel>>& image, Pixel&pixel, int width, int height) {
    image.resize(width);
    for (int x = 0; x < width; ++x) {
        image[x].resize(height);
        for (int y = 0; y < height; ++y) {
            image[x][y].r = pixel.r;
            image[x][y].g = pixel.g;
            image[x][y].b = pixel.b;
            image[x][y].a = pixel.a;
        }
    }
};

void set_color_to_image(std::vector<std::vector<Pixel>>& image, int x, int y, Pixel& pixel) {
    image[x][y].r = pixel.r;
    image[x][y].g = pixel.g;
    image[x][y].b = pixel.b;
    image[x][y].a = pixel.a;
};

void flip_vertically(std::vector<std::vector<Pixel>>& image) {
    int height      = image.size();
    int half_height = height / 2;

    for (int row = 0; row < half_height; ++row) {
        std::swap(image[row], image[height - 1 - row]);
    }
}

void flip_horizontally(std::vector<std::vector<Pixel>>& image) {
    int width  = image.size();
    int height = image[0].size();

    for (int row = 0; row < height; ++row) {
        std::reverse(image[row].begin(), image[row].end());
    }
}

void SDL_SetPixel(SDL_Surface* surface, int x, int y, Uint32 color) {
    Uint32* pixel = (Uint32*)((Uint8*)surface->pixels + y * surface->pitch + x * sizeof(Uint32));
    *pixel = color;
}

void save_to_png(const std::string& filename, std::vector<std::vector<Pixel>>& image, int width, int height) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL_Init Error: " << SDL_GetError() << std::endl;
        return;
    }
    int imgFlags = IMG_INIT_PNG;
    if ((IMG_Init(imgFlags) & imgFlags) != imgFlags) {
        std::cerr << "IMG_Init Error: " << IMG_GetError() << std::endl;
        SDL_Quit();
        return;
    }

     SDL_Surface* imageSurface = SDL_CreateRGBSurface(0, width, height, 32, 0, 0, 0, 0);

    if (!imageSurface) {
        std::cerr << "SDL_CreateRGBSurface Error: " << SDL_GetError() << std::endl;
        IMG_Quit();
        SDL_Quit();
        return;
    }

    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            Pixel pixel = image[x][y];
            Uint32 color = SDL_MapRGBA(imageSurface->format, pixel.r, pixel.g, pixel.b, pixel.a);
            SDL_SetPixel(imageSurface, x, y, color);
        }
    }

    if (IMG_SavePNG(imageSurface, filename.c_str()) < 0) {
        std::cerr << "IMG_SavePNG Error: " << IMG_GetError() << std::endl;
    } else {
        std::cout << "Image saved successfully." << std::endl;
    }

    SDL_FreeSurface(imageSurface);
    IMG_Quit();
    SDL_Quit();
}

int main() {

    std::cout<<"Hello Pixel"<<std::endl;

    int image_width  = 100;
    int image_height = 100;

    Pixel color_white = {255, 255, 255, 255};
    Pixel color_red   = {255,   0,   0, 255};
    Pixel color_black = {  0,   0,   0, 255};

    std::vector<std::vector<Pixel>> random_image;
    std::vector<std::vector<Pixel>> white_image;
    std::vector<std::vector<Pixel>> red_image;
    std::vector<std::vector<Pixel>> canvas_image;

    generate_random_image(random_image, image_width, image_height             );
    fill_image_with_color(white_image , color_white, image_width, image_height);
    fill_image_with_color(red_image   , color_red  , image_width, image_height);
    fill_image_with_color(canvas_image, color_black, image_width, image_height);

    set_color_to_image(canvas_image, 10, 10, color_red);
    set_color_to_image(canvas_image, 20, 20, color_red);
    set_color_to_image(canvas_image, 30, 30, color_red);
    set_color_to_image(canvas_image,  5, 60, color_white);
    flip_horizontally(canvas_image); // origin at the left bottom corner of the image

    save_to_png("random_image.png", random_image, image_width, image_height);
    save_to_png("white_image.png" , white_image , image_width, image_height);
    save_to_png("red_image.png"   , red_image   , image_width, image_height);
    save_to_png("canvas_image.png", canvas_image, image_width, image_height);


    return 0;
}