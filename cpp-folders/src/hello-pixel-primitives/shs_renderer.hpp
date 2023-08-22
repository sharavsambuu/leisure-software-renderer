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


    class Color
    {
    public:
        Color()
        {
        }
        Color(shs::Pixel pixel)
        {
            this->pixel = pixel;
        }
        Color(std::uint8_t r, std::uint8_t g, std::uint8_t b)
        {
            this->pixel = shs::Pixel{r, g, b, 255};
        }
        Color(std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a)
        {
            this->pixel = shs::Pixel{r, g, b, a};
        }
        ~Color()
        {
        }

        void change_color(shs::Pixel pixel)
        {
            this->pixel.r = pixel.r;
            this->pixel.g = pixel.g;
            this->pixel.b = pixel.b;
            this->pixel.a = pixel.a;
        }
        void change_color(std::uint8_t r, std::uint8_t g, std::uint8_t b)
        {
            this->pixel.r = r;
            this->pixel.g = g;
            this->pixel.b = b;
        }
        void change_color(std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a)
        {
            this->pixel.r = r;
            this->pixel.g = g;
            this->pixel.b = b;
            this->pixel.a = a;
        }

        void change_red_channel(std::uint8_t r)
        {
            this->pixel.r = r;
        }
        void change_green_channel(std::uint8_t g)
        {
            this->pixel.g = g;
        }
        void change_blue_channel(std::uint8_t b)
        {
            this->pixel.b = b;
        }
        void change_alpha_channel(std::uint8_t a)
        {
            this->pixel.a = a;
        }

        std::uint8_t get_red_channel() 
        {
            return this->pixel.r;
        }
        std::uint8_t get_green_channel() 
        {
            return this->pixel.g;
        }
        std::uint8_t get_blue_channel() 
        {
            return this->pixel.b;
        }
        std::uint8_t get_alpha_channel() 
        {
            return this->pixel.a;
        }

        void set_pixel(shs::Pixel pixel)
        {
            this->pixel = pixel;
        }
        shs::Pixel get_pixel() 
        {
            return this->pixel;
        }

        static shs::Color red()
        {
            return shs::Color(shs::Pixel{255, 0, 0, 255});
        }
        static shs::Color green()
        {
            return shs::Color(shs::Pixel{0, 255, 0, 255});
        }
        static shs::Color blue()
        {
            return shs::Color(shs::Pixel{0, 0, 255, 255});
        }
        static shs::Color black()
        {
            return shs::Color(shs::Pixel{0, 0, 0, 255});
        }
        static shs::Color white()
        {
            return shs::Color(shs::Pixel{255, 255, 255, 255});
        }

    private:
        shs::Pixel pixel;
    };


    class Canvas
    {
    public:
        Canvas()
        {
        }
        Canvas(int width, int height)
        {
            this->width  = width;
            this->height = height;
            srand(static_cast<unsigned>(time(nullptr))); // seeding
            this->canvas.resize(this->width);
            for (int x = 0; x < this->width; ++x)
            {
                this->canvas[x].resize(this->height);
                for (int y = 0; y < this->height; ++y)
                {
                    this->canvas[x][y].r = rand() % 256;
                    this->canvas[x][y].g = rand() % 256;
                    this->canvas[x][y].b = rand() % 256;
                    this->canvas[x][y].a = 255;
                }
            }
        }
        Canvas(int width, int height, shs::Color color)
        {
            this->width  = width;
            this->height = height;
            this->canvas.resize(this->width);
            for (int x = 0; x < this->width; ++x)
            {
                this->canvas[x].resize(this->height);
                for (int y = 0; y < this->height; ++y)
                {
                    this->canvas[x][y].r = color.get_red_channel();
                    this->canvas[x][y].g = color.get_green_channel();
                    this->canvas[x][y].b = color.get_blue_channel();
                    this->canvas[x][y].a = color.get_alpha_channel();
                }
            }
        }
        Canvas(int width, int height, shs::Pixel pixel)
        {
            this->width  = width;
            this->height = height;
            this->canvas.resize(this->width);
            for (int x = 0; x < this->width; ++x)
            {
                this->canvas[x].resize(this->height);
                for (int y = 0; y < this->height; ++y)
                {
                    this->canvas[x][y].r = pixel.r;
                    this->canvas[x][y].g = pixel.g;
                    this->canvas[x][y].b = pixel.b;
                    this->canvas[x][y].a = pixel.a;
                }
            }
        }
        ~Canvas()
        {
        }

        int get_width() 
        {
            return this->width;
        }
        int get_height() 
        {
            return this->height;
        }

        void flip_vertically()
        {
            int half_height = this->height / 2;
            for (int row = 0; row < half_height; ++row)
            {
                std::swap(this->canvas[row], this->canvas[height - 1 - row]);
            }
        }
        static void flip_vertically(shs::Canvas & canvas)
        {
            canvas.flip_vertically();
        }
        void flip_horizontally()
        {
            for (int row = 0; row < this->height; ++row)
            {
                std::reverse(this->canvas[row].begin(), this->canvas[row].end());
            }
        }
        static void flip_horizontally(shs::Canvas & canvas)
        {
            canvas.flip_horizontally();
        }

        void draw_pixel(int x, int y, shs::Pixel pixel)
        {
            int location_x = x;
            int location_y = y;
            if (x < 0)
            {
                location_x = 0;
            }
            if (y < 0)
            {
                location_y = 0;
            }
            if (x > this->width)
            {
                location_x = this->width;
            }
            if (y > this->height)
            {
                location_y = this->height;
            }
            this->canvas[location_x][location_y].r = pixel.r;
            this->canvas[location_x][location_y].g = pixel.g;
            this->canvas[location_x][location_y].b = pixel.b;
            this->canvas[location_x][location_y].a = pixel.a;
        };
        void draw_pixel(int x, int y, shs::Color color)
        {
            this->draw_pixel(x, y, color.get_pixel());
        };
        static void draw_pixel(shs::Canvas & canvas, int x, int y, shs::Color color)
        {
            canvas.draw_pixel(x, y, color.get_pixel());
        };
        static void draw_pixel(shs::Canvas & canvas, int x, int y, shs::Pixel pixel)
        {
            canvas.draw_pixel(x, y, pixel);
        };

        static void draw_line_naive(shs::Canvas &canvas, int x0, int y0, int x1, int y1, shs::Color color)
        {
            float step = 0.01;
            for (float t = 0.0; t < 1.0; t += step)
            {
                int x = x0 + (x1 - x0) * t;
                int y = y0 + (y1 - y0) * t;
                shs::Canvas::draw_pixel(canvas, x, y, color);
            }
        }
        static void draw_line_second(shs::Canvas &canvas, int x0, int y0, int x1, int y1, shs::Color color)
        {
            for (int x = x0; x <= x1; x++)
            {
                float t = (x - x0) / (float)(x1 - x0);
                int y = y0 * (1. - t) + y1 * t;
                shs::Canvas::draw_pixel(canvas, x, y, color);
            }
        }
        static void draw_line(shs::Canvas &canvas, int x0, int y0, int x1, int y1, shs::Color color)
        {
            bool steep = false;
            if (std::abs(x0 - x1) < std::abs(y0 - y1))
            { // if the line is steep, we transpose the image
                std::swap(x0, y0);
                std::swap(x1, y1);
                steep = true;
            }
            if (x0 > x1)
            { // make it left−to−right
                std::swap(x0, x1);
                std::swap(y0, y1);
            }
            for (int x = x0; x <= x1; x++)
            {
                float t = (x - x0) / (float)(x1 - x0);
                int y = y0 * (1. - t) + y1 * t;
                if (steep)
                {
                    // if transposed, de−transpose
                    shs::Canvas::draw_pixel(canvas, y, x, color);
                }
                else
                {
                    shs::Canvas::draw_pixel(canvas, x, y, color);
                }
            }
        }

        void SDL_SetPixel(SDL_Surface *surface, int x, int y, Uint32 color)
        {
            Uint32 *pixel = (Uint32 *)((Uint8 *)surface->pixels + y * surface->pitch + x * sizeof(Uint32));
            *pixel = color;
        }
        void save_png(const std::string &filename)
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

            SDL_Surface *surface = SDL_CreateRGBSurface(0, this->width, this->height, 32, 0, 0, 0, 0);

            if (!surface)
            {
                std::cerr << "SDL_CreateRGBSurface Error: " << SDL_GetError() << std::endl;
                IMG_Quit();
                SDL_Quit();
                return;
            }

            for (int x = 0; x < this->width; ++x)
            {
                for (int y = 0; y < this->height; ++y)
                {
                    shs::Pixel pixel = canvas[x][y];
                    Uint32 color = SDL_MapRGBA(surface->format, pixel.r, pixel.g, pixel.b, pixel.a);
                    this->SDL_SetPixel(surface, x, y, color);
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

    private:
        std::vector<std::vector<shs::Pixel>> canvas;
        int width;
        int height;
    };


}