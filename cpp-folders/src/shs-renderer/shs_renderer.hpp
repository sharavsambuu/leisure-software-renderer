#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/constants.hpp>


namespace shs
{

    struct Color
    {
        std::uint8_t r;
        std::uint8_t g;
        std::uint8_t b;
        std::uint8_t a;
    };

    class Pixel
    {
    public:
        Pixel()
        {
        }
        Pixel(shs::Color color)
        {
            this->color = color;
        }
        Pixel(std::uint8_t r, std::uint8_t g, std::uint8_t b)
        {
            this->color = shs::Color{r, g, b, 255};
        }
        Pixel(std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a)
        {
            this->color = shs::Color{r, g, b, a};
        }
        ~Pixel()
        {
        }

        void change_color(shs::Color color)
        {
            this->color.r = color.r;
            this->color.g = color.g;
            this->color.b = color.b;
            this->color.a = color.a;
        }
        void change_color(std::uint8_t r, std::uint8_t g, std::uint8_t b)
        {
            this->color.r = r;
            this->color.g = g;
            this->color.b = b;
        }
        void change_color(std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a)
        {
            this->color.r = r;
            this->color.g = g;
            this->color.b = b;
            this->color.a = a;
        }

        void change_red_channel(std::uint8_t r)
        {
            this->color.r = r;
        }
        void change_green_channel(std::uint8_t g)
        {
            this->color.g = g;
        }
        void change_blue_channel(std::uint8_t b)
        {
            this->color.b = b;
        }
        void change_alpha_channel(std::uint8_t a)
        {
            this->color.a = a;
        }

        std::uint8_t get_red_channel()
        {
            return this->color.r;
        }
        std::uint8_t get_green_channel()
        {
            return this->color.g;
        }
        std::uint8_t get_blue_channel()
        {
            return this->color.b;
        }
        std::uint8_t get_alpha_channel()
        {
            return this->color.a;
        }

        void set_color(shs::Color color)
        {
            this->color = color;
        }
        shs::Color get_color()
        {
            return this->color;
        }

        static shs::Pixel red_pixel()
        {
            return shs::Pixel{255, 0, 0, 255};
        }
        static shs::Pixel green_pixel()
        {
            return shs::Pixel{0, 255, 0, 255};
        }
        static shs::Pixel blue_pixel()
        {
            return shs::Pixel{0, 0, 255, 255};
        }
        static shs::Pixel black_pixel()
        {
            return shs::Pixel{0, 0, 0, 255};
        }
        static shs::Pixel white_pixel()
        {
            return shs::Pixel{255, 255, 255, 255};
        }
        static shs::Pixel random_pixel() 
        {
            return shs::Pixel{
                uint8_t(rand()%256), 
                uint8_t(rand()%256), 
                uint8_t(rand()%256), 
                uint8_t(255)
                };

        }

    private:
        shs::Color color;
    };

    class Canvas
    {
    public:
        Canvas()
        {
        }
        Canvas(int width, int height)
        {
            this->width = width;
            this->height = height;
            srand(static_cast<unsigned>(time(nullptr))); // seeding
            this->canvas.resize(this->width);
            for (int x = 0; x < this->width; x++)
            {
                this->canvas[x].resize(this->height);
                for (int y = 0; y < this->height; y++)
                {
                    this->canvas[x][y].r = rand() % 256;
                    this->canvas[x][y].g = rand() % 256;
                    this->canvas[x][y].b = rand() % 256;
                    this->canvas[x][y].a = 255;
                }
            }
        }
        Canvas(int width, int height, shs::Pixel pixel)
        {
            this->width = width;
            this->height = height;
            this->canvas.resize(this->width);
            for (int x = 0; x < this->width; x++)
            {
                this->canvas[x].resize(this->height);
                for (int y = 0; y < this->height; y++)
                {
                    this->canvas[x][y].r = pixel.get_red_channel();
                    this->canvas[x][y].g = pixel.get_green_channel();
                    this->canvas[x][y].b = pixel.get_blue_channel();
                    this->canvas[x][y].a = pixel.get_alpha_channel();
                }
            }
        }
        Canvas(int width, int height, shs::Color color)
        {
            this->width = width;
            this->height = height;
            this->canvas.resize(this->width);
            for (int x = 0; x < this->width; x++)
            {
                this->canvas[x].resize(this->height);
                for (int y = 0; y < this->height; y++)
                {
                    this->canvas[x][y].r = color.r;
                    this->canvas[x][y].g = color.g;
                    this->canvas[x][y].b = color.b;
                    this->canvas[x][y].a = color.a;
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

        void flip_horizontally()
        {
            int half_width = this->width / 2;
            for (int column = 0; column < half_width; column++)
            {
                std::swap(this->canvas[column], this->canvas[width - 1 - column]);
            }
        }
        static void flip_horizontally(shs::Canvas &canvas)
        {
            canvas.flip_horizontally();
        }
        void flip_vertically()
        {
            for (int row = 0; row < this->width; row++)
            {
                std::reverse(this->canvas[row].begin(), this->canvas[row].end());
            }
        }
        static void flip_vertically(shs::Canvas &canvas)
        {
            canvas.flip_vertically();
        }

        shs::Color get_color_at(int x, int y)
        {
            return this->canvas[x][y];
        }
        static shs::Color get_color_at(shs::Canvas &canvas, int x, int y)
        {
            return canvas.get_color_at(x, y);
        }
        shs::Pixel get_pixel_at(int x, int y)
        {
            return shs::Pixel(this->canvas[x][y]);
        }
        static shs::Pixel get_pixel_at(shs::Canvas &canvas, int x, int y)
        {
            return canvas.get_pixel_at(x, y);
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
            this->canvas[location_x][location_y].r = pixel.get_red_channel();
            this->canvas[location_x][location_y].g = pixel.get_green_channel();
            this->canvas[location_x][location_y].b = pixel.get_blue_channel();
            this->canvas[location_x][location_y].a = pixel.get_alpha_channel();
        };

        static void draw_pixel(shs::Canvas &canvas, int x, int y, shs::Color color)
        {
            canvas.draw_pixel(x, y, shs::Pixel(color));
        };
        static void draw_pixel(shs::Canvas &canvas, int x, int y, shs::Pixel pixel)
        {
            canvas.draw_pixel(x, y, pixel);
        };

        static void fill_pixel(shs::Canvas &canvas, int x0, int y0, int x1, int y1, shs::Pixel pixel)
        {
            for (int x = x0; x < x1; x++)
            {
                for (int y = y0; y < y1; y++)
                {
                    shs::Canvas::draw_pixel(canvas, x, y, pixel);
                }
            }
        }
        static void fill_random_pixel(shs::Canvas &canvas, int x0, int y0, int x1, int y1)
        {
            for (int x = x0; x < x1; x++)
            {
                for (int y = y0; y < y1; y++)
                {
                    shs::Canvas::draw_pixel(canvas, x, y, shs::Pixel::random_pixel());
                }
            }
        }

        static void draw_line(shs::Canvas &canvas, int x0, int y0, int x1, int y1, shs::Pixel pixel)
        {
            bool steep = false;
            if (std::abs(x0 - x1) < std::abs(y0 - y1))
            {
                std::swap(x0, y0);
                std::swap(x1, y1);
                steep = true;
            }
            if (x0 > x1)
            {
                std::swap(x0, x1);
                std::swap(y0, y1);
            }
            int dx = x1 - x0;
            int dy = y1 - y0;
            int derror2 = std::abs(dy) * 2;
            int error2 = 0;
            int y = y0;
            for (int x = x0; x <= x1; x++)
            {
                if (steep)
                {
                    shs::Canvas::draw_pixel(canvas, y, x, pixel);
                }
                else
                {
                    shs::Canvas::draw_pixel(canvas, x, y, pixel);
                }
                error2 += derror2;
                if (error2 > dx)
                {
                    y += (y1 > y0 ? 1 : -1);
                    error2 -= dx * 2;
                }
            }
        }

        inline static glm::vec3 barycentric_coordinate(const glm::vec2 &P, const std::vector<glm::vec2> &triangle_vertices)
        {
            if (triangle_vertices.size() != 3)
            {
                // Ensure there are exactly three vertices in the triangle
                throw std::invalid_argument("The triangle must have exactly three vertices.");
            }

            // Extract the vertices A, B, and C
            const glm::vec2 &A = triangle_vertices[0];
            const glm::vec2 &B = triangle_vertices[1];
            const glm::vec2 &C = triangle_vertices[2];

            // Calculate the area of the full triangle
            float areaABC = glm::cross(glm::vec3(B - A, 0), glm::vec3(C - A, 0)).z;

            // Calculate the barycentric coordinates
            glm::vec3 barycentric;

            barycentric.x = glm::cross(glm::vec3(B - P, 0), glm::vec3(C - P, 0)).z / areaABC;
            barycentric.y = glm::cross(glm::vec3(C - P, 0), glm::vec3(A - P, 0)).z / areaABC;
            barycentric.z = 1.0f - barycentric.x - barycentric.y;

            return barycentric;
        }

        static void draw_triangle(shs::Canvas &canvas, std::vector<glm::vec2> &in_vertices, shs::Pixel pixel)
        {
            int max_x = canvas.get_width ();
            int max_y = canvas.get_height();

            // converting from my coordinate system to popular rasterizer's coordinate system
            // my coordinate system is origin is bottom left corner, y axis moves to upward and x axis moves to the right
            // rasterizer's coordinate system is origin is top left corner, y axis moves to downward and x axis move to the right
            std::vector<glm::vec2> vertices;
            std::copy(in_vertices.begin(), in_vertices.end(), std::back_inserter(vertices));
            for (auto& point : vertices)
            {
                point.y = max_y-point.y;
            }


            glm::vec2 bboxmin(canvas.get_width() - 1, canvas.get_height() - 1);
            glm::vec2 bboxmax(0, 0);
            glm::vec2 clamp(canvas.get_width() - 1, canvas.get_height() - 1);

            for (int i = 0; i < 3; i++)
            {
                bboxmin = glm::max(glm::vec2(0), glm::min(bboxmin, vertices[i]));
                bboxmax = glm::min(clamp, glm::max(bboxmax, vertices[i]));
            }

            glm::vec2 p;
            for (p.x = bboxmin.x; p.x <= bboxmax.x; p.x++)
            {
                for (p.y = bboxmin.y; p.y <= bboxmax.y; p.y++)
                {
                    glm::vec3 bc_screen = shs::Canvas::barycentric_coordinate(p, vertices);

                    if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0)
                        continue;

                    // draw pixel in my coordinate system
                    glm::ivec2 p_my_coordinate_system = p;
                    p_my_coordinate_system.x = std::clamp<int>(        p_my_coordinate_system.x, 0, max_x);
                    p_my_coordinate_system.y = std::clamp<int>(max_y - p_my_coordinate_system.y, 0, max_y);
                    shs::Canvas::draw_pixel(canvas, p_my_coordinate_system.x, p_my_coordinate_system.y, pixel);
                }
            }
        }

        static void set_rawcolor_at_SDLSurface(SDL_Surface *surface, int x, int y, Uint32 raw_color)
        {
            Uint32 *pixel = (Uint32 *)((Uint8 *)surface->pixels + y * surface->pitch + x * sizeof(Uint32));
            *pixel = raw_color;
        }
        static void set_color_at_SDLSurface(SDL_Surface *surface, int x, int y, shs::Color color)
        {
            Uint32 raw_color = SDL_MapRGBA(surface->format, color.r, color.g, color.b, color.a);
            shs::Canvas::set_rawcolor_at_SDLSurface(surface, x, y, raw_color);
        }
        static void set_pixel_at_SDLSurface(SDL_Surface *surface, int x, int y, shs::Pixel pixel)
        {
            shs::Canvas::set_color_at_SDLSurface(surface, x, y, pixel.get_color());
        }
        static void copy_to_SDLSurface(SDL_Surface *surface, shs::Canvas* canvas)
        {
            for (int x = 0; x < canvas->get_width(); x++)
            {
                for (int y = 0; y < canvas->get_height(); y++)
                {
                    shs::Color color = canvas->get_color_at(x, y);
                    shs::Canvas::set_color_at_SDLSurface(surface, x, y, color);
                }
            }
        }

        SDL_Surface* create_sdl_surface() 
        {
            SDL_Surface *surface = SDL_CreateRGBSurface(0, this->width, this->height, 32, 0, 0, 0, 0);
            return surface;
        }
        static SDL_Surface* create_sdl_surface(int width, int height) 
        {
            SDL_Surface *surface = SDL_CreateRGBSurface(0, width, height, 32, 0, 0, 0, 0);
            return surface;
        }
        static SDL_Surface* create_sdl_surface(int width, int height, shs::Color color) 
        {
            SDL_Surface *surface = SDL_CreateRGBSurface(0, width, height, 32, color.r, color.g, color.b, color.a);
            return surface;
        }
        SDL_Surface* create_sdl_surface(shs::Color color) 
        {
            SDL_Surface *surface = SDL_CreateRGBSurface(0, this->width, this->height, 32, color.r, color.g, color.b, color.a);
            return surface;
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

            for (int x = 0; x < this->width; x++)
            {
                for (int y = 0; y < this->height; y++)
                {
                    shs::Color color = canvas[x][y];
                    Uint32 raw_color = SDL_MapRGBA(surface->format, color.r, color.g, color.b, color.a);
                    shs::Canvas::set_rawcolor_at_SDLSurface(surface, x, y, raw_color);
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
        std::vector<std::vector<shs::Color>> canvas;
        int width;
        int height;
    };


    class Camera3D
    {
    public:
        Camera3D()
        {
        }
        ~Camera3D()
        {
        }

        void update()
        {
            this->direction = glm::vec3(
                cos(this->vertical_angle) * sin(this->horizontal_angle),
                sin(this->vertical_angle),
                cos(this->vertical_angle) * cos(this->horizontal_angle)
            );
            glm::vec3 right = glm::vec3(
                sin(horizontal_angle - glm::pi<float>() / 2.0f),
                0.0,
                cos(horizontal_angle - glm::pi<float>() / 2.0f)
            );
            this->up = glm::cross(right, direction);

            this->projection_matrix = glm::perspective(this->field_of_view, this->width / this->height, this->z_near, z_far);
            this->view_matrix       = glm::lookAt(this->position, this->position + this->direction, this->up);
        }

        glm::mat4 view_matrix;
        glm::mat4 projection_matrix;

        glm::vec3 position;
        glm::vec3 direction;
        glm::vec3 up;

        float horizontal_angle;
        float vertical_angle;

        float width;
        float height;

        float field_of_view;
        float z_near;
        float z_far;

    private:

    };
}